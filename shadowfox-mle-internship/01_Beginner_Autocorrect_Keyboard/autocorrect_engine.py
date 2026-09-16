"""
Autocorrect Keyboard System & Next-Word Context Engine
------------------------------------------------------
A production-grade dual engine combining:
1. Noisy Channel Model & Damerau-Levenshtein Edit Distance for spelling correction.
2. N-Gram Markov Language Model (Trigram + Bigram with Laplace smoothing & backoff)
   for next-word anticipation and context-sensitive candidate re-ranking.
"""

import os
import re
import math
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Optional, Set
from spellchecker import SpellChecker


def tokenize(text: str) -> List[str]:
    """Tokenize lowercase alphabetical words."""
    return re.findall(r"\b[a-z]+(?:'[a-z]+)?\b", text.lower())


class NGramLanguageModel:
    """
    Trigram language model with Bigram and Unigram backoff and additive smoothing.
    P(w_i | w_{i-2}, w_{i-1}) = (Count(w_{i-2}, w_{i-1}, w_i) + alpha) /
                               (Count(w_{i-2}, w_{i-1}) + alpha * |V|)
    """
    def __init__(self, alpha: float = 0.01, backoff_weight: float = 0.4):
        self.alpha = alpha
        self.backoff_weight = backoff_weight
        self.unigrams = Counter()
        self.bigrams = Counter()
        self.trigrams = Counter()
        self.context_bigrams = defaultdict(Counter)  # (w1, w2) -> Counter({w3: count})
        self.context_unigrams = defaultdict(Counter) # w1 -> Counter({w2: count})
        self.vocab: Set[str] = set()
        self.total_words = 0

    def fit(self, tokens: List[str]):
        """Train language model on tokenized text sequence."""
        self.unigrams.update(tokens)
        self.vocab.update(tokens)
        self.total_words += len(tokens)

        # Build bigrams and trigrams
        for i in range(len(tokens) - 1):
            w1, w2 = tokens[i], tokens[i+1]
            self.bigrams[(w1, w2)] += 1
            self.context_unigrams[w1][w2] += 1

        for i in range(len(tokens) - 2):
            w1, w2, w3 = tokens[i], tokens[i+1], tokens[i+2]
            self.trigrams[(w1, w2, w3)] += 1
            self.context_bigrams[(w1, w2)][w3] += 1

    def score_unigram(self, word: str) -> float:
        """P(w) with Laplace smoothing."""
        v_size = max(len(self.vocab), 1)
        count = self.unigrams[word]
        return (count + self.alpha) / (self.total_words + self.alpha * v_size)

    def score_bigram(self, w1: str, w2: str) -> float:
        """P(w2 | w1) with backoff to unigram."""
        w1_count = self.unigrams[w1]
        if w1_count > 0:
            count = self.bigrams.get((w1, w2), 0)
            return (count + self.alpha) / (w1_count + self.alpha * len(self.vocab))
        return self.backoff_weight * self.score_unigram(w2)

    def score_trigram(self, w1: str, w2: str, w3: str) -> float:
        """P(w3 | w1, w2) with backoff to bigram and unigram."""
        w1_w2_count = self.bigrams.get((w1, w2), 0)
        if w1_w2_count > 0:
            count = self.trigrams.get((w1, w2, w3), 0)
            return (count + self.alpha) / (w1_w2_count + self.alpha * len(self.vocab))
        return self.backoff_weight * self.score_bigram(w2, w3)

    def predict_next_words(self, context_tokens: List[str], top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Anticipates the most likely next words given preceding context tokens.
        Falls back from trigram -> bigram -> unigram gracefully.
        """
        candidates = Counter()
        cleaned_context = [w.lower() for w in context_tokens if w.isalpha()]

        if len(cleaned_context) >= 2:
            w1, w2 = cleaned_context[-2], cleaned_context[-1]
            if (w1, w2) in self.context_bigrams:
                for w3, count in self.context_bigrams[(w1, w2)].most_common(top_k * 3):
                    prob = self.score_trigram(w1, w2, w3)
                    candidates[w3] = max(candidates[w3], prob)

        if len(candidates) < top_k and len(cleaned_context) >= 1:
            w2 = cleaned_context[-1]
            if w2 in self.context_unigrams:
                for w3, count in self.context_unigrams[w2].most_common(top_k * 3):
                    prob = self.score_bigram(w2, w3)
                    candidates[w3] = max(candidates[w3], prob)

        # If still insufficient, fill with top frequency unigrams
        if len(candidates) < top_k:
            for w, _ in self.unigrams.most_common(top_k * 2):
                if w not in candidates:
                    candidates[w] = self.score_unigram(w) * 0.1
                if len(candidates) >= top_k:
                    break

        return candidates.most_common(top_k)


class AutocorrectKeyboardEngine:
    """
    Unified keyboard prediction and autocorrection engine.
    Integrates:
    - Edit distance 1 & 2 candidate generation (Damerau-Levenshtein)
    - SpellChecker dictionary (1.6B word unigram prior)
    - Trigram/Bigram language model for contextual candidate disambiguation
    - Prefix-based word completion
    """
    def __init__(self, corpus_paths: Optional[List[str]] = None, alpha: float = 0.01):
        self.lm = NGramLanguageModel(alpha=alpha)
        self.spell = SpellChecker(distance=2)
        self.alphabet = "abcdefghijklmnopqrstuvwxyz"

        if corpus_paths:
            for path in corpus_paths:
                if os.path.exists(path):
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read()
                        tokens = tokenize(text)
                        self.lm.fit(tokens)
                        self.spell.word_frequency.load_words(tokens)

    def _edits1(self, word: str) -> Set[str]:
        """Generate all edits that are one edit distance away."""
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [L + R[1:] for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces = [L + c + R[1:] for L, R in splits if R for c in self.alphabet]
        inserts = [L + c + R for L, R in splits for c in self.alphabet]
        return set(deletes + transposes + replaces + inserts)

    def _edits2(self, word: str) -> Set[str]:
        """Generate all edits that are two edits away."""
        return set(e2 for e1 in self._edits1(word) for e2 in self._edits1(e1))

    def known(self, words: Set[str]) -> Set[str]:
        """Filter words that exist in the dictionary or corpus."""
        return set(w for w in words if w in self.lm.vocab or w in self.spell)

    def is_valid_word(self, word: str) -> bool:
        """Check if a word is recognized as valid English."""
        clean = word.lower().strip()
        return clean in self.lm.vocab or clean in self.spell

    def suggest_corrections(
        self,
        word: str,
        context_words: Optional[List[str]] = None,
        top_k: int = 3
    ) -> List[Tuple[str, float]]:
        """
        Suggest ranked spelling corrections for an input word.
        Uses Bayesian noisy-channel ranking:
        Score(c) = log P(w | c) + log P_lex(c) + beta * log P(c | context)
        """
        clean = word.lower().strip()
        if not clean or not clean.isalpha():
            return [(word, 1.0)]

        # Candidate collection
        candidates: Set[str] = set()
        if self.is_valid_word(clean):
            candidates.add(clean)

        e1_candidates = self.known(self._edits1(clean))
        if e1_candidates:
            candidates.update(e1_candidates)
        else:
            e2_candidates = self.known(self._edits2(clean))
            if e2_candidates:
                candidates.update(list(e2_candidates)[:30])

        spell_cands = self.spell.candidates(clean)
        if spell_cands:
            candidates.update(spell_cands)

        if not candidates:
            return [(clean, 0.0)]

        total_words_dict = max(self.spell.word_frequency.total_words, 1000000)
        scored: List[Tuple[str, float]] = []

        for cand in candidates:
            # 1. Edit distance penalty (Error model P(w|c))
            if cand == clean:
                edit_penalty = 0.0
            elif cand in self._edits1(clean):
                edit_penalty = -1.5
            else:
                edit_penalty = -3.5

            # 2. Lexical prior P_lex(c)
            freq = self.spell.word_frequency[cand]
            if freq == 0 and cand in self.lm.vocab:
                freq = self.lm.unigrams[cand] * 10
            p_lex = (freq + 1.0) / (total_words_dict + 500000.0)
            log_lex = math.log10(p_lex)

            # 3. Contextual transition P(c | context)
            log_context = 0.0
            if context_words:
                c_tokens = [w for w in context_words if w.isalpha()]
                if len(c_tokens) >= 2:
                    p_ctx = self.lm.score_trigram(c_tokens[-2], c_tokens[-1], cand)
                    log_context = math.log10(max(p_ctx, 1e-7))
                elif len(c_tokens) >= 1:
                    p_ctx = self.lm.score_bigram(c_tokens[-1], cand)
                    log_context = math.log10(max(p_ctx, 1e-7))

            # Combined log score (Noisy channel with context prior)
            # If context was present, give it weight 0.5
            if context_words and len(context_words) > 0:
                combined_score = edit_penalty + 0.6 * log_lex + 0.4 * log_context
            else:
                combined_score = edit_penalty + log_lex

            scored.append((cand, combined_score))

        # Sort by log score descending
        scored.sort(key=lambda x: x[1], reverse=True)
        # Normalize top-k scores into pseudo-probabilities via softmax for clean presentation
        top_scored = scored[:top_k]
        max_s = top_scored[0][1] if top_scored else 0.0
        exp_scores = [math.exp(s - max_s) for _, s in top_scored]
        sum_exp = sum(exp_scores) if exp_scores else 1.0
        normalized = [(c, round(es / sum_exp, 4)) for (c, _), es in zip(top_scored, exp_scores)]
        return normalized

    def autocomplete_prefix(self, prefix: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Suggest completions for an incomplete word prefix."""
        pref = prefix.lower().strip()
        if not pref:
            return []
        matches = [w for w in self.lm.vocab if w.startswith(pref)]
        if not matches:
            # Check spell dictionary
            matches = [w for w in self.spell.word_frequency if w.startswith(pref)]
            matches = matches[:50]

        scored = [(w, self.spell.word_frequency[w]) for w in matches]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [(w, float(c)) for w, c in scored[:top_k]]

    def process_keystroke_state(self, full_buffer: str, top_k: int = 3) -> Dict:
        """
        Simulates keyboard suggestion bar state.
        - If buffer ends with space -> Next-word prediction.
        - If buffer ends with letters -> In-word spellcheck / autocomplete suggestions.
        """
        if not full_buffer.strip():
            return {
                "mode": "idle",
                "current_word": "",
                "suggestions": [w for w, _ in self.lm.unigrams.most_common(top_k)],
                "is_misspelled": False
            }

        ends_with_space = full_buffer.endswith(" ")
        tokens = tokenize(full_buffer)

        if ends_with_space:
            # Word was just finished: Next-Word Anticipation
            next_words = self.lm.predict_next_words(tokens, top_k=top_k)
            return {
                "mode": "next_word_prediction",
                "context": tokens[-2:] if len(tokens) >= 2 else tokens,
                "suggestions": [w for w, _ in next_words],
                "scores": [round(s, 5) for _, s in next_words],
                "is_misspelled": False
            }
        else:
            # Current word in progress or typed without trailing space
            current_word = tokens[-1] if tokens else ""
            context = tokens[:-1]
            is_valid = self.is_valid_word(current_word)

            if is_valid:
                # Word is valid: give prefix extensions or self
                completions = self.autocomplete_prefix(current_word, top_k=top_k)
                return {
                    "mode": "prefix_autocomplete",
                    "current_word": current_word,
                    "suggestions": [w for w, _ in completions] if completions else [current_word],
                    "is_misspelled": False
                }
            else:
                # Misspelled word: give autocorrect suggestions
                corrections = self.suggest_corrections(current_word, context_words=context, top_k=top_k)
                return {
                    "mode": "autocorrect",
                    "current_word": current_word,
                    "suggestions": [w for w, _ in corrections],
                    "scores": [round(s, 5) for _, s in corrections],
                    "is_misspelled": True
                }
