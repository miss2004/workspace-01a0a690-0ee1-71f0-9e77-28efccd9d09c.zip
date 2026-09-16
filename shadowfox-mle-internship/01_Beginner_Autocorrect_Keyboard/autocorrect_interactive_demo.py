"""
Interactive Keyboard Simulator Demo
-----------------------------------
Simulates a real-time mobile keyboard suggestion bar.
Usage:
    python autocorrect_interactive_demo.py
"""

import os
import sys
from autocorrect_engine import AutocorrectKeyboardEngine


def run_interactive_demo():
    corpus_dir = os.path.join(os.path.dirname(__file__), "corpus")
    corpora = [
        os.path.join(corpus_dir, "sherlock_holmes.txt"),
        os.path.join(corpus_dir, "pride_and_prejudice.txt")
    ]
    print("=" * 65)
    print("      AUTOCORRECT KEYBOARD & NEXT-WORD ANTICIPATION DEMO")
    print("=" * 65)
    print("Loading language models and vocabulary corpus...")
    engine = AutocorrectKeyboardEngine(corpus_paths=corpora)
    print(f"Engine Ready! Vocabulary size: {len(engine.lm.vocab):,} unique words.")
    print("\nHow it works:")
    print(" - Type words. When a word is misspelled, top 3 corrections appear.")
    print(" - Add a trailing SPACE to trigger next-word contextual anticipation.")
    print(" - Type 'quit' or 'exit' to finish.\n")

    test_phrases = [
        "sherlock holmes ",
        "the quik brown fox",
        "in my opinon ",
        "definately not a problem",
        "i have recived your"
    ]

    print("Running preset phrase simulations:")
    print("-" * 65)
    for phrase in test_phrases:
        state = engine.process_keystroke_state(phrase, top_k=3)
        mode = state["mode"]
        sugs = state["suggestions"]
        scores = state.get("scores", [])
        score_str = ", ".join([f"{sug} ({sc:.2f})" for sug, sc in zip(sugs, scores)]) if scores else ", ".join(sugs)
        print(f"Input: '{phrase}'")
        print(f"  -> Mode: [{mode.upper()}] | Suggestions: {score_str}\n")

    print("-" * 65)
    print("Interactive Mode: (Enter any sentence or unfinished word)")
    while True:
        try:
            user_input = input("keyboard> ")
            if user_input.strip().lower() in ["quit", "exit"]:
                print("Exiting demo.")
                break
            if not user_input:
                continue

            state = engine.process_keystroke_state(user_input, top_k=3)
            mode = state["mode"]
            sugs = state["suggestions"]
            scores = state.get("scores", [])
            print(f"  [{mode}] -> " + " | ".join([f"[{i+1}] {w}" for i, w in enumerate(sugs)]))
            if scores:
                print(f"  Probabilities: " + ", ".join([f"{w}: {sc:.3f}" for w, sc in zip(sugs, scores)]))
            print()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting demo.")
            break


if __name__ == "__main__":
    run_interactive_demo()
