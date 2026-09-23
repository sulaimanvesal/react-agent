"""Offline demo: answer multi-hop questions with ReAct.

Runs entirely without an API key using MockBackend, which stands in for the
paper's ReAct generation with a deterministic oracle over the bundled
HotpotQA-style corpus. Shows the full Thought -> Action -> Observation
trajectory for each question, then the final answer.

Usage:
    python examples/run_demo.py                # all 3 sample questions
    python examples/run_demo.py --question 2   # just question 2
"""
from __future__ import annotations

import argparse
import sys

sys.path.insert(0, ".")

from react import MockBackend, ReActAgent, SAMPLE_QUESTIONS


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ReAct demo: multi-hop QA over a bundled mini-corpus"
    )
    parser.add_argument(
        "--question", type=int, choices=[1, 2, 3], default=None,
        help="run only this sample question (1-3); default runs all",
    )
    args = parser.parse_args()

    questions = SAMPLE_QUESTIONS
    if args.question is not None:
        questions = [SAMPLE_QUESTIONS[args.question - 1]]

    agent = ReActAgent(MockBackend(), max_steps=6)
    correct = 0
    for i, item in enumerate(questions, 1):
        idx = SAMPLE_QUESTIONS.index(item) + 1
        print(f"===== Question {idx} =====")
        trajectory = agent.solve(item["question"])
        print(agent.format_trajectory(trajectory))
        ok = trajectory.final_answer.strip().lower() == item["gold"].lower()
        correct += ok
        print(f"\nGold: {item['gold']} | Got: {trajectory.final_answer} "
              f"| {'CORRECT' if ok else 'WRONG'}\n")

    print(f"Score: {correct}/{len(questions)}")


if __name__ == "__main__":
    main()
