"""Tests for the ReAct implementation.

All tests run offline against MockBackend (deterministic). They verify the
paper's Sec. 3 loop: termination on Finish, graceful handling of invalid
actions, the Search/Lookup Wikipedia API, the max_steps stop, trajectory
parsing, and end-to-end success on all three demo questions.
"""
import pytest

from react import (
    ARTICLES,
    KnowledgeBase,
    LookupTool,
    MockBackend,
    ReActAgent,
    SAMPLE_QUESTIONS,
    SearchTool,
    build_react_prompt,
    parse_generation,
)


@pytest.fixture()
def agent():
    return ReActAgent(MockBackend(), max_steps=6)


def test_loop_terminates_on_finish(agent):
    item = SAMPLE_QUESTIONS[0]
    traj = agent.solve(item["question"])
    assert traj.finished
    assert traj.final_answer == item["gold"]
    assert traj.steps[-1].action == "Finish"


def test_unknown_action_handled_gracefully():
    class RogueBackend:
        def __init__(self):
            self.calls = 0

        def generate(self, prompt):
            self.calls += 1
            if self.calls == 1:
                return "Thought: Let me try something creative.\nAction: Dance[now]"
            return "Thought: That failed; I'll answer directly.\nAction: Finish[Ember Skies]"

    agent = ReActAgent(RogueBackend(), max_steps=4)
    traj = agent.solve(SAMPLE_QUESTIONS[0]["question"])
    assert traj.finished
    assert "Invalid action" in traj.steps[0].observation
    assert "Dance" in traj.steps[0].observation
    assert traj.final_answer == "Ember Skies"


def test_malformed_generation_reports_invalid_action():
    thought, name, arg = parse_generation("just some rambling text")
    assert name == "" and arg == ""
    agent = ReActAgent(MockBackend(), max_steps=2)
    # a backend emitting no Action[...] at all: loop must not crash
    agent.backend = type("B", (), {"generate": lambda self, p: "hmm"})()
    traj = agent.solve("Which album by Midnight Harvest was produced by Silas Reed?")
    assert not traj.finished
    assert len(traj.steps) == 2
    assert all("Invalid action" in s.observation for s in traj.steps)


def test_search_exact_match_returns_page():
    kb = KnowledgeBase(ARTICLES)
    obs = SearchTool(kb).call("Maren Kessler")
    assert obs.startswith("Page: Maren Kessler")
    assert "The Cartographer's Daughter" in obs


def test_search_miss_suggests_similar_titles():
    kb = KnowledgeBase(ARTICLES)
    obs = SearchTool(kb).call("Maren Kesslr")  # typo: no exact page
    assert "Could not find" in obs
    assert "Maren Kessler" in obs  # suggested as similar


def test_lookup_returns_matching_sentence_and_advances():
    kb = KnowledgeBase(ARTICLES)
    search, lookup = SearchTool(kb), LookupTool(kb)
    search.call("Silas Reed")
    first = lookup.call("film")
    assert "Cartographer's Daughter" in first
    # only one sentence mentions "film": the next call reports exhaustion
    assert "No more sentences" in lookup.call("film")


def test_lookup_without_search_reports_no_page():
    kb = KnowledgeBase(ARTICLES)
    assert "No page is open" in LookupTool(kb).call("anything")


def test_max_steps_stops_loop():
    class NeverFinishes:
        def generate(self, prompt):
            return "Thought: still thinking.\nAction: Search[Midnight Harvest]"

    agent = ReActAgent(NeverFinishes(), max_steps=3)
    traj = agent.solve("Which album by Midnight Harvest was produced by Silas Reed?")
    assert not traj.finished
    assert traj.final_answer == ""
    assert len(traj.steps) == 3


def test_trajectory_steps_parse_correctly(agent):
    traj = agent.solve(SAMPLE_QUESTIONS[1]["question"])
    assert traj.finished
    actions = [(s.action, s.argument) for s in traj.steps]
    assert actions[0] == ("Search", "Maren Kessler")
    assert actions[1][0] == "Search" and "Cartographer" in actions[1][1]
    assert actions[-1] == ("Finish", "Silas Reed")
    for step in traj.steps[:-1]:
        assert step.observation, "every non-Finish step needs an observation"
    assert all(s.thought for s in traj.steps)


def test_all_demo_questions_reach_gold_answers(agent):
    for item in SAMPLE_QUESTIONS:
        traj = agent.solve(item["question"])
        assert traj.finished, f"no Finish for: {item['question']}"
        assert traj.final_answer.strip().lower() == item["gold"].lower(), (
            f"wrong answer for: {item['question']}"
        )


def test_prompt_contains_fewshot_trajectories():
    prompt = build_react_prompt("Which album by Midnight Harvest was produced by Silas Reed?")
    assert "Search[entity]" in prompt
    assert "Finish[answer]" in prompt
    assert prompt.count("Question:") >= 3  # 2 few-shots + the live question
    assert "Finish[Dana Whitfield]" in prompt  # few-shot answer present
