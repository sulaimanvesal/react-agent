"""The ReAct agent: the paper's Sec. 3 Thought -> Action -> Observation loop.

Paper Sec. 3: the LM is prompted with a task description plus a few
in-context ReAct trajectories, then generates Thought and Action lines
interleaved. Each action is dispatched to an external tool; the tool's output
becomes the next Observation, which is appended to the context. The loop
stops when the model emits ``Finish[answer]``.

``build_react_prompt`` assembles the paper-style prompt: instructions, the
action space, two few-shot trajectories written against the bundled corpus
(paper: "only one or two in-context examples"), then the live question.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .llm import LLMBackend
from .tasks import ARTICLES
from .tools import KnowledgeBase, LookupTool, SearchTool, Tool

ACTION_RE = re.compile(r"(\w+)\s*\[(.*)\]")
FINISH = "Finish"


@dataclass
class Step:
    """One loop iteration: the thought, the parsed action, the observation."""

    thought: str
    action: str       # e.g. "Search", "Lookup", "Finish", or "<invalid>"
    argument: str     # the text inside [...]
    observation: str = ""


@dataclass
class Trajectory:
    """The full paper-style trace for one question."""

    question: str
    steps: list[Step] = field(default_factory=list)
    final_answer: str = ""
    finished: bool = False


# -- paper Sec. 3: the few-shot ReAct prompt --------------------------------
# Two in-context trajectories over the bundled corpus (different questions
# from the demo set, so the agent generalizes the pattern, not the answers).

_FEWSHOTS = """\
Question: Who produced Midnight Harvest's second album?
Thought 1: I need the producer of the band's second album. I'll search for the band first.
Action 1: Search[Midnight Harvest]
Observation 1: Page: Midnight Harvest
Midnight Harvest is an American indie rock band formed in Portland, Oregon in 2015. The band consists of vocalist June Park and guitarist Theo Marsh. Their debut album, Ember Skies (2019), was produced by Silas Reed, while their second album, Lantern Light (2022), was produced by Dana Whitfield. Ember Skies won the 2020 Alder Award for Best Rock Album. The band is known for blending folk harmonies with distorted guitars.
Thought 2: The second album is Lantern Light (2022), produced by Dana Whitfield. Let me look it up to confirm.
Action 2: Lookup[Lantern Light]
Observation 2: Their debut album, Ember Skies (2019), was produced by Silas Reed, while their second album, Lantern Light (2022), was produced by Dana Whitfield.
Thought 3: Confirmed: Lantern Light was produced by Dana Whitfield.
Action 3: Finish[Dana Whitfield]

Question: Which director frequently collaborates with composer Silas Reed?
Thought 1: I need the director who frequently collaborates with Silas Reed. I'll search for him.
Action 1: Search[Silas Reed]
Observation 1: Page: Silas Reed
Silas Reed is an American record producer and composer born in 1978. He produced the 2019 debut album Ember Skies by Midnight Harvest. He also composed the soundtrack for the 2021 film The Cartographer's Daughter. Reed is a frequent collaborator of director Maren Kessler. His production style blends analog tape recording with electronic textures.
Thought 2: The article directly names director Maren Kessler as a frequent collaborator.
Action 2: Finish[Maren Kessler]
"""


def build_react_prompt(question: str) -> str:
    """Paper Sec. 3 prompt: instructions + few-shot trajectories + question."""
    return (
        "Solve the question by interleaving Thought, Action, and Observation "
        "steps, exactly like the examples below.\n\n"
        "Available actions:\n"
        "- Search[entity]: open the knowledge-base page for the entity and "
        "see its first sentences.\n"
        "- Lookup[keyword]: find the next sentence containing the keyword on "
        "the currently open page.\n"
        "- Finish[answer]: stop and give the final answer.\n\n"
        "On each turn output exactly two lines:\n"
        "Thought: <your reasoning about what to do next>\n"
        "Action: <one action, e.g. Search[Midnight Harvest]>\n\n"
        "Here are some examples:\n\n"
        f"{_FEWSHOTS}\n"
        f"Question: {question}\n"
    )


def parse_generation(text: str) -> tuple[str, str, str]:
    """Split a generation into (thought, action_name, action_argument).

    Returns action_name "" when no ``Name[...]`` action is present, so the
    caller can report it as an invalid action instead of crashing.
    """
    thought = ""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("thought"):
            thought = re.sub(r"(?i)^thought\s*\d*\s*:\s*", "", stripped)
            break
    match = ACTION_RE.search(text)
    if match:
        return thought, match.group(1), match.group(2).strip()
    return thought or text.strip(), "", ""


class ReActAgent:
    """Paper Sec. 3: the Thought -> Action -> Observation loop."""

    def __init__(
        self,
        backend: LLMBackend,
        tools: dict[str, Tool] | None = None,
        max_steps: int = 6,
    ) -> None:
        self.backend = backend
        self.max_steps = max_steps
        if tools is None:
            kb = KnowledgeBase(ARTICLES)  # shared: Lookup sees Search's page
            tools = {"Search": SearchTool(kb), "Lookup": LookupTool(kb)}
        self.tools = tools

    @property
    def valid_actions(self) -> str:
        names = sorted(self.tools) + [FINISH]
        return ", ".join(f"{n}[...]" for n in names)

    def solve(self, question: str) -> Trajectory:
        """Run the ReAct loop; stop on Finish[answer] or max_steps."""
        trajectory = Trajectory(question=question)
        prompt = build_react_prompt(question)

        for step_no in range(1, self.max_steps + 1):
            response = self.backend.generate(prompt)
            thought, action_name, argument = parse_generation(response)

            if action_name == FINISH:
                trajectory.steps.append(
                    Step(thought=thought, action=FINISH, argument=argument)
                )
                trajectory.final_answer = argument
                trajectory.finished = True
                return trajectory

            tool = self.tools.get(action_name)
            if tool is None:
                # Paper Sec. 3: reasoning handles exceptions - the model sees
                # the invalid-action observation and can recover next turn.
                observation = (
                    f"Invalid action: {action_name}[{argument}]. "
                    f"Valid actions are: {self.valid_actions}."
                )
                label = action_name or "<none>"
            else:
                observation = tool.call(argument)
                label = action_name

            trajectory.steps.append(
                Step(thought=thought, action=label, argument=argument,
                     observation=observation)
            )
            prompt += (
                f"Thought {step_no}: {thought}\n"
                f"Action {step_no}: {label}[{argument}]\n"
                f"Observation {step_no}: {observation}\n"
            )

        return trajectory  # max_steps hit without Finish

    @staticmethod
    def format_trajectory(trajectory: Trajectory) -> str:
        """Render a paper-style Thought/Action/Observation trace."""
        lines = [f"Question: {trajectory.question}"]
        for i, step in enumerate(trajectory.steps, 1):
            lines.append(f"Thought {i}: {step.thought}")
            lines.append(f"Action {i}: {step.action}[{step.argument}]")
            if step.action != FINISH:
                lines.append(f"Observation {i}: {step.observation}")
        if trajectory.finished:
            lines.append(f"Answer: {trajectory.final_answer}")
        else:
            lines.append(
                f"(stopped after {len(trajectory.steps)} steps without Finish)"
            )
        return "\n".join(lines)
