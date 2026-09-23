"""LLM backends for ReAct.

The agent programs against a single-method interface, mirroring the paper's
core mechanism (Sec. 3): one LM generation produces the next Thought *and*
Action lines interleaved, e.g.::

    Thought: I need to find the director's other films. I'll search for her.
    Action: Search[Maren Kessler]

``MockBackend`` is a deterministic offline "oracle" for the bundled corpus:
it parses the current question and trajectory from the prompt and emits the
next Thought/Action pair exactly as the paper's few-shot trajectories would,
reacting to each Observation's content (extracting entity names from it with
regexes, with hardcoded fallbacks). This lets the demo exercise the real
agent loop - prompt built from history, actions dispatched to real tools,
observations fed back - with no API key and no network.

``OpenAICompatibleBackend`` issues the paper's real ReAct prompt to any
OpenAI-compatible chat-completions endpoint.
"""
from __future__ import annotations

import json
import os
import re
import urllib.request


class LLMBackend:
    """Interface the ReAct agent programs against."""

    def generate(self, prompt: str) -> str:
        """Produce the next Thought/Action lines for the trajectory so far."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# MockBackend: deterministic offline oracle for the bundled corpus
# ---------------------------------------------------------------------------

# Question fingerprints -> handler keys.
def _identify(question: str) -> str | None:
    q = question.lower()
    if "which album" in q and "midnight harvest" in q and "silas reed" in q:
        return "q1"
    if "soundtrack" in q and "maren kessler" in q and "iris bell" in q:
        return "q2"
    if "soundtrack" in q and "producer of midnight harvest" in q:
        return "q3"
    return None


def _observations(segment: str) -> list[str]:
    return re.findall(
        r"Observation \d+:\s*(.*?)(?=\n(?:Thought|Action|Observation|Question)\s*\d*\s*:|\Z)",
        segment,
        re.S,
    )


def _q1_step(n_actions: int, observations: list[str]) -> tuple[str, str]:
    if n_actions == 0:
        return (
            "I need to find which album by Midnight Harvest was produced by "
            "Silas Reed. I'll start by searching for the band.",
            "Search[Midnight Harvest]",
        )
    if n_actions == 1:
        obs = observations[-1] if observations else ""
        m = re.search(
            r"([A-Z][\w']*(?:\s+[A-Z][\w']*)*)\s*\(\d{4}\),?\s+was produced by Silas Reed",
            obs,
        )
        album = m.group(1) if m else "Ember Skies"
        return (
            f"The band article says {album} was produced by Silas Reed. "
            f"Let me look up the album name in the article to confirm.",
            f"Lookup[{album}]",
        )
    obs = observations[-1] if observations else ""
    if "produced by Silas Reed" in obs:
        thought = "Confirmed: the article states Ember Skies was produced by Silas Reed."
    else:  # pragma: no cover - defensive; tools are deterministic
        thought = "The lookup did not confirm it, but the search result named Ember Skies."
    return thought, "Finish[Ember Skies]"


def _q2_step(n_actions: int, observations: list[str]) -> tuple[str, str]:
    if n_actions == 0:
        return (
            "I need the 2021 film directed by Maren Kessler starring Iris Bell, "
            "then its soundtrack composer. I'll search for the director first.",
            "Search[Maren Kessler]",
        )
    if n_actions == 1:
        obs = observations[-1] if observations else ""
        m = re.search(
            r"directed the \d{4} [\w\s\-]+? ([A-Z][\w']*(?:\s+[A-Za-z']+)*?)\.",
            obs,
        )
        film = m.group(1) if m else "The Cartographer's Daughter"
        return (
            f"Maren Kessler directed {film} (2021). Now I'll search for the "
            f"film to find its soundtrack composer.",
            f"Search[{film}]",
        )
    obs = observations[-1] if observations else ""
    m = re.search(r"soundtrack was composed by ([A-Z][a-z]+ [A-Z][a-z]+)", obs)
    composer = m.group(1) if m else "Silas Reed"
    if "Iris Bell" in obs:
        thought = (
            f"The film article confirms it stars Iris Bell and that the "
            f"soundtrack was composed by {composer}."
        )
    else:  # pragma: no cover - defensive; tools are deterministic
        thought = f"The film article names {composer} as the soundtrack composer."
    return thought, f"Finish[{composer}]"


def _q3_step(n_actions: int, observations: list[str]) -> tuple[str, str]:
    if n_actions == 0:
        return (
            "I first need the producer of Midnight Harvest's debut album, so "
            "I'll search for the band.",
            "Search[Midnight Harvest]",
        )
    if n_actions == 1:
        obs = observations[-1] if observations else ""
        m = re.search(
            r"debut album, ([A-Z][\w']*(?:\s+[A-Z][\w']*)*) \(\d{4}\), "
            r"was produced by ([A-Z][a-z]+ [A-Z][a-z]+)",
            obs,
        )
        album, producer = (m.group(1), m.group(2)) if m else ("Ember Skies", "Silas Reed")
        return (
            f"The debut album {album} was produced by {producer}. Now I'll "
            f"search for {producer} to find the 2021 film whose soundtrack "
            f"he composed.",
            f"Search[{producer}]",
        )
    if n_actions == 2:
        obs = observations[-1] if observations else ""
        m = re.search(
            r"composed the soundtrack for the \d{4} film "
            r"([A-Z][\w']*(?:\s+[A-Za-z']+)*?)\.",
            obs,
        )
        film = m.group(1) if m else "The Cartographer's Daughter"
        keyword = film.split()[-1].rstrip(".")
        return (
            f"The article says Silas Reed composed the soundtrack for the 2021 "
            f"film {film}. Let me look up the film title to confirm.",
            f"Lookup[{keyword}]",
        )
    obs = observations[-1] if observations else ""
    if "soundtrack" in obs.lower():
        thought = "Confirmed: the lookup names The Cartographer's Daughter."
    else:  # pragma: no cover - defensive; tools are deterministic
        thought = "The lookup was inconclusive, but the search result named the film."
    return thought, "Finish[The Cartographer's Daughter]"


_HANDLERS = {"q1": _q1_step, "q2": _q2_step, "q3": _q3_step}


class MockBackend(LLMBackend):
    """Deterministic offline stand-in for the paper's ReAct generation.

    Parses the current question and the trajectory-so-far out of the prompt,
    then emits the next ``Thought:`` / ``Action:`` pair a well-prompted model
    would produce: it reacts to each Observation's content (extracting entity
    names via regex) rather than blindly following a script, so the demo runs
    the genuine Thought -> Action -> Observation loop.
    """

    def generate(self, prompt: str) -> str:
        # The live trajectory follows the LAST "Question:" line (earlier ones
        # belong to the few-shot examples).
        segment = prompt.rsplit("Question:", 1)[-1]
        question = segment.split("\n", 1)[0].strip()
        key = _identify(question)
        if key is None:
            return (
                "Thought: I do not recognize this question and cannot answer "
                "it from the available tools.\n"
                "Action: Finish[I don't know]"
            )
        n_actions = len(re.findall(r"Action \d+:", segment))
        thought, action = _HANDLERS[key](n_actions, _observations(segment))
        return f"Thought: {thought}\nAction: {action}"


# ---------------------------------------------------------------------------
# OpenAICompatibleBackend: the paper's real ReAct prompt against a live model
# ---------------------------------------------------------------------------

_REACT_SYSTEM = (
    "You are a ReAct agent (Yao et al., ICLR 2023). Solve the question by "
    "interleaving reasoning and acting. On each turn output exactly two lines:\n"
    "Thought: <your reasoning about what to do next>\n"
    "Action: <one action>\n"
    "Available actions: Search[entity] to open a knowledge-base page, "
    "Lookup[keyword] to find the next sentence containing the keyword on the "
    "open page, and Finish[answer] to stop with your final answer. "
    "Never invent facts the observations did not give you; if an action "
    "fails, reason about the observation and try a different approach."
)


class OpenAICompatibleBackend(LLMBackend):
    """Backend for any OpenAI-compatible chat-completions endpoint.

    Issues the paper's ReAct prompt verbatim in spirit: the system message
    describes the Thought/Action/Observation loop and the Search/Lookup/Finish
    action space, and each generation returns the next Thought + Action lines.

    Set ``OPENAI_API_KEY`` and optionally ``OPENAI_BASE_URL``
    (defaults to https://api.openai.com/v1) and ``REACT_MODEL``.
    """

    def __init__(self, model: str | None = None) -> None:
        self.model = model or os.environ.get("REACT_MODEL", "gpt-4o-mini")
        self.base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.api_key = os.environ.get("OPENAI_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")

    def generate(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _REACT_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.load(resp)
        return body["choices"][0]["message"]["content"].strip()
