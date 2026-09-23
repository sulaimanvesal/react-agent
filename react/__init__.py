"""ReAct: Synergizing Reasoning and Acting in Language Models.

A runnable Python implementation of Yao et al., "ReAct: Synergizing Reasoning
and Acting in Language Models" (ICLR 2023, arXiv:2210.03629).

ReAct interleaves free-form reasoning traces ("thoughts") with task-specific
actions (tool calls) in a single generation loop:

    Thought -> Action -> Observation -> Thought -> ... -> Finish[answer]

Reasoning helps the model induce, track, and update action plans; acting lets
it gather fresh information from external sources (here, a small Wikipedia-like
API over a bundled HotpotQA-style corpus, mirroring the paper's Sec. 4.1)
instead of relying only on frozen internal knowledge.
"""

from .agent import ReActAgent, Step, Trajectory, build_react_prompt, parse_generation
from .llm import LLMBackend, MockBackend, OpenAICompatibleBackend
from .tasks import ARTICLES, SAMPLE_QUESTIONS
from .tools import KnowledgeBase, LookupTool, SearchTool, Tool

__all__ = [
    "ReActAgent",
    "Step",
    "Trajectory",
    "build_react_prompt",
    "parse_generation",
    "LLMBackend",
    "MockBackend",
    "OpenAICompatibleBackend",
    "Tool",
    "SearchTool",
    "LookupTool",
    "KnowledgeBase",
    "ARTICLES",
    "SAMPLE_QUESTIONS",
]

__version__ = "0.1.0"
