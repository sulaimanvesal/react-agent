# ReAct Agent

A runnable Python implementation of **"ReAct: Synergizing Reasoning and Acting in Language Models"** — Yao, Zhao, Yu, Du, Shafran, Narasimhan & Cao (Princeton / Google Research), ICLR 2023 · [arXiv:2210.03629](https://arxiv.org/abs/2210.03629).

Chain-of-Thought reasons but can't touch the outside world; act-only agents gather information but can't plan or recover from mistakes. ReAct **interleaves free-form reasoning traces ("thoughts") with task-specific actions** in a single generation loop:

```
Thought → Action → Observation → Thought → … → Finish[answer]
```

Reasoning helps the model induce, track, and update action plans (and handle exceptions); acting lets it pull fresh evidence from external sources instead of relying on frozen internal knowledge. On HotpotQA the paper shows this grounding overcomes the hallucination and error propagation that plague pure chain-of-thought.

This repo implements the paper's §3 loop with the §4.1 HotpotQA setup: a small Wikipedia-like API (`Search` / `Lookup` / `Finish`) over a bundled multi-hop QA mini-corpus. It runs **fully offline** — no API key needed.

## Setup

```bash
git clone https://github.com/sulaimanvesal/react-agent.git
cd react-agent
pip install -r requirements.txt   # pytest only
```

## Usage

Answer all three bundled multi-hop questions out of the box (offline `MockBackend` stands in for the paper's ReAct generation):

```bash
python examples/run_demo.py                 # all 3 sample questions
python examples/run_demo.py --question 2    # just question 2
```

Run the tests:

```bash
pytest -q
```

### Use it as a library

```python
from react import ReActAgent, MockBackend, OpenAICompatibleBackend

# offline, deterministic
agent = ReActAgent(MockBackend(), max_steps=6)
traj = agent.solve("Which album by Midnight Harvest was produced by Silas Reed?")
print(agent.format_trajectory(traj))
print(traj.final_answer)   # Ember Skies

# with a real model (OpenAI-compatible endpoint)
agent = ReActAgent(OpenAICompatibleBackend())
traj = agent.solve("your question here")
```

To use a real LLM, set `OPENAI_API_KEY` (and optionally `OPENAI_BASE_URL` / `REACT_MODEL`); `OpenAICompatibleBackend` issues the paper's ReAct prompt — system message describing the Thought/Action/Observation loop and the Search/Lookup/Finish action space.

To plug in your own knowledge base, pass a different `articles` dict to `KnowledgeBase` (or your own `Tool` subclasses) — the agent dispatches any `Name[...]` action through the tools mapping.

## Architecture

```mermaid
flowchart TB
    subgraph Agent["ReActAgent (react/agent.py)"]
        P["build_react_prompt()\n§3: instructions + few-shot trajectories"]
        L["solve()\n§3: Thought → Action → Observation loop"]
        F["format_trajectory()"]
    end
    subgraph Backend["LLMBackend (react/llm.py)"]
        Mock["MockBackend\ndeterministic oracle over the corpus"]
        OAI["OpenAICompatibleBackend\nreal ReAct prompt, urllib only"]
    end
    subgraph Tools["Tools (react/tools.py)"]
        S["SearchTool\nSearch[entity] → first 5 sentences"]
        Lk["LookupTool\nLookup[keyword] → next matching sentence"]
    end
    KB["KnowledgeBase\nbundled HotpotQA-style corpus (react/tasks.py)\n§4.1: 6 articles, 3 two-hop questions"]

    P -->|generate\\(\\)| Backend
    L -->|Action\\[...\\]| Tools
    Tools -->|reads| KB
    L -->|Observation| Backend
    L --> F
```

## Paper → code mapping

| Paper (arXiv:2210.03629) | This repo |
|---|---|
| §3 ReAct loop — LM generates Thought and Action interleaved; action output becomes the next Observation | `react/agent.py` — `ReActAgent.solve()`: one `backend.generate()` per turn, dispatch, append observation, repeat |
| §3 Prompting — task description plus one or two in-context ReAct trajectories | `react/agent.py` — `build_react_prompt()`: instructions, action space, two few-shot trajectories over the bundled corpus |
| §4.1 HotpotQA Wikipedia API — `Search[entity]` returns first 5 sentences or similar titles; `Lookup[keyword]` returns the next matching sentence | `react/tools.py` — `KnowledgeBase.search()` / `.lookup()` with the paper's exact semantics; `SearchTool` / `LookupTool` expose them |
| Thought/Action/Observation trace format | `react/agent.py` — `Trajectory` / `Step` dataclasses; `format_trajectory()` renders the paper-style trace |
| `Finish[answer]` action stops the trajectory | `react/agent.py` — `solve()` returns the `Trajectory` with `final_answer` on `Finish` |
| Invalid actions / exceptions handled via reasoning over the observation | `react/agent.py` — unknown actions yield an "Invalid action" observation and the loop continues, like the paper's exception handling |
| Step limit (paper trajectories are bounded) | `react/agent.py` — `max_steps` (default 6); trajectory reports unfinished if hit |

Notes on fidelity: the paper prompts a frozen PaLM-540B; `MockBackend` substitutes a deterministic oracle that parses the question and each observation and emits the next Thought/Action pair the paper's trajectories would contain, so the demo exercises the genuine loop mechanics offline — swap in `OpenAICompatibleBackend` for real LM behavior. The HotpotQA corpus is replaced by a tiny self-written 6-article mini-corpus (all facts are original fiction) with three two-hop questions.

## Project layout

```
react-agent/
├── react/
│   ├── __init__.py      # public API
│   ├── agent.py         # ReActAgent: the §3 Thought → Action → Observation loop
│   ├── llm.py           # LLMBackend, MockBackend, OpenAICompatibleBackend
│   ├── tools.py         # KnowledgeBase + Search/Lookup tools (paper §4.1 API)
│   └── tasks.py         # bundled HotpotQA-style mini-corpus + 3 questions
├── examples/
│   └── run_demo.py      # offline multi-hop QA demo (no API key)
├── tests/
│   └── test_react.py    # 11 tests: loop, tools, parsing, end-to-end answers
├── requirements.txt
├── LICENSE              # MIT
└── README.md
```

## Citation

```bibtex
@inproceedings{yao2023react,
  title={ReAct: Synergizing Reasoning and Acting in Language Models},
  author={Yao, Shunyu and Zhao, Jeffrey and Yu, Dian and Du, Nan
          and Shafran, Izhak and Narasimhan, Karthik and Cao, Yuan},
  booktitle={International Conference on Learning Representations},
  year={2023}
}
```

## License

MIT — see [LICENSE](LICENSE).
