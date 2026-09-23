"""Tools: the paper's simple Wikipedia API (Sec. 4.1).

The ReAct paper gives its HotpotQA agent three actions backed by a Wikipedia
API:

  * ``Search[entity]`` - if the entity exists, return the first 5 sentences
    of its page; otherwise return up to 5 similar entity names.
  * ``Lookup[keyword]`` - return the next sentence in the current page that
    contains the keyword (each call advances through the page).
  * ``Finish[answer]`` - stop and return the answer (handled by the agent,
    not a tool).

``KnowledgeBase`` implements this API over the bundled article dict; the
``Tool`` subclasses expose it to the agent through the uniform
``name`` / ``description`` / ``call(input)`` interface.
"""
from __future__ import annotations

import re


def split_sentences(text: str) -> list[str]:
    """Split article text into sentences (keeps the trailing period)."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]


class KnowledgeBase:
    """The paper's Wikipedia API over a bundled article dict."""

    def __init__(self, articles: dict[str, str]) -> None:
        self.articles = articles
        self.current: str | None = None  # title of the page "open" by Search
        self._lookup_positions: dict[tuple[str, str], int] = {}

    # -- paper Sec. 4.1: Search[entity] ----------------------------------
    def search(self, query: str) -> str:
        query = query.strip()
        for title in self.articles:
            if title.lower() == query.lower():
                self.current = title
                self._lookup_positions.clear()
                first_five = " ".join(split_sentences(self.articles[title])[:5])
                return f"Page: {title}\n{first_five}"
        similar = self._similar_titles(query)
        if similar:
            return (
                f"Could not find '{query}'. Similar pages: "
                + ", ".join(similar)
                + ". Try Search[...] with one of these titles."
            )
        return f"Could not find '{query}' and no similar pages exist."

    def _similar_titles(self, query: str, top: int = 5) -> list[str]:
        tokens = set(re.findall(r"[a-z]+", query.lower()))
        scored: list[tuple[int, str]] = []
        for title in self.articles:
            overlap = len(tokens & set(re.findall(r"[a-z]+", title.lower())))
            if overlap:
                scored.append((overlap, title))
        scored.sort(key=lambda t: (-t[0], t[1]))
        return [t for _, t in scored[:top]]

    # -- paper Sec. 4.1: Lookup[keyword] ----------------------------------
    def lookup(self, keyword: str) -> str:
        keyword = keyword.strip()
        if self.current is None:
            return "No page is open. Use Search[...] first to open a page."
        sentences = split_sentences(self.articles[self.current])
        hits = [s for s in sentences if keyword.lower() in s.lower()]
        if not hits:
            return f"No sentence in '{self.current}' contains '{keyword}'."
        key = (self.current, keyword.lower())
        pos = self._lookup_positions.get(key, 0)
        if pos >= len(hits):
            return (
                f"No more sentences in '{self.current}' contain '{keyword}'. "
                f"({len(hits)} matching sentence(s) shown already.)"
            )
        self._lookup_positions[key] = pos + 1
        return hits[pos]


class Tool:
    """Uniform interface the ReAct agent dispatches actions through."""

    name: str = ""
    description: str = ""

    def call(self, argument: str) -> str:
        """Run the tool and return the observation string."""
        raise NotImplementedError


class SearchTool(Tool):
    """Search[entity]: open a page; first 5 sentences, or similar titles."""

    name = "Search"
    description = (
        "Search[entity]: if the entity has a page, open it and return its "
        "first 5 sentences; otherwise return similar page titles."
    )

    def __init__(self, kb: KnowledgeBase) -> None:
        self.kb = kb

    def call(self, argument: str) -> str:
        return self.kb.search(argument)


class LookupTool(Tool):
    """Lookup[keyword]: next sentence in the open page containing keyword."""

    name = "Lookup"
    description = (
        "Lookup[keyword]: return the next sentence in the currently open "
        "page that contains the keyword. Call repeatedly to advance."
    )

    def __init__(self, kb: KnowledgeBase) -> None:
        self.kb = kb

    def call(self, argument: str) -> str:
        return self.kb.lookup(argument)
