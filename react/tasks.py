"""Task data: a bundled HotpotQA-style multi-hop QA mini-corpus.

Paper Sec. 4.1 evaluates ReAct on HotpotQA, where the agent answers multi-hop
questions by interacting with a simple Wikipedia API (search / lookup). The
real HotpotQA corpus is far too large to bundle, so this module ships a tiny
self-written Wikipedia-like knowledge base: six short articles with mutually
consistent facts, plus three two-hop questions whose answers require a
Search + Lookup trajectory (the paper's characteristic evidence-gathering
pattern).

All facts below are original fiction written for this demo - no text is copied
from Wikipedia.
"""
from __future__ import annotations

ARTICLES: dict[str, str] = {
    "Midnight Harvest": (
        "Midnight Harvest is an American indie rock band formed in Portland, Oregon in 2015. "
        "The band consists of vocalist June Park and guitarist Theo Marsh. "
        "Their debut album, Ember Skies (2019), was produced by Silas Reed, while their "
        "second album, Lantern Light (2022), was produced by Dana Whitfield. "
        "Ember Skies won the 2020 Alder Award for Best Rock Album. "
        "The band is known for blending folk harmonies with distorted guitars."
    ),
    "Ember Skies": (
        "Ember Skies is the debut studio album by the American band Midnight Harvest, released in 2019. "
        "The album was produced by Silas Reed and recorded in a converted barn outside Portland. "
        "Its lead single, Glass Harbor, reached number 12 on the alternative charts. "
        "Ember Skies won the 2020 Alder Award for Best Rock Album. "
        "Critics praised the album's layered vocal harmonies."
    ),
    "Silas Reed": (
        "Silas Reed is an American record producer and composer born in 1978. "
        "He produced the 2019 debut album Ember Skies by Midnight Harvest. "
        "He also composed the soundtrack for the 2021 film The Cartographer's Daughter. "
        "Reed is a frequent collaborator of director Maren Kessler. "
        "His production style blends analog tape recording with electronic textures."
    ),
    "The Cartographer's Daughter": (
        "The Cartographer's Daughter is a 2021 adventure drama film directed by Maren Kessler. "
        "The film's soundtrack was composed by Silas Reed. "
        "It stars Iris Bell as Wren, a mapmaker's apprentice. "
        "The film won Best Cinematography at the 2022 Harbor Film Festival. "
        "Kessler shot the film on location in the Faroe Islands."
    ),
    "Maren Kessler": (
        "Maren Kessler is a Norwegian film director born in Bergen in 1981. "
        "She directed the 2021 adventure drama The Cartographer's Daughter. "
        "Her second feature, Salt Meridian (2024), premiered at the Venice Film Festival. "
        "Kessler frequently collaborates with composer Silas Reed. "
        "Her films are noted for their stark coastal landscapes."
    ),
    "Lantern Light": (
        "Lantern Light is the second studio album by Midnight Harvest, released in 2022. "
        "The album was produced by Dana Whitfield. "
        "It features the single Paper Moons. "
        "Lantern Light debuted at number 8 on the indie charts. "
        "The album marked a shift toward synthesizer-driven arrangements."
    ),
}

# Three two-hop questions over the corpus above. Each needs at least one
# Search (find the right page) and one Lookup (pin down the exact fact),
# mirroring the paper's HotpotQA trajectories.
SAMPLE_QUESTIONS: list[dict[str, str]] = [
    {
        "question": "Which album by Midnight Harvest was produced by Silas Reed?",
        "gold": "Ember Skies",
    },
    {
        "question": (
            "Who composed the soundtrack for the 2021 film directed by "
            "Maren Kessler that stars Iris Bell?"
        ),
        "gold": "Silas Reed",
    },
    {
        "question": (
            "Which 2021 film had its soundtrack composed by the producer "
            "of Midnight Harvest's debut album?"
        ),
        "gold": "The Cartographer's Daughter",
    },
]
