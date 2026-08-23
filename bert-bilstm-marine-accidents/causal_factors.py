"""Multi-label causal-factor extraction from incident narratives.

Zhao et al. (2025) hand-labeled each of their accident reports with one or
more of 32 causal-factor categories (their Table 10 lists all 32 by name),
then fed those multi-label tags into Apriori (Section 3.2/4.4) to mine which
factors co-occur. The MAIB dataset used in this project only ships a single
*incident-type* label per report (Collision, Grounding/Stranding, ...) --
there is no human causal-factor annotation to mine.

To still run the paper's Apriori stage (apriori_analysis.py) end to end,
this module heuristically re-derives multi-label causal-factor tags per
report by keyword/phrase matching against the paper's own 32 category
names. This is *not* the paper's method -- the paper's labels came from
manual expert review of each report, not keyword matching -- so treat the
resulting tags, and everything mined from them, as an approximation good
enough to demonstrate the Apriori pipeline, not as a validated
re-annotation of the dataset. Keywords below were written by inspecting
common phrasing in MAIB-style accident narratives and are not exhaustive;
extend CAUSAL_FACTOR_KEYWORDS freely.
"""
import re

# category name -> list of case-insensitive regex patterns; a report is
# tagged with a category if ANY pattern matches its (cleaned) text.
CAUSAL_FACTOR_KEYWORDS = {
    "Inadequate manning": [
        r"short[- ]handed", r"under[- ]?manned", r"understaffed",
        r"insufficient crew", r"inadequate manning", r"lack of crew",
        r"not enough crew",
    ],
    "Signals not given as required": [
        r"(?:did not|failed to) (?:give|sound|display) .*signal",
        r"no (?:sound |navigation )?signal", r"signal.* not given",
        r"navigation lights? .*not (?:displayed|shown|on)",
    ],
    "Complex waterway": [
        r"narrow channel", r"congested (?:water|traffic)", r"busy waterway",
        r"traffic separation scheme", r"confined waters?", r"restricted waters?",
    ],
    "Weak safety awareness": [
        r"safety awareness", r"complacen\w*", r"lacked awareness",
        r"safety culture", r"unaware of the risk",
    ],
    "Inadequate safety management system": [
        r"safety management system", r"\bSMS\b", r"procedures? (?:was|were) not followed",
        r"lack of (?:a )?procedure",
    ],
    "Improper shore-based command": [
        r"shore[- ]based", r"shore management", r"office instructed",
        r"company instructed", r"shore staff",
    ],
    "Pilot at fault": [
        r"pilot('s)? (?:error|fault|mistake)", r"pilotage error",
        r"pilot failed",
    ],
    "Rough sea state": [
        r"heavy weather", r"rough sea", r"high seas?", r"storm", r"gale",
        r"heavy swell", r"adverse weather",
    ],
    "Unforeseen": [
        r"unforeseen", r"unexpected(?:ly)?", r"without warning", r"sudden(?:ly)?",
    ],
    "Improper operation": [
        r"improper(?:ly)? operat\w*", r"incorrect(?:ly)? operat\w*",
        r"mishandled", r"operational error", r"wrong (?:lever|control|valve)",
    ],
    "Failure to use safe speed": [
        r"excessive speed", r"too fast", r"did not reduce speed",
        r"failed to (?:reduce|maintain) (?:a )?safe speed", r"unsafe speed",
    ],
    "Failure to fulfill ship's obligations": [
        r"failed to (?:comply|fulfil|fulfill)", r"did not comply",
        r"breach of (?:the )?(?:colregs?|regulations?)",
    ],
    "Poor communication": [
        r"miscommunicat\w*", r"poor communication", r"communication (?:breakdown|failure)",
        r"language barrier", r"failed to communicate",
    ],
    "Improper storage of goods": [
        r"improperly stowed", r"goods.* not secured", r"storage of goods",
    ],
    "Work through fatigue": [
        r"fatigue", r"\btired\b", r"exhausted", r"long hours", r"lack of rest",
        r"overworked",
    ],
    "Negligent lookout": [
        r"lookout", r"failed to (?:see|notice|observe)",
        r"(?:did not|failed to) keep (?:a )?proper watch", r"inadequate watch",
    ],
    "Lack of training": [
        r"lack of training", r"untrained", r"inexperienced",
        r"insufficient training", r"not (?:properly )?trained",
    ],
    "Poor visibility": [
        r"poor visibility", r"\bfog\b", r"low visibility", r"darkness",
        r"reduced visibility",
    ],
    "Mismanagement by the shipowner or company": [
        r"shipowner", r"company failed", r"owner('s)? neglect", r"mismanagement",
    ],
    "Unfit crew": [
        r"unfit crew", r"medically unfit", r"not fit for duty", r"crew.*unfit",
    ],
    "Vessel unseaworthy": [
        r"unseaworth\w*", r"structural (?:damage|weakness)", r"corrosion",
        r"not seaworthy",
    ],
    "Vessel Damage": [
        r"hull damage", r"vessel (?:was )?damaged", r"damage to the (?:vessel|ship)",
    ],
    "Inadequate command of the ship's master": [
        r"master('s)? (?:error|failed|judg?ement)", r"captain('s)? (?:error|failed)",
        r"poor command",
    ],
    "Improper planning": [
        r"poor planning", r"passage plan", r"inadequate planning", r"did not plan",
    ],
    "Improper use of equipment": [
        r"misused", r"incorrect use of", r"improper use of", r"used incorrectly",
    ],
    "Equipment failure": [
        r"equipment (?:failure|failed|malfunction\w*)", r"broke down",
        r"mechanical failure", r"\bdefect\w*",
    ],
    "Inadequately equipped facilities": [
        r"inadequately equipped", r"lacked (?:equipment|facilities)",
        r"not properly equipped",
    ],
    "Improper stowage of cargo": [
        r"cargo (?:shifted|not secured|improperly stowed)", r"\bstowage\b",
    ],
    "Overloading": [
        r"overload\w*", r"exceeded .*capacity", r"over capacity",
        r"too much (?:cargo|weight)",
    ],
    "Improper protection checks": [
        r"protection checks?", r"failed to inspect", r"inspection.*not (?:carried out|done)",
    ],
    "Illegal operation": [
        r"illegal operation", r"unauthori[sz]ed operation", r"without authori[sz]ation",
    ],
    "Illegal modifications": [
        r"illegal modification", r"unauthori[sz]ed modification",
        r"altered without approval", r"modified without",
    ],
}

_COMPILED = {
    category: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    for category, patterns in CAUSAL_FACTOR_KEYWORDS.items()
}


def extract_causal_factors(text: str) -> list[str]:
    """Return every causal-factor category whose keywords appear in `text`."""
    return [
        category
        for category, patterns in _COMPILED.items()
        if any(pattern.search(text) for pattern in patterns)
    ]
