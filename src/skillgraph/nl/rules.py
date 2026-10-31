"""Deterministic intent parser: the offline fallback and the evaluation baseline.

It recognises the phrasings of the required templates and close variants using
keyword cues plus the resolver's alias index. It is deliberately simple; the
evaluation set (``eval/``) measures where it falls short of the LLM parser.
"""

from __future__ import annotations

import re

from skillgraph.nl.schema import Intent, ParsedQuery
from skillgraph.services.resolver import Kind, Resolver

_PATH = re.compile(r"\b(fastest|quickest|shortest|path|route|roadmap|plan|how (do|can|should) i (become|get))", re.I)
_GAP = re.compile(r"\b(gaps?|missing|lacking|lack)\b", re.I)
_TRANSFER = re.compile(r"\btransfer(able|rable|s)?\b|\bcarry over\b|\bportable\b|\boverlap\b|\bmove\b[^?.]*\bbetween\b", re.I)
_COURSE = re.compile(r"\b(courses?|training|classes?|programmes?|programs?|certifications?)\b|"
                     r"\b(where|how) (can|do|should) i (learn|study)\b", re.I)
_SKILLS_FOR = re.compile(r"\b(what|which) skills\b|\bskills (do|does|are|would)\b|\brequire(d|ments)?\b|\bneed\b", re.I)
_THOROUGH = re.compile(r"\b(thorough|complete|comprehensive|full)\b", re.I)

# "I know Python and SQL." / "I have experience in X, Y" / "my skills are ..."
_KNOW = re.compile(
    r"(?<!do )(?<!what )\b(?:i (?:already )?(?:know|have|can do)|i(?: am|'m|m) good at|my skills (?:are|include)|"
    r"experience (?:in|with)|skilled in|familiar with)\s+(?P<list>[^.?!;]+)", re.I)
# "I am a healthcare data analyst" / "I work as a ..."
_IAM = re.compile(r"(?:\b(?:i am|i'm|im|i work as|working as|currently (?:a|an))|(?:^|[.!?]\s+)as)\s+"
                  r"(?:an?\s+)?(?P<role>[^.?!,;]+)", re.I)
# "... become a Data Scientist" / "path to AI Engineer" / "for a Sustainability Consultant role"
_TARGET = re.compile(
    r"\b(?:become|becoming|be|into|as|for|(?:path|route|move|switch|transition|get) to|towards?)\s+"
    r"(?:an?\s+|the\s+)?(?P<role>[a-z][\w /&+-]*?)"
    r"(?:\s+role)?(?=[?.!,]|$|\s+(?:role|position|job))", re.I)
_BETWEEN = re.compile(r"\bbetween\s+(?:the\s+)?(?P<a>[\w &]+?)\s+and\s+(?:the\s+)?(?P<b>[\w &]+?)(?:\s+economies|\s+economy|[?.!]|$)", re.I)
_DIRECTED = re.compile(r"\b(?:which|what)\s+(?P<a>[\w &]+?)\s+skills\b.*?\b(?:to|into|for)\s+(?:the\s+)?(?P<b>[\w &]+?)(?:\s+economy|\s+sector|[?.!]|$)", re.I)
_SPLIT = re.compile(r"\s*(?:,|\band\b|&|/|\bplus\b|\bas well as\b)\s*", re.I)


class RuleParser:
    name = "rules"

    def __init__(self, resolver: Resolver) -> None:
        self.r = resolver

    # ------------------------------------------------------------------
    def parse(self, question: str) -> ParsedQuery:
        q = " ".join(question.split())
        if _TRANSFER.search(q):
            parsed = self._transfer(q)
            if parsed:
                return parsed

        known = self._known_skills(q)
        current = self._current_role(q)
        target = self._target_role(q, exclude=current)

        if _COURSE.search(q) and not self.r.scan(q, Kind.ROLE):
            skills = [self.r.name(Kind.SKILL, s) for s in self.r.scan(q, Kind.SKILL)]
            if skills:
                return ParsedQuery(intent=Intent.COURSES_FOR_SKILL, skills=skills)
        if target:
            if _GAP.search(q) and (known or current):
                intent = Intent.GAP_ANALYSIS
            elif _PATH.search(q) or ((known or current) and not _SKILLS_FOR.search(q)):
                intent = Intent.LEARNING_PATH
            elif known or current:
                intent = Intent.GAP_ANALYSIS
            else:
                intent = Intent.SKILLS_FOR_ROLE
            return ParsedQuery(intent=intent, target_role=target, current_role=current,
                               known_skills=known, fastest=not _THOROUGH.search(q))
        return ParsedQuery(intent=Intent.OUT_OF_SCOPE, known_skills=known, current_role=current)

    # ------------------------------------------------------------------
    def _transfer(self, q: str) -> ParsedQuery | None:
        m = _BETWEEN.search(q)
        bidirectional = bool(m)
        if not m:
            m = _DIRECTED.search(q)
        if m:
            a, b = self.r.economy(m.group("a")), self.r.economy(m.group("b"))
            if a and b:
                return ParsedQuery(intent=Intent.TRANSFERABLE_SKILLS, source_economy=a, target_economy=b,
                                   bidirectional=bidirectional)
        # fall back to the first two economy words in order of appearance
        found: list[str] = []
        for word in re.findall(r"[a-z]+", q.lower()):
            e = self.r.economy(word)
            if e and e not in found:
                found.append(e)
        if len(found) >= 2:
            return ParsedQuery(intent=Intent.TRANSFERABLE_SKILLS, source_economy=found[0],
                               target_economy=found[1], bidirectional="between" in q.lower())
        # one or no economy named: keep the intent so the pipeline asks for the missing one
        mentions = [w for w in re.findall(r"[a-z]+", q.lower()) if self.r.economy(w)]
        return ParsedQuery(intent=Intent.TRANSFERABLE_SKILLS, source_economy=found[0] if found else None,
                           target_economy=found[0] if len(mentions) >= 2 and found else None)

    def _known_skills(self, q: str) -> list[str]:
        m = _KNOW.search(q)
        if not m:
            return []
        segment = re.split(r"\b(?:what|how|which|can|should|is|and want|but)\b|\s[-–—]\s", m.group("list"), maxsplit=1, flags=re.I)[0]
        out: list[str] = []
        for part in _SPLIT.split(segment):
            part = part.strip(" .")
            if not part:
                continue
            res = self.r.resolve(part, Kind.SKILL)
            out.append(res.name if res.ok else part)
        return [s for i, s in enumerate(out) if s not in out[:i]]

    def _current_role(self, q: str) -> str | None:
        m = _IAM.search(q)
        if not m:
            return None
        phrase = m.group("role")
        ids = self.r.scan(phrase, Kind.ROLE)
        if ids:
            return self.r.name(Kind.ROLE, ids[0])
        words = phrase.split()
        for n in range(min(5, len(words)), 0, -1):          # "healthcare data analyst exploring ..."
            res = self.r.resolve(" ".join(words[:n]), Kind.ROLE)
            if res.ok and res.match.value in ("exact", "alias"):
                return res.name
        return None

    def _target_role(self, q: str, exclude: str | None) -> str | None:
        candidates = []
        for m in _TARGET.finditer(q):
            phrase = m.group("role").strip()
            ids = self.r.scan(phrase, Kind.ROLE)
            if ids:
                candidates.append(self.r.name(Kind.ROLE, ids[0]))
                continue
            res = self.r.resolve(phrase, Kind.ROLE)
            if res.ok:
                candidates.append(res.name)
            elif len(phrase.split()) <= 4 and (res.candidates or re.match(r"becom", m.group(0), re.I)):
                candidates.append(phrase)          # ambiguous, misspelled or unknown; the pipeline says which
        candidates = [c for c in candidates if c != exclude]
        if candidates:
            return candidates[-1]
        ids = [i for i in self.r.scan(q, Kind.ROLE) if self.r.name(Kind.ROLE, i) != exclude]
        return self.r.name(Kind.ROLE, ids[-1]) if ids else None
