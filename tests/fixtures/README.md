# Test fixtures

- `synthetic-act.html` — a hand-written miniature act exercising every
  paragraph class the e-Laws parser handles (toc, heading1, headnote,
  section, subsection, amendments, footnote, nested inline tags,
  entities).
- `truncating-act.html` — a miniature act shaped like the real defect
  this parser had: a continuation paragraph that e-Laws classes as
  `p.section` but which carries no leading section number. Parsed
  wrongly it truncates s.1 mid-clause, and what remains ("No employer
  shall dismiss a worker, (a) ... (b) ...,") reads as an unconditional
  prohibition — a stronger claim than the section makes. Corpus text
  that is wrong in the confident direction is the failure this package
  exists to prevent.
- `oreg-242-21.html` — captured e-Laws JSON-API markup for O. Reg.
  242/21 (Mediation Prior to Notice of Decision) under the Ontario New
  Home Warranties Plan Act, from
  https://www.ontario.ca/laws/regulation/210242. Ontario legislation is
  © King's Printer for Ontario; it is reproduced here, unmodified and
  attributed, solely as test data and is **not** covered by this
  repository's Apache-2.0 license. It is a snapshot: the law may have
  been amended since capture (which is exactly the class of drift the
  fidelity verifier detects in live mode).
