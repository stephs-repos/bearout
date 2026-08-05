# Pre-registration: claim-level grounding validator (2026-06-12)

Phase B of the drawing-board plan; the P3.1 structural fix. Committed
before implementation and before any measurement (G2). Lead exhibit:
S12 — the verifier deleted "[fixture sentence redacted: sealed instrument]
[redacted]" 3/3 while holding the chunk that says
"[source excerpt redacted: sealed instrument]… [redacted]
the plan" at rank 2 of its own sources block.

## Design under test ("claim" mode; v1 "sentence" mode stays default until the gate passes)

1. **Decompose** (1 LLM call/answer): restate each sentence as atomic
   single-fact claims (JSON, per-sentence grouping). Absence claims
   ("the sources do not specify X") stay claims. Sentences with no
   factual content yield no claims and are kept.
2. **Verify per claim** (1 small call/claim, bounded concurrency): same
   fail-closed one-word protocol as v1, but the unit is one atomic claim
   — no composite to be "in doubt" about. The strictness is unchanged;
   the question is smaller.
3. **Repair, never amputate** (≤1 recompose call, only when claims
   fail): rewrite the answer removing exactly the failed claims; the
   recomposed answer is re-checked once; abstain if verified-claim
   coverage < 50% or the repair fails. The mutilated-enumeration
   outcome ("The second is…") becomes impossible: output is either a
   coherent answer or an abstain.
4. **Fail-safe**: decompose/parse failure falls back to the v1 sentence
   path (never fail-open). Per-claim verdicts are persisted in
   diagnostics (replaces the unexplainable one-word output).

## Acceptance gate (A3 instrument, 474 items, sentence-level mapping:
sentence verdict = all its claims pass)

- false-strip: materially below the 20.5% baseline (target ≤ 12%)
- false-pass: ≤ 3.1% (baseline 2.1% + the G8 1pt cap)
- wrong_section false-pass: ≤ 2/30 (citation checking must not weaken)
- logged-unsupported false-pass: ≤ 4/22 (fabrications must not gain a
  hiding place; expectation is improvement — atomization exposes them)

Gate fails → claim mode is NOT made default; findings recorded; next
candidate (majority-of-3) gets its own prereg. No iterating the claim
prompts more than twice against A3 (anti-fishing cap, same as WS3).

## Predicted movers (advisory spot-checks ×3, service layer; headline
claims wait for the next G3 median-of-3)

- S12: thesis + first-layer sentences survive; answer enumerates all
  three layers; no mutilation.
- Q30: the composed [section reference redacted] answer survives → answers (citation
  check may still gate the eval verdict).
- Q36/Q2b/S4/S9 abstain-flake class: ≥2 stabilize to answered.
- Guards: OOS 0/22 unaffected (validator runs after the OOS gates);
  latency p50 not worse than v1 (decompose+parallel claims vs
  sequential sentences).

## Out of scope here

Threshold recalibration (own prereg), citation-surface conventions,
majority-of-3 (only if this gate fails).

## OUTCOME (2026-06-12): GATE FAILED — claim mode does NOT ship

- iter0: false-strip 52.4% (vs gate ≤~12%, v1 20.5%); false-pass 2.1% ✓.
- Diagnosis on specimens: the verifier was mostly sound; the DECOMPOSER
  manufactured kill surface — separate provenance meta-claims for inline
  citations, strengthened absolutes the sentence never asserted,
  near-duplicate claims drawing contradictory verdicts.
- iter1 (the one allowed iteration: citations attached to facts, no
  strengthening, dedupe): 41.1% — improved, still double v1. CAP REACHED.
- S12's own five fixture sentences: v1 judged 4/5 WRONG; claim mode
  judges 4/5 RIGHT in both iterations. The design cures the lead exhibit
  and is unshippable corpus-wide — the instrument worked exactly as
  intended, twice.
- STRUCTURAL LESSON: sentence verdicts = AND over claims, so atomization
  MULTIPLIES per-call noise unless per-claim accuracy is far better than
  per-sentence accuracy; measured, it is not (verifier = Sonnet 4.5
  already). Context loss (claims judged in isolation) adds systematic
  strips on top.
- Next-candidate analysis from the data: v1's run-to-run drift is only
  2-3%, so v1 strips are mostly SYSTEMATIC → majority-of-3 of the same
  verifier attacks only the small random slice (LOW expected yield;
  de-prioritized). Strongest remaining candidates: (a) STRIP-RESCUE —
  keep v1 verbatim, add a second-look call ONLY on stripped sentences,
  with the full answer as context, that may overturn only by QUOTING the
  supporting source text (asymmetric: cannot loosen passes; targets
  false-strips precisely; extra cost only on strips); (b) context-aware
  sentence verification (full answer visible, verdict per sentence —
  changes the information available, not the criteria). Each requires
  its own prereg + A3 gate. NOT implemented pending Steph's go.

Sentence mode remains the production default. Failed-run artifacts:
a3-claim-mode-failed-iter{0,1}-2026-06-12.json.
