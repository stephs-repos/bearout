# Rejected: claim-level decomposition

**This directory is evidence, not a feature.** The code here is not part of the
installed package and should not be used. It is kept so that every number in the
project's write-up can be checked against the run that produced it.

## What was tried

The shipped gate ([`bearout.gate`](../../src/bearout/gate.py)) asks one
question per sentence: *is this whole sentence entailed by the sources?* When a
sentence fails, it is deleted.

Two things about that bother you immediately. The question is large — a sentence
can carry five facts, and one shaky clause sinks the other four. And deletion
mutilates: drop the sentence that said "the first is a one-year warranty" and the
answer now reads "The second is a two-year warranty…" with no first.

So: decompose each sentence into atomic single-fact claims, verify each claim on
its own, and **repair** the answer by recomposing it without the failed claims
instead of deleting sentences. Smaller questions, coherent output. It should
work.

## What happened

It was preregistered — design and pass/fail gate written down *before* any
measurement ([`PREREGISTRATION.md`](PREREGISTRATION.md)), including an
anti-fishing cap of at most two tuning iterations against the fixture, so it
could not be quietly tuned until the number looked good.

Measured on the same sealed 474-item independently labelled fixture as the shipped gate:

| Design | False-strip | False-pass | Verdict |
|---|---|---|---|
| **v1 sentence** (shipped) | **20.5%** | **2.1%** | baseline |
| claim mode, iteration 0 | 52.4% | 2.1% | gate failed |
| claim mode, iteration 1 | 41.1% | 1.7% | gate failed, cap reached |

Gate was: false-strip materially below 20.5% (target ≤12%) while holding
false-pass ≤3.1%. Iteration 0 missed by a factor of four. Iteration 1 — with the
decomposer's three diagnosed pathologies fixed — halved the gap and still landed
at roughly double the design it was meant to beat. Cap reached, design rejected,
sentence mode remained the default.

Note the *shape* of the result: claim mode was slightly **safer** (1.7% vs 2.1%
false-pass — fewer fabrications leaked) and about twice as over-cautious. It
didn't fail by being bad at verification. It moved along the
safety/completeness tradeoff, in the wrong direction for a usable answer.

## Why it failed

**The structural reason is the interesting part, and it generalizes.**

A sentence survives only if *all* of its claims survive. The sentence verdict is
an AND over claim verdicts. So decomposition multiplies per-call noise: split one
sentence into four claims and you now need four correct judgements instead of
one. That only pays off if per-claim accuracy is *much* higher than per-sentence
accuracy — and measured, it isn't. The judge was already a capable model; one
atomic claim is not meaningfully easier for it than one sentence.

On top of that, claims judged in isolation lose the context that made them
resolvable, which adds *systematic* strips rather than random ones.

This runs against the intuition — and against the FactScore-style decomposition
advice — that atomizing claims makes verification easier. It makes each
*question* easier and the *aggregate* worse.

Iteration 0 had three fixable pathologies on top of the structural problem: the
decomposer emitted separate provenance meta-claims for inline citations,
strengthened statements into absolutes the sentence never asserted, and produced
near-duplicate claims that drew contradictory verdicts. Iteration 1 fixed all
three (citations stay attached to their fact, no strengthening, dedupe). The
structural ceiling remained.

## The part worth sitting with

The design **cured the case that motivated it.** On the five fixture sentences of
the lead exhibit — where the shipped gate deleted a true statement about warranty
layers while holding a source chunk that plainly supported it — v1 judged 4 of 5
wrong and claim mode judged 4 of 5 right, in *both* iterations.

It was still rejected, because across 474 items it was twice as bad.

That is the whole argument for having an instrument. A compelling example
pointed one way; the measurement pointed the other; the measurement won.

## What would be tried next

Both from the post-mortem, neither implemented:

1. **Strip-rescue.** Keep the shipped gate verbatim; add a second look *only* at
   sentences it stripped, with the full answer as context, allowed to overturn a
   strip only by quoting the supporting source text. Asymmetric by construction —
   it cannot loosen a pass, only rescue a strip — so it targets false-strips
   without risking the safety axis, and costs extra calls only when something was
   stripped.
2. **Context-aware sentence verification.** Same criteria, but the judge sees the
   question and the full answer. This *was* measured (15.7% false-strip — a real
   improvement) and rejected on the safety axis: false-pass rose to 4.5%, past
   the 3.1% ceiling.

Also unmeasured and worth isolating: **repair without decomposition.** The
post-mortem blames the decomposer, not the recompose step, but nobody has run
sentence-level verdicts with repair-instead-of-delete as a standalone change.

## Files

- `claim_gate.py` — the rejected implementation (iteration 1 prompts).
- `test_claim_gate.py` — its tests. They pass; passing tests are not a passing gate.
- `PREREGISTRATION.md` — the prereg and its recorded outcome, reproduced
  unedited from the internal record.
- `artifacts/` — per-item verdicts for all three runs (474 items each: item id,
  origin, true label, whether the labelling judges were unanimous, perturbation type,
  the verifier's verdict). Enough to recompute every rate above. The fixture's
  sentence text and source blocks are **not** included — the instrument stays
  sealed so it cannot be trained or tuned against.

Recompute any row:

```bash
python3 -c "
import json,sys
d=json.load(open(sys.argv[1])); o=d['outcomes']
sup=[x for x in o if x['true_label']=='supported']
uns=[x for x in o if x['true_label']=='unsupported']
fs=sum(1 for x in sup if x['verdict']=='unsupported')
fp=sum(1 for x in uns if x['verdict']=='supported')
print(f'false-strip {100*fs/len(sup):.1f}%  false-pass {100*fp/len(uns):.1f}%')
" artifacts/claim-mode-iter1-failed.json
```
