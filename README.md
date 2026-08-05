# Substantiate

**A runtime grounding gate that fails closed — and publishes its own error rates.**

Every sentence of a generated answer is checked against its cited sources. What
can't be supported doesn't ship. And because a gate whose own accuracy is unknown
is theatre, the gate's error rates are measured on a sealed, expert-labeled
fixture and published below.

Two components, extracted from a live consumer legal-information product:

- **The gate** — per-sentence verification, fail-closed, abstain rather than ship
  a gutted answer.
- **The corpus verifier** — because grounding against a stale index is worthless.
  Every stored chunk checked against the authoritative source (Ontario's e-Laws
  API) for currency drift, extraction fidelity, and silent drops.

---

## Measured error rates

Two error types, and they are not symmetric:

- **False pass** — an unsupported claim slips through and reaches the user. The
  harm-bearing direction.
- **False strip** — a supported sentence is wrongly removed. Costs completeness,
  not correctness.

| | Rate | In judge terms |
|---|---|---|
| **False pass** | **2.4%** | TNR 97.6% |
| False strip | 20.2% | TPR 79.8% |

The gap is a deliberate choice, not an accident. In consumer legal information a
missing sentence is a smaller failure than a wrong one, so the gate is tuned to
pay a recall tax for a low leak rate. If your domain's asymmetry runs the other
way, this is the wrong operating point — and now you can see that before adopting
it, which is the point of publishing.

**Fixture card**

| | |
|---|---|
| Items | 474 — 153 logged production sentences, 54 constructed-entailed, 267 constructed-perturbed |
| Perturbations | exactly one factual corruption each: wrong number / wrong section / inverted condition / added qualifier / wrong actor / wrong outcome |
| Labels | binary (supported / unsupported) |
| Labelling | logged items: **3 independent blind judges**, no access to live verdicts, majority vote, 2-1 splits flagged (5 of 474). Constructed items: entailment or corruption by construction, plus an independent adversarial verification pass |
| Sealing | frozen by content hash; any regeneration is a new instrument version |
| Judge model | Claude Sonnet 4.5, temperature 0 |

**Provenance of the numbers.** The run above was executed against fixture v1 and
scored **20.5% / 2.1%**. The fixture was later re-audited and two labels corrected
(both supported → unsupported); re-scoring the same unchanged verdicts against the
corrected labels gives the **20.2% / 2.4%** headline. Both are shown because the
difference between them is the audit trail. 23 further contested labels remain
deferred to expert review — the instrument is good, not perfect, and its residual
uncertainty is disclosed rather than hidden.

The fixture itself is **not** published. Training or tuning against the exam
destroys the only measure you have. What is published is the methodology, the
per-item verdicts, and the numbers.

---

## What was rejected, and why

The obvious improvement to a sentence-level gate is to decompose each sentence
into atomic claims and verify those — smaller questions, and you can repair the
answer instead of deleting sentences. It was preregistered with a pass/fail gate
and an anti-fishing cap of two tuning iterations, then measured:

| Design | False strip | False pass |
|---|---|---|
| **sentence-level** (shipped) | **20.5%** | **2.1%** |
| claim decomposition, iteration 0 | 52.4% | 2.1% |
| claim decomposition, iteration 1 | 41.1% | 1.7% |
| MiniCheck-770M alone (off-the-shelf detector, same fixture) | 21.1% | 30.1% |

All rows in this table are scored against **fixture v1 labels**, the version every
one of these runs was executed against, so the designs are compared on identical
ground. (The headline above uses the corrected v3 labels; under those, the shipped
gate is 20.2/2.4 and claim iteration 1 is 41.5/2.4 — the ranking is unchanged.)
Every rate here is recomputable from
[`artifacts/`](experiments/claim-decomposition/artifacts/).

It failed, twice, and did not ship. The structural reason generalizes: a sentence
survives only if *all* its claims do, so the verdict is an AND over claims, and
decomposition **multiplies** per-call noise unless per-claim accuracy far exceeds
per-sentence accuracy. Measured, it does not.

The design did cure the specific case that motivated it — and was rejected
anyway, because one convincing example is not an eval.

Code, preregistration, and per-item run artifacts:
[`experiments/claim-decomposition/`](experiments/claim-decomposition/).

The last row deserves its own note: an off-the-shelf trained detector had
comparable false-strip on this domain and missed **30% of fabrications**. Domain
fixtures exist for a reason.

---

## Where this sits

Comparison is against each product's **documented** behaviour; corrections
welcome via an issue.

| | Runtime gate | Per-sentence | Fail-closed by default | Publishes its own error rates | Verifies the corpus |
|---|---|---|---|---|---|
| **substantiate** | ✅ | ✅ | ✅ | ✅ | ✅ |
| Vertex Check Grounding | returns scores | ✅ | ❌ your policy | ❌ | ❌ |
| Azure groundedness + correction | ✅ | ✅ | ❌ configurable | ❌ | ❌ |
| Bedrock contextual grounding | ✅ | response-level threshold | ⚠️ threshold you set | ❌ | ❌ |
| Vectara HHEM / Corrector | ✅ | ✅ | ❌ score + correct | ❌ | ❌ |
| Guardrails `provenance_llm` | ✅ | ✅ | ❌ configurable | ❌ | ❌ |
| RAGAS / DeepEval | ❌ offline eval | ✅ | n/a | ❌ | ❌ |
| LettuceDetect / MiniCheck / Lynx | detector models, not gates | ✅ | n/a | ✅ on public benchmarks | ❌ |

**Substantiate is not another eval library.** RAGAS and DeepEval score answers
after the fact; this decides what ships. And per-sentence checking is not the
novel part — several products above do it. What is uncommon: shipping fail-closed
as the *default* rather than handing you a score, and publishing the checker's own
confusion matrix. What appears to be uncontested: nobody verifies the corpus
itself against its authoritative source.

**The judge is swappable.** It's a single `chat()` protocol
([`llm.py`](src/substantiate/llm.py)), so a trained detector — LettuceDetect,
MiniCheck, Lynx, or your own — can be the backend. The numbers above are for a
prompted judge; substituting a detector is a different operating point that you
should re-measure on your own fixture.

---

## Quickstart

```bash
pip install substantiate[anthropic]
export ANTHROPIC_API_KEY=...
```

```python
import asyncio
from substantiate import validate_grounding
from substantiate.adapters.anthropic import AnthropicChat

outcome = asyncio.run(
    validate_grounding(
        answer="You must report the defect under Reg. 892 s. 4.4(2). "
        "Tarion will then pay you $100,000.",
        sources_block="<your retrieved sources here>",
        llm=AnthropicChat(),
    )
)

print(outcome.grounded)  # False if too little survived — abstain, don't ship
print(outcome.answer)  # surviving text ("" when not grounded)
print(outcome.removed)  # exactly what was suppressed
print(outcome.verdicts)  # (sentence, supported) for all of them — the audit trail
```

Sentence splitting is citation-aware, which is load-bearing rather than cosmetic:
naive splitting shatters `Reg. 892 s. 4.4(2)` into fragments like `892 s.`, which
any honest judge then rejects — so a well-sourced answer over-abstains. One
observed case lost 18 of 25 "sentences" that way.

## Corpus fidelity

A grounding gate is only as good as the corpus it grounds against. Three layers,
per document:

- **L1 — currency and parser drift.** Re-fetch the official text, rebuild the
  exact chunk ingest would store, compare per section.
- **L2 — extraction fidelity.** Re-extract with a second, independent mechanism;
  every stored body must appear as a contiguous substring of it. L1 pass + L2
  fail is the signature of a parser bug that survived ingest.
- **L3 — silent drops.** Independent section inventory, compared both ways.

Ontario's e-Laws API is the first source adapter; the corpus side is a pluggable
reader (Postgres reference adapter included). Latest live run against the
product's corpus: **418/418 sections verified across 9 statutes and regulations.**

## Design principles

1. **Fail closed.** Supported means directly stated or entailed. Judge errors,
   API failures, and unparseable verdicts all count as unsupported.
2. **Abstain over mutilate.** Below the keep-ratio the answer is withheld
   entirely. A half-answer that has lost its qualifiers is worse than no answer.
3. **Measure the instrument.** Publish the confusion matrix, seal the fixture,
   preregister changes, cap the tuning iterations.

We will never describe this as "hallucination-free." Mechanism and audit trail,
always.

## Status

Alpha, extracted from a live product. The gate and corpus verifier ship with
their test suites. Next: a public-benchmark row (RAGTruth / LLM-AggreFact) so the
numbers can be compared on shared ground, and a harness for measuring a judge
against *your* labeled set.

## License

Apache-2.0. The e-Laws fixture under `tests/fixtures/` is Ontario legislation,
© King's Printer for Ontario, reproduced as test data and not covered by that
license.
