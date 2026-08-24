# bearout

[![CI](https://github.com/stephs-repos/bearout/actions/workflows/ci.yml/badge.svg)](https://github.com/stephs-repos/bearout/actions/workflows/ci.yml)

**A runtime grounding gate that fails closed and publishes its own error rates.**

bearout is an open-source demonstration of the anti-confabulation mechanisms
built for a past project, a consumer legal-information prototype. Every sentence
of a generated answer is checked against its cited sources, and what can't be
supported doesn't ship. Because a gate whose own accuracy is unknown is theatre,
the gate's error rates are measured on a sealed, independently labelled fixture
and published below.

Two components:

- **The gate**, which verifies each sentence, fails closed, and abstains rather
  than shipping a gutted answer.
- **The corpus verifier**, which checks every stored chunk against the
  authoritative source (Ontario's e-Laws API) for currency drift, extraction
  fidelity, and silent drops, because grounding against a stale index is
  worthless.

Both are hand-rolled on the Python standard library rather than built on
LangChain, LlamaIndex, or any other LLM framework. The core installs with zero
dependencies; the Anthropic judge adapter, the e-Laws fetcher, and the Postgres
corpus reader are optional extras.

---

## Measured error rates

Two error types, and they are not symmetric:

- **False pass.** An unsupported claim slips through and reaches the user, which
  is the direction that does the harm.
- **False strip.** A supported sentence is wrongly removed, which costs
  completeness rather than correctness.

| | Rate | In judge terms |
|---|---|---|
| **False pass** | **2.4%** | TNR 97.6% |
| False strip | 20.2% | TPR 79.8% |

The gap between the two rates was put there on purpose. In consumer legal
information a missing sentence is a smaller failure than a wrong one, so the gate
is tuned to pay a recall tax for a low leak rate. If your domain's asymmetry runs
the other way, this is the wrong operating point, and you can see that before
adopting it, which is the point of publishing.

**Fixture card**

| | |
|---|---|
| Items | 474 (153 real sentences logged from the prototype, 54 constructed-entailed, 267 constructed-perturbed) |
| Perturbations | exactly one factual corruption each: wrong number / wrong section / inverted condition / added qualifier / wrong actor / wrong outcome |
| Labels | binary (supported / unsupported) |
| Labelling | logged items: **3 independent blind AI judges**, no access to live verdicts, majority vote, 2-1 splits flagged (5 of the 153). Constructed items: entailment or corruption by construction, plus an independent adversarial verification pass |
| Sealing | frozen by content hash; any regeneration is a new instrument version |
| Judge model | Claude Sonnet 4.5, temperature 0 |

**Provenance of the numbers.** The run above was executed against fixture v1 and
scored **20.5% / 2.1%**. The fixture was later re-audited and two labels corrected
(both supported → unsupported); re-scoring the same unchanged verdicts against the
corrected labels gives the **20.2% / 2.4%** headline. Both are shown because the
difference between them is the audit trail. 23 further contested labels remain
deferred to expert review. The instrument is good rather than perfect, and its
residual uncertainty is disclosed rather than hidden.

The fixture itself is **not** published. Training or tuning against the exam
destroys the only measure you have. What is published is the methodology, the
per-item verdicts, and the numbers.

---

## What was rejected, and why

The obvious improvement to a sentence-level gate is to decompose each sentence
into atomic claims and verify those, since smaller questions should be easier to
judge and a failing claim can be repaired instead of costing the whole sentence.
It was preregistered with a pass/fail gate
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
gate is 20.2/2.4 and claim iteration 1 is 41.5/2.4, and the ranking is unchanged.)
Every rate here, the MiniCheck row included, is recomputable from the per-item
artifacts in
[`experiments/claim-decomposition/artifacts/`](https://github.com/stephs-repos/bearout/tree/main/experiments/claim-decomposition/artifacts).

The design failed twice and did not ship, and the structural reason generalizes
beyond this domain. A sentence survives only if *all* its claims do, so the
verdict is an AND over claims, and decomposition **multiplies** per-call noise
unless per-claim accuracy far exceeds per-sentence accuracy, which measurement
showed it does not.

The design did cure the specific case that motivated it, and it was rejected
anyway, because one convincing example is not an eval.

Code, preregistration, and per-item run artifacts:
[`experiments/claim-decomposition/`](https://github.com/stephs-repos/bearout/tree/main/experiments/claim-decomposition).

The last row deserves its own note. An off-the-shelf trained detector had
comparable false-strip on this domain and missed **30% of fabrications**, which
is why domain fixtures exist.

---

## Where this sits

Comparison is against each product's **documented** behaviour; corrections
welcome via an issue.

| | Runtime gate | Per-sentence | Fail-closed by default | Publishes its own error rates | Verifies the corpus |
|---|---|---|---|---|---|
| **bearout** | ✅ | ✅ | ✅ | ✅ | ✅ |
| Vertex Check Grounding | returns scores | ✅ | ❌ your policy | ❌ | ❌ |
| Azure groundedness + correction | ✅ | ✅ | ❌ configurable | ❌ | ❌ |
| Bedrock contextual grounding | ✅ | response-level threshold | ⚠️ threshold you set | ❌ | ❌ |
| Vectara HHEM / Corrector | ✅ | ✅ | ❌ score + correct | ❌ | ❌ |
| Guardrails `provenance_llm` | ✅ | ✅ | ❌ configurable | ❌ | ❌ |
| RAGAS / DeepEval | ❌ offline eval | ✅ | n/a | ❌ | ❌ |
| LettuceDetect / MiniCheck / Lynx | detector models, not gates | ✅ | n/a | ✅ on public benchmarks | ❌ |

**bearout is not another eval library.** RAGAS and DeepEval score answers
after the fact; this decides what ships. Per-sentence checking is not the novel
part either, since several products above do it. What is uncommon is shipping
fail-closed as the *default* rather than handing you a score, and publishing the
checker's own confusion matrix. What appears to be uncontested is verifying the
corpus itself against its authoritative source, which none of the products above
do.

**The judge is swappable.** It's a single `chat()` protocol
([`llm.py`](https://github.com/stephs-repos/bearout/blob/main/src/bearout/llm.py)), so a trained detector such as
LettuceDetect, MiniCheck, Lynx, or your own can be the backend. The numbers above
are for a prompted judge; substituting a detector is a different operating point
that you should re-measure on your own fixture.

---

## Quickstart

```bash
pip install 'bearout[anthropic]'
export ANTHROPIC_API_KEY=...
```

No API key? `python examples/01_gate_offline.py` runs the gate against a
scripted judge, and `python examples/03_fidelity_elaws.py` verifies a real
Ontario regulation against the government source with no key at all (it needs
the `fetchers` extra: `pip install 'bearout[fetchers]'`). See
[`examples/`](https://github.com/stephs-repos/bearout/tree/main/examples).

```python
import asyncio
from bearout import validate_grounding
from bearout.adapters.anthropic import AnthropicChat

outcome = asyncio.run(
    validate_grounding(
        answer="You must report the defect under Reg. 892 s. 4.4(2). "
        "Tarion will then pay you $100,000.",
        sources_block="<your retrieved sources here>",
        llm=AnthropicChat(),
    )
)

print(outcome.grounded)  # False if too little survived; abstain, don't ship
print(outcome.answer)  # surviving text ("" when not grounded)
print(outcome.removed)  # exactly what was suppressed
print(outcome.verdicts)  # (sentence, supported) for all of them; the audit trail
```

`AnthropicChat()` defaults to the judge the rates above were measured with, pinned
to a dated snapshot and run at temperature 0, so a default run is the configuration
those numbers describe. Pass `model=` to use a different judge, and re-measure
before relying on the published rates, because a different judge is a different
operating point.

Sentence splitting is citation-aware, which is load-bearing rather than cosmetic.
Naive splitting shatters `Reg. 892 s. 4.4(2)` into fragments like `892 s.`, which
any honest judge then rejects, so a well-sourced answer over-abstains, and one
observed case lost 18 of 25 "sentences" that way.

## Corpus fidelity

A grounding gate is only as good as the corpus it grounds against, so the
verifier runs three layers per document:

- **L1. Currency and parser drift.** Re-fetch the official text, rebuild the
  exact chunk ingest would store, compare per section.
- **L2. Extraction fidelity.** Re-extract with a second, independent mechanism;
  every stored body must appear as a contiguous substring of it. L1 pass + L2
  fail is the signature of a parser bug that survived ingest.
- **L3. Silent drops.** Independent section inventory, compared both ways.

Ontario's e-Laws API is the first source adapter (`pip install 'bearout[fetchers]'`
for the live fetch); the corpus side is a pluggable reader (Postgres reference
adapter included, `pip install 'bearout[pgvector]'`). Latest live run against the
prototype's corpus: **418/418 sections verified across 9 statutes and regulations.**

## Design principles

1. **Fail closed.** Supported means directly stated or entailed. Judge errors,
   API failures, and unparseable verdicts all count as unsupported.
2. **Abstain over mutilate.** Below the keep-ratio the answer is withheld
   entirely. A half-answer that has lost its qualifiers is worse than no answer.
3. **Measure the instrument.** Publish the confusion matrix, seal the fixture,
   preregister changes, cap the tuning iterations.

We will never describe this as "hallucination-free." What it offers is a
mechanism and an audit trail, and that is all it will ever claim.

## Status

Alpha, extracted from a working prototype. The gate and corpus verifier ship with
their test suites. The next steps are a public-benchmark row (RAGTruth /
LLM-AggreFact) so the numbers can be compared on shared ground, and a harness for
measuring a judge against *your* labeled set.

## License

Apache-2.0. The e-Laws fixture under `tests/fixtures/` is Ontario legislation,
© King's Printer for Ontario, reproduced as test data and not covered by that
license.
