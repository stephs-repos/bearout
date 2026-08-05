# Substantiate

**Fail-closed grounding for RAG: every sentence verified against its cited
source, or suppressed. The verifier's own error rates measured and published.**

If you do grounded generation, you probably don't know your verifier's error
rate — and the literature says a strict prompted LLM judge wrongly flags
roughly half of what it flags (RAGTruth: prompted GPT-4-turbo judge at
P46.9 / R97.9). Substantiate is the gate architecture extracted from a live
consumer legal-information product, plus the instruments we used to measure it
honestly.

## What's here

- **`substantiate.gate`** — the claim-level grounding gate:
  **decompose** an answer into atomic single-fact claims, **verify** each
  claim independently against the retrieved sources (fail-closed: when in
  doubt, unsupported), and **repair** rather than amputate — the answer is
  recomposed without exactly the failed claims, novel content in the repair
  is re-verified, and a repair that won't verify becomes an abstain.
- **`substantiate.fidelity`** — three-layer verification that your corpus
  faithfully mirrors its authoritative source: **L1** currency + parser
  drift (exact chunk reconstruction), **L2** extraction fidelity (an
  independent second extractor; L1-pass + L2-fail is the stable-parser-bug
  signature), **L3** silent drops (both-way section inventory). Ontario's
  e-Laws law API is the first source adapter; the corpus side is a
  pluggable reader (Postgres reference adapter included). Latest live run
  on the product's live corpus: **418/418 sections verified across 9 statutes
  and regulations.**

## Honest numbers

Measured on a sealed, human-labeled instrument (hash-pinned; the fixture
never ships — bring your own labeled set, the harness pattern is documented):

| Metric | Value | Meaning |
|---|---|---|
| False-strip | **20.5%** | a *correct* sentence needlessly suppressed or repaired (the safe direction) |
| False-pass | **2.1%** | a fabricated claim that survived verification (the dangerous direction) |

That asymmetry is a chosen operating point: in a legal domain, an omitted
true sentence costs completeness; a passed fabrication costs a homeowner a
false belief about their rights. The roadmap down from 20.5% is a trained
detector (the literature's named remedy), shipping as a drop-in verifier
backend.

We will never describe this as "hallucination-free." Mechanism and audit
trail, always.

## Quickstart

```bash
pip install substantiate[anthropic]
export ANTHROPIC_API_KEY=...
```

```python
import asyncio
from substantiate import validate_with_claims
from substantiate.adapters.anthropic import AnthropicChat

llm = AnthropicChat()

outcome = asyncio.run(
    validate_with_claims(
        answer="You have 30 days to appeal. The deposit limit is $100,000.",
        sentences=["You have 30 days to appeal.", "The deposit limit is $100,000."],
        sources_block="<your retrieved sources here>",
        llm=llm,
    )
)

print(outcome.grounded, outcome.repaired)
print(outcome.answer)  # the surviving (possibly repaired) answer
print(outcome.removed_claims)  # exactly what was suppressed, and why it's gone
```

`validate_with_claims` returns `None` when decomposition fails — the caller
falls back to a *stricter* path, never fail-open.

## Design principles

1. **Fail closed.** A claim is supported only if directly stated or entailed
   by the sources. Verifier errors and API failures count as unsupported.
2. **Repair, never amputate.** Sentence-level deletion mutilates
   enumerations ("The second is…" with no first). Recomposition without the
   failed claims keeps answers coherent; a repair that smuggles new
   unverified content is rejected.
3. **Measure the instrument.** A gate whose own error rate is unknown is
   theater. Publish false-strip and false-pass, keep the labeled fixture
   sealed (training or prompt-tuning on the exam destroys the measure).

## Status

Alpha. Extracted from a live product. The gate and the fidelity
verifier are ported with their test suites; the verifier-precision harness
(measure your own false-strip/false-pass against a labeled set) is next.
API may move before 0.1.0.

## License

Apache-2.0
