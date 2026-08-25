# Examples

Three runnable examples. Two need no API key and no account.

| | Needs a key | Cost | What it shows |
|---|---|---|---|
| [`01_gate_offline.py`](01_gate_offline.py) | no | free | The gate's decision logic: strip, abstain, split, fail closed |
| [`02_gate_live.py`](02_gate_live.py) | yes | a few cents | The same gate against a real judge, with cost and latency measured |
| [`03_fidelity_elaws.py`](03_fidelity_elaws.py) | no | free | Corpus verification against a live government source |

```bash
uv run python examples/01_gate_offline.py
uv run python examples/03_fidelity_elaws.py            # add --offline to skip the network

# Point the verifier at any Ontario statute or regulation:
uv run python examples/03_fidelity_elaws.py \
    --url https://www.ontario.ca/laws/statute/90o31 \
    --title "Ontario New Home Warranties Plan Act, R.S.O. 1990, c. O.31"

export ANTHROPIC_API_KEY=...
uv run python examples/02_gate_live.py                 # add --save to write a transcript
```

## 01 — the gate, offline

Four scenarios, each isolating one property:

- **Strip and ship.** A fabricated dollar figure is removed; the two sound sentences survive.
- **Abstain.** When only one sentence of four survives, the whole answer is withheld rather than shipped as a remnant that reads complete.
- **Citation-aware splitting.** A naive split on `[.!?]` turns three sentences into six fragments — shattering `Reg. 892 s. 4.4(2)` and breaking mid-parenthetical. A judge correctly rejects fragments, so the measured false-strip rate silently becomes a property of your tokenizer.
- **Fail closed.** The judge raises; both genuinely-supported sentences are suppressed anyway.

The judge is a **scripted stub** that reads verdicts from a table. It demonstrates what the gate does with verdicts, not how accurate a judge is — for that, see the measured error rates in the top-level README.

## 02 — the gate, live

The same two answer scenarios against a real model, with every verifier call metered: call count, input and output tokens, cache reads, wall time, and an estimated cost.

**This is not an accuracy measurement.** It shows what the gate does on a handful of sentences. The published false-pass and false-strip rates come from a sealed, independently labelled 474-item set that is deliberately not in this repository — see the top-level README for the methodology and the results. Cost and latency are reported here because a gate that makes one model call per sentence has a real per-answer price, and that is the second question anyone asks after accuracy.

The meter is a wrapper around the adapter rather than instrumentation inside the gate. That is deliberate: the gate keeps telemetry out of the code path the published error rates describe, so cost accounting belongs to the adapter's side of the `ChatLLM` protocol.

Prices are pinned per model with an as-of date; a model with no pinned price reports tokens and latency and no cost, because a stale price is worse than no price.

> The error rates in the top-level README were measured with a specific judge model, which is the default here. Running against a different model is a different operating point, and the example says so whenever `--model` picks one.

## 03 — corpus fidelity, live

Verifies a stored corpus of O. Reg. 242/21 against Ontario's e-Laws API, in two passes. The live fetch needs the `fetchers` extra (`pip install 'bearout[fetchers]'`); `--offline` does not.

**Pass 1** builds a faithful corpus through the real ingest contract (`chunk_text`, the single shared definition of a stored chunk, so ingest and verifier cannot drift apart). Every section verifies.

**Pass 2** injects four defects that nothing else in a RAG stack would notice, because none of them change the *shape* of a retrieval result:

| Injected | Caught by | Verdict |
|---|---|---|
| A section silently dropped | L3 — independent inventory | `missing_in_corpus` |
| A cross-reference that no longer matches the source | L1 — exact rebuild, and L2 — independent re-extraction | `text_mismatch` + `containment_fail` |
| A section the source no longer shows | L3 — inventory, other direction | `extra_in_corpus` |
| A section stored truncated | L4 — coverage | `incomplete` |

The truncation is the one to watch. It draws no `containment_fail` at all, because containment asks whether the stored text is a *subset* of the official text and a truncation is still a subset. What is left is well-formed law that stops early, so it reads as complete: the qualifier that would have contradicted it is simply not there. Coverage is the only layer that asks whether the official text is covered rather than merely echoed.

The double finding on the edited section is the other interesting one. L1 says the stored text differs from what ingest would rebuild today. L2 says it isn't even a contiguous substring of a *second, independent* extraction. L1 alone cannot separate a real amendment from a parser bug — but L1-pass with L2-fail is the unambiguous signature of a parser bug that survived ingest, which is precisely what a checker re-using the same parser can never catch.

Both passes exit 0: the injected defects are the demonstration, not a failure of it. For the real exit-code contract (`0` clean, `2` fetch error, `5` fidelity violation, with `5` winning over `2`), see `bearout.fidelity.run`.

The regulation text is fetched live from the King's Printer for Ontario. `--offline` reads the captured fixture instead — a snapshot, useful for CI and flights, never authoritative.

### Verifying any Ontario law

`--url` accepts any `ontario.ca/laws/{statute,regulation}/...` document, so you can point the verifier at a law it has never seen:

```bash
uv run python examples/03_fidelity_elaws.py \
    --url https://www.ontario.ca/laws/statute/90o31 \
    --title "Ontario New Home Warranties Plan Act, R.S.O. 1990, c. O.31"
```

That one is a 61-section statute; it verifies clean on the first pass and flags all three injected defects on the second.

`--title` matters more than it looks. Most regulations carry no headnotes, so the document title becomes the label prepended to every stored chunk. It is part of the ingest contract, not decoration — which is why it is an explicit argument rather than something guessed from the URL.

Two combinations are rejected before any network call: a URL the e-Laws adapter cannot map to its JSON API, and `--offline` together with `--url` (there is no captured fixture for an arbitrary document). Both exit 1 with an explanation rather than failing opaquely mid-fetch.
