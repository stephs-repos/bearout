# Experiments

Designs that were measured and **rejected**. Kept as evidence so every number in
the project's write-ups can be checked against the run that produced it.

Nothing here is part of the installed package. `pip install bearout` gives
you one gate.

- [`claim-decomposition/`](claim-decomposition/) — decompose sentences into atomic
  claims, verify each, repair instead of delete. Preregistered, failed its gate
  twice (52.4% then 41.1% false-strip vs the shipped 20.5%), not shipped.
