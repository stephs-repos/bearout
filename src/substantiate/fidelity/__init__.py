"""Corpus fidelity verification (extraction in progress).

Three-layer verification that a RAG corpus faithfully mirrors its
authoritative source (first adapter: Ontario's e-Laws JSON API):

L1 — currency + parser drift: re-fetch the official text, rebuild the
     exact chunk text ingest would store, exact-compare per section.
L2 — extraction fidelity: re-extract with a second, independent
     mechanism; every stored section body must appear as a contiguous
     normalized substring of it.  L1-pass + L2-fail is the signature of
     a parser bug that survived ingest.
L3 — silent drops: an independent section inventory compared BOTH ways
     (sections the official text has but the corpus lacks, and corpus
     sections the official text no longer shows).
"""
