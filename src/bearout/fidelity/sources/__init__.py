"""Authoritative-source adapters: fetch + parse official document text.

First adapter: Ontario's e-Laws JSON API (``elaws``).  An adapter
contributes two things — an async fetch of the official markup, and a
parser from that markup to :class:`~bearout.fidelity.sources.elaws.StatuteSection`.
"""
