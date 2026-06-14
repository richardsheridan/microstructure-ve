"""Test suite for microstructure-ve.

Run the whole suite (unit tests + doctests) from the repo root with::

    python -m unittest discover -s tests -t .

The numpy-only tier runs under the msve env (numpy + scipy). Tests that need
DOLFINx are gated with ``@unittest.skipUnless`` and skip cleanly when it is absent.
"""
