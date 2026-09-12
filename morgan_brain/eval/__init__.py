"""Measuring what the memory actually returns.

Retrieval quality has been the project's largest unmeasured claim: the semantic upper index
is justified by a paper's numbers, not by this system's, and the suite that looked like a
quality harness ran over a hash embedder and proved only that the plumbing was connected.
Everything here exists to turn an assumption into a number.

``retrieval`` scores a recall run against labelled probes. When the learning loop returns,
its promotion gate belongs beside it -- a candidate that cannot be scored cannot be gated.
"""
