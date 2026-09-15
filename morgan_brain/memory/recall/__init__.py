"""Turning a query into ranked memories.

``fusion`` merges the vector and keyword rankings by reciprocal rank. Rank-only: scores do
not survive fusion, so the relevance threshold, ``floor``, judges the vector scores before they
reach it.
"""
