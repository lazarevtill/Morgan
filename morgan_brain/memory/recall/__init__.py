"""Turning a query into ranked memories.

``fusion`` merges the vector, keyword and entity rankings by reciprocal rank. Rank-only:
scores do not survive fusion, so any relevance threshold belongs on a signal before it reaches
here.
"""
