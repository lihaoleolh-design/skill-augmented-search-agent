---
name: entity-disambiguation
description: Use when a question distinguishes between similarly named people, organizations, locations, or titles.
---
# Entity Disambiguation

1. Identify ambiguous names and every available constraint.
2. Search for candidate entities and aliases.
3. Compare their roles, organizations, locations, and active dates.
4. Inspect source passages for direct evidence of identity.
5. Reject candidates that contradict the question's constraints.
6. If ambiguity remains, search a discriminating detail rather than guessing.
7. Return only the supported entity or Unknown if unresolved.

A shared name or adjacent mention is not sufficient evidence.
