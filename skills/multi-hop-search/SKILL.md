---
name: multi-hop-search
description: Use for questions requiring multiple dependent facts, entities, or documents to establish the answer.
---
# Multi-Hop Search

1. Identify the required answer type and the relationships to verify.
2. Decompose the question into dependent sub-questions.
3. Search the first unresolved clue with its rarest phrase/date/title terms;
   do not send the full natural-language question to BM25.
4. When clues are independent, issue two complementary initial searches rather
   than two paraphrases of the same query.
5. Search the next clue using supported facts from the previous result.
6. Build an evidence chain. Co-occurrence is not proof of a relationship.
7. Recheck contradictory dates, roles, and identities in source documents.
8. Stop only when the requested relationship is supported. Return a short answer.

Never invent an intermediate entity or relationship. Respect tool budgets.
