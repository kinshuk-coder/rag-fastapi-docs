# Known Limitations

Findings from building and iterating on the evaluation set (Milestone 5),
kept as an honest record for the project README rather than smoothed over.

## 1. "Unanswerable" ground truth requires checking content, not just filenames

When building the eval set's deliberately out-of-scope questions (Celery,
Redis, Stripe, etc.), absence was initially verified by grepping **source
filenames** in the corpus for these keywords. This missed passing mentions
*inside* otherwise-unrelated files.

Concretely: `tutorial/background-tasks.md` briefly recommends "bigger tools
like Celery" (with Redis/RabbitMQ as message queue options) as an aside,
without any dedicated integration guide. A question asking "how do I set up
Celery task queues alongside FastAPI?" was originally labeled unanswerable,
but the system correctly found this real, grounded snippet and answered from
it rather than refusing — which was the *correct* behavior. The eval label
was wrong, not the pipeline.

**Fix applied:** verify absence by searching chunk **text content**, not just
filenames, before labeling a question unanswerable. The replacement question
(MFA/SMS-based authentication) was re-verified this way.

**Lesson:** a keyword search over filenames tells you a topic isn't the
*subject* of any document; it doesn't tell you the topic is never
*mentioned*. For ground-truth construction, always check the actual content.

## 2. Compound, multi-topic questions expose a real dense-retrieval limitation

Question q022 asked: *"How would I structure a FastAPI app that needs a
database, background startup/shutdown logic, and file uploads all together?"*

All 5 retrieved chunks were about project **structure/organization**
(`Bigger Applications - Multiple Files`, `Sub Applications - Mounts`) — none
touched the three specific features named (database, lifespan events, file
uploads).

This isn't a retrieval bug so much as a known constraint of single-vector
dense embeddings: one embedding represents one dominant semantic signal in a
sentence. Here, the word "structure" dominated the embedding more than the
three specific nouns it was structuring for, so retrieval optimized for the
wrong axis of the question.

**This is unlikely to be fixed by hybrid search (Milestone 6)** — that
technique addresses lexical vs. semantic mismatch, not compound information
needs. The standard fix is **query decomposition**: splitting a compound
question into independent sub-queries (e.g. "database setup", "startup
shutdown events", "file uploads"), retrieving separately for each, then
merging/deduplicating results before generation.

**Status:** left as a documented, out-of-scope limitation rather than
patched. Flagged as candidate future work rather than something masked by
loosening the eval set's expected sources to match whatever retrieval
happened to return.