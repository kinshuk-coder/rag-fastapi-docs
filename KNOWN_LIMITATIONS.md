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

## 3. LLM-judge faithfulness/relevance scores don't measure completeness

Question q005 ("How do I declare a request body using a Pydantic model?")
scored a perfect 5/5 on both faithfulness and relevance after Milestone 7's
hybrid search + reranking changes. The actual generated answer was:

> "You can declare a request body using a Pydantic model [3]. To declare a
> request body, you use Pydantic models with all their power and benefits [3]."

This is circular - it restates the question as an answer, with no code
example, no mention of how the parameter is type-hinted, nothing a developer
could actually act on. Retrieval had correctly found the right source chunk
(`tutorial/body.md`, appearing at both rank #1 and #3); the problem was
entirely in generation, not retrieval - yet the judge scored it perfectly.

**Root cause:** the judge rubric (see `eval/run_eval.py`'s `JUDGE_SYSTEM_PROMPT`)
only asks two questions: does the answer contradict the context
(faithfulness), and is it on-topic (relevance). Neither axis checks whether
the answer contains concrete, actionable detail. A vague, technically
non-contradictory, on-topic answer can score perfectly on both dimensions
while being close to useless.

**Status:** addressed in the current code, pending a fresh full evaluation
run. The generation prompt now requires relevant concrete code, parameter, or
method details, and the judge includes a third `completeness` dimension that
scores answers against the actionable detail available in the retrieved
context. This does not make a single LLM judge definitive; the original
manual spot-check remains a necessary complement to aggregate scores.

**Lesson:** an LLM-as-judge setup is only as good as its rubric. Perfect
scores are a reason to spot-check real outputs, not a reason to stop
looking - this is the same instinct that caught the Milestone 5 findings,
just applied to the scoring system itself instead of the eval set.
