# Retrieval results

Rank shown = position of the first passage from the right paper (- = not in top 10).

## Keyword questions (42)

| Pipeline | Hit@5 | MRR@10 | ms / query |
|---|---|---|---|
| bm25 | 0.976 | 0.952 | 3 |
| dense | 0.881 | 0.864 | 10 |
| hybrid | 0.952 | 0.960 | 13 |
| hybrid+rerank | 0.976 | 0.961 | 2580 |

Questions where the pipelines disagree:

| Question | bm25 | dense | hybrid | hybrid+rerank |
|---|---|---|---|---|
| Why do companies keep rediscovering the same RAG pitfalls? | 3 | 7 | 1 | 1 |
| Which baselines like TreeQA and Biorag were compared against a RAG method that uses evidence constraints? | 1 | - | 7 | 5 |
| How can documents be turned into model parameters instead of being pasted into the prompt? | 7 | 1 | 1 | 1 |
| What limitations come from building a taxonomy out of preprints that keep getting updated? | 2 | 1 | 1 | 1 |
| Which multimodal embeddings were compared for retrieving answers from videos? | 1 | 7 | 1 | 1 |
| How was the image query benchmark built with Google Image Search and simulated users for agentic multimodal RAG? | 1 | 3 | 1 | 1 |
| Why can RAGAs answer relevancy stay high even when the retrieved context from theses is irrelevant? | 1 | - | 6 | 6 |
| What threshold worked best when MetaRAG was reproduced with a newer GPT-3.5? | 1 | 6 | 1 | 1 |
| Can using different chunk representations for retrieval and for generation improve RAG? | 1 | 2 | 1 | 1 |

## Reworded questions (41)

| Pipeline | Hit@5 | MRR@10 | ms / query |
|---|---|---|---|
| bm25 | 0.756 | 0.578 | 6 |
| dense | 0.902 | 0.732 | 15 |
| hybrid | 0.878 | 0.739 | 40 |
| hybrid+rerank | 0.976 | 0.797 | 2434 |

Questions where the pipelines disagree:

| Question | bm25 | dense | hybrid | hybrid+rerank |
|---|---|---|---|---|
| Is there a cost to letting the scoring model compare more passages at once? | - | 2 | 3 | 3 |
| Can a model reflect on what it already knows before walking a graph of facts to answer a question? | - | 2 | 3 | 3 |
| Generation that loops back to fetch more evidence and rewrite its own output | 4 | 2 | 1 | 1 |
| A review of how organisations engineer retrieval pipelines and make them trustworthy | 1 | 4 | 1 | 2 |
| Storing a document's knowledge in the model's weights rather than in the prompt | 5 | 1 | 2 | 2 |
| Answering questions over several organisations' private data without pooling it, using secure hardware | 2 | 1 | 1 | 1 |
| An agent that plans whether to look up images or text before answering a visual question | - | 1 | 4 | 4 |
| Skipping document lookup for small talk and questions that don't need it | - | 1 | 4 | 4 |
| Writing the related-work section of a paper automatically | 5 | 2 | 1 | 1 |
| Adapting a question answering system to a new domain using only unlabeled local documents | 1 | 1 | 3 | 3 |
| Repeating an earlier experiment on a model that monitors its own reasoning to see if the results still hold | 6 | 1 | 1 | 1 |
| Helping help-desk engineers fix tickets by showing similar old cases | 4 | 1 | 1 | 1 |
| Training a small open model to label which documents are relevant, cheaply | 3 | 2 | 1 | 1 |
| Tricking custom chatbots into harmful output by poisoning the files they search | 1 | 2 | 1 | 1 |
| Deciding when and what to look up based on which upcoming words the model is unsure about | 1 | - | 6 | 1 |
| Scoring a retrieval question answering pipeline without human-written reference answers | 9 | - | - | 1 |
| A light evaluator that grades retrieved documents and falls back to web search when they are bad | 2 | 5 | 2 | 2 |
| Searching over speech recordings directly instead of transcripts for a voice assistant | 3 | 1 | 1 | 1 |
| Several cooperating agents that personalise product suggestions from a shopper's history | 3 | 1 | 1 | 1 |
| Highlighting the exact region of a page image that supports an answer | 5 | 2 | 2 | 2 |
| Lessons from putting a retrieval chatbot into production inside a company | - | 2 | - | 3 |
| Spotting topics that nobody has written about yet | 3 | 4 | 1 | 1 |
| An evaluation campaign where teams submitted long cited answers to narrative-style questions | 1 | 8 | 2 | 2 |
| Building the retrieval graph around questions rather than entities | - | 5 | 7 | 3 |
| Combining facts from several sources that may contradict each other, without making things up | - | 1 | 2 | 1 |

