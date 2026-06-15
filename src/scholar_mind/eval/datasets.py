from __future__ import annotations

from uuid import uuid4

from scholar_mind.models.domain import RAGEvalSample


class EvalDatasetBuilder:
    def __init__(self, paper_repository):
        self.paper_repository = paper_repository

    def rag_samples(self) -> list[RAGEvalSample]:
        papers = self.paper_repository.all_papers()
        samples: list[RAGEvalSample] = []
        for paper in papers:
            relevant_chunks = [
                chunk["chunk_id"]
                for chunk in self.paper_repository.list_chunks({"paper_ids": [paper.paper_id]})[:3]
            ]
            samples.append(
                RAGEvalSample(
                    sample_id=uuid4().hex,
                    query=paper.title,
                    query_type="precise",
                    relevant_chunk_ids=relevant_chunks,
                    category=paper.categories[0],
                )
            )
        return samples
