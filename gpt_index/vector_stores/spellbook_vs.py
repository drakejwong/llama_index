"""Spellbook Vector store index."""
from typing import Any, Dict, List, Optional

from gpt_index.vector_stores.types import (Node, NodeEmbeddingResult,
                                           VectorStore, VectorStoreQueryResult)

try:
    import spellbook as sb  # noqa: F401
except ImportError:
    raise ImportError(
        '`spellbook` package not found, please run `pip install pinecone-client`'
    )


class SpellbookVectorStore(VectorStore):
    stores_text: bool

    def __init__(self, name: str) -> None:
        self._vector_store = sb.VectorStore(name=name)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'VectorStore':
        return cls(**config_dict)

    @property
    def client(self) -> Any:
        """Get client."""
        return self._vector_store

    @property
    def config_dict(self) -> dict:
        """Get config dict."""
        return {
            'vector_store_name': self._vector_store.name,
        }

    def add(
        self,
        embedding_results: List[NodeEmbeddingResult],
    ) -> List[str]:
        """Add embedding results to vector store."""
        res = self._vector_store.upload_documents_with_embeddings(
            items=[
                sb.DocumentWithEmbedding(
                    text=result.node.text
                    or '',  # HACK: node.text should not be nullable
                    embedding=result.embedding,
                    id=result.node.doc_id,
                )
                for result in embedding_results
            ]
        )
        return [doc['id'] for doc in res['data']['items']]

    def delete(self, doc_id: str, **delete_kwargs: Any) -> None:
        """Delete doc."""
        self._vector_store.delete_documents(ids=[doc_id])

    def query(
        self,
        query_embedding: List[float],
        similarity_top_k: int,
        doc_ids: Optional[List[str]] = None,
        query_str: Optional[str] = None,
    ) -> VectorStoreQueryResult:
        """Query vector store."""
        res = self._vector_store.similarity_search(
            query_embedding=query_embedding,
            k=similarity_top_k,
        )
        nodes, ids = [], []  # TODO: similarities
        for doc in res['data']['items']:
            node = Node(text=doc.text, doc_id=doc.id)
            if hasattr(doc, 'metadata'):
                node.extra_info = doc.metadata
            nodes.append(node)
            ids = doc.id
        return VectorStoreQueryResult(nodes=nodes, ids=ids)
