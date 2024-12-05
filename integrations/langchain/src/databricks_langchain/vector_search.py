from databricks_langchain import DatabricksVectorSearch
from typing import (
    List,
    Optional
)
from langchain_core.embeddings import Embeddings

class VectorSearchRetrieverTool():
    def __new__(
            self,
            index_name: str,
            endpoint: Optional[str] = None,
            embedding: Optional[Embeddings] = None,
            text_column: Optional[str] = None,
            columns: Optional[List[str]] = None,
    ):
        vector_store = DatabricksVectorSearch(
            endpoint=endpoint,
            index_name=index_name,
            embedding = embedding,
            text_column = text_column,
            columns = columns
        )
        return vector_store.as_retriever().as_tool()