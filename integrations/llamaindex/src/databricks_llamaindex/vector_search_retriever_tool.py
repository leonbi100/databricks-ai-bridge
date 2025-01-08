from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel, Field, PrivateAttr, model_validator

from llama_index.core.tools import FunctionTool
from llama_index.core.tools.types import ToolMetadata
from llama_index.core.embeddings import BaseEmbedding

from databricks_ai_bridge.vector_search_retriever_tool import VectorSearchRetrieverToolMixin, VectorSearchRetrieverToolInput
from databricks_ai_bridge.utils.vector_search import IndexDetails, parse_vector_search_response, validate_and_get_text_column, validate_and_get_return_columns


class VectorSearchRetrieverTool(FunctionTool, VectorSearchRetrieverToolMixin):
    """
    """

    text_column: Optional[str] = Field(
        None,
        description="The name of the text column to use for the embeddings. "
                    "Required for direct-access index or delta-sync index with "
                    "self-managed embeddings.",
    )
    embedding: Optional[BaseEmbedding] = Field(
        None, description="Embedding model for self-managed embeddings."
    )

    # The BaseTool class requires 'name' and 'description' fields which we will populate in validate_tool_inputs()
    name: str = Field(default="", description="The name of the tool")
    description: str = Field(default="", description="The description of the tool")
    return_direct: bool = Field(
        default=False,
        description="Whether the tool should return the output directly",
    )

    @model_validator(mode="after")
    def _validate_tool_inputs(self):
        from databricks.vector_search.client import VectorSearchClient  # import here so we can mock in tests

        self._index = VectorSearchClient().get_index(index_name=self.index_name)
        self._index_details = IndexDetails(self._index)
        self.text_column = validate_and_get_text_column(self.text_column, self._index_details)
        self.columns = validate_and_get_return_columns(self.columns or [], self.text_column, self._index_details)

        return self

    def __init__(
            self,
            index_name: str,
            num_results: int = 10,
            columns: Optional[List[str]] = None,
            filters: Optional[Dict[str, Any]] = None,
            query_type: str = "ANN",
            tool_name: Optional[str] = None,
            tool_description: Optional[str] = None,
            text_column: Optional[str] = None,
            embedding: Optional[BaseEmbedding] = None
    ):
        def similarity_search(query: str) -> List[Dict[str, Any]]:
            """
            """

            def get_query_text_vector(query: str) -> Tuple[Optional[str], Optional[List[float]]]:
                if self._index_details.is_databricks_managed_embeddings():
                    if self.embedding:
                        raise ValueError(
                            f"The index '{self._index_details.name}' uses Databricks-managed embeddings. "
                            "Do not pass the `embedding` parameter when executing retriever calls."
                        )
                    return query, None

                # For non-Databricks-managed embeddings
                if not self.embedding:
                    raise ValueError("The embedding model name is required for non-Databricks-managed "
                                     "embeddings Vector Search indexes in order to generate embeddings for retrieval queries.")

                text = query if self.query_type and self.query_type.upper() == "HYBRID" else None
                vector = self.embedding.get_text_embedding(text=query)
                if (index_embedding_dimension := self._index_details.embedding_vector_column.get("embedding_dimension")) and \
                        len(vector) != index_embedding_dimension:
                    raise ValueError(
                        f"Expected embedding dimension {index_embedding_dimension} but got {len(vector)}"
                    )
                return text, vector

            query_text, query_vector = get_query_text_vector(query)
            search_resp = self._index.similarity_search(
                columns=self.columns,
                query_text=query_text,
                query_vector=query_vector,
                filters=self.filters,
                num_results=self.num_results,
                query_type=self.query_type,
            )
            docs_with_score: List[Tuple[Dict, float]] = \
                parse_vector_search_response(
                    search_resp,
                    self._index_details,
                    self.text_column,
                    ignore_cols=[],
                    document_class=dict
                )
            return docs_with_score

        metadata = ToolMetadata(
            name=self.name,
            description=self.description,
            fn_schema=VectorSearchRetrieverToolInput,
            return_direct=self.return_direct,
        )

        # Pass the function to FunctionTool's init
        super().__init__(fn=similarity_search, metadata=metadata)

