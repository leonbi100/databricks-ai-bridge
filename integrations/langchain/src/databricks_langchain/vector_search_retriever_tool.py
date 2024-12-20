from typing import Any, Dict, List, Optional, Type

from langchain_core.embeddings import Embeddings
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr, model_validator

from databricks_langchain.utils import IndexDetails
from databricks_langchain.vectorstores import DatabricksVectorSearch


class VectorSearchRetrieverToolInput(BaseModel):
    query: str = Field(
        description="The string used to query the index with and identify the most similar "
        "vectors and return the associated documents."
    )


class VectorSearchRetrieverTool(BaseTool):
    """
    A utility class to create a vector search-based retrieval tool for querying indexed embeddings.
    This class integrates with a Databricks Vector Search and provides a convenient interface
    for building a retriever tool for agents.
    """

    index_name: str = Field(
        ..., description="The name of the index to use, format: 'catalog.schema.index'."
    )
    num_results: int = Field(10, description="The number of results to return.")
    columns: Optional[List[str]] = Field(
        None, description="Columns to return when doing the search."
    )
    filters: Optional[Dict[str, Any]] = Field(None, description="Filters to apply to the search.")
    query_type: str = Field(
        "ANN", description="The type of this query. Supported values are 'ANN' and 'HYBRID'."
    )
    tool_name: Optional[str] = Field(None, description="The name of the retrieval tool.")
    tool_description: Optional[str] = Field(None, description="A description of the tool.")
    text_column: Optional[str] = Field(
        None,
        description="The name of the text column to use for the embeddings. "
        "Required for direct-access index or delta-sync index with "
        "self-managed embeddings.",
    )
    embedding: Optional[Embeddings] = Field(
        None, description="Embedding model for self-managed embeddings."
    )

    # The BaseTool class requires 'name' and 'description' fields which we will populate in validate_tool_inputs()
    name: str = Field(default="", description="The name of the tool")
    description: str = Field(default="", description="The description of the tool")
    args_schema: Type[BaseModel] = VectorSearchRetrieverToolInput

    _vector_store: DatabricksVectorSearch = PrivateAttr()

    @model_validator(mode="after")
    def validate_tool_inputs(self):
        kwargs = {
            "index_name": self.index_name,
            "embedding": self.embedding,
            "text_column": self.text_column,
            "columns": self.columns,
        }
        dbvs = DatabricksVectorSearch(**kwargs)
        self._vector_store = dbvs

        def get_tool_description():
            default_tool_description = (
                "A vector search-based retrieval tool for querying indexed embeddings."
            )
            index_details = IndexDetails(dbvs.index)
            if index_details.is_delta_sync_index():
                from databricks.sdk import WorkspaceClient

                source_table = index_details.index_spec.get("source_table", "")
                w = WorkspaceClient()
                source_table_comment = w.tables.get(full_name=source_table).comment
                if source_table_comment:
                    return (
                        default_tool_description
                        + f" The queried index uses the source table {source_table} with the description: "
                        + source_table_comment
                    )
                else:
                    return (
                        default_tool_description
                        + f" The queried index uses the source table {source_table}"
                    )
            return default_tool_description

        self.name = self.tool_name or self.index_name
        self.description = self.tool_description or get_tool_description()

        return self

    def _run(self, query: str) -> str:
        return self._vector_store.similarity_search(
            query, k=self.num_results, filter=self.filters, query_type=self.query_type
        )
