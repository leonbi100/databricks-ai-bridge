from databricks_langchain import DatabricksVectorSearch
from typing import (
    List,
    Optional
)
from langchain_core.embeddings import Embeddings
from langchain.tools.retriever import create_retriever_tool

from typing import Any, Dict, Optional, Type

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)

class VectorSearchRetrieverToolInput(BaseModel):
    query: str = Field(description="query used to search the index")

class VectorSearchRetrieverTool():
    """
    A utility class to create a vector search-based retrieval tool for querying indexed embeddings.
    This class integrates with a Databricks Vector Search and provides a convenient interface
    for building a retriever tool for agents.

    Parameters:
        index_name (str):
            The name of the index to use. Format: “catalog.schema.index”. endpoint:
        num_results (int):
            The number of results to return. Defaults to 10.
        columns (Optional[List[str]]):
            The list of column names to get when doing the search. Defaults to [primary_key, text_column].
        filters (Optional[Dict[str, Any]]):
            The filters to apply to the search. Defaults to None.
        query_type (str):
            The type of query to run. Defaults to "ANN".
        tool_name (str):
            The name of the retrieval tool to be created. This will be passed to the language model,
            so should be unique and somewhat descriptive.
        tool_description (str):
            A description of the tool's functionality. This will be passed to the language model,
            so should be descriptive.
    """
    name: str = Field(description="The name of the tool")
    description: str = Field(description="The description of the tool")
    args_schema: Type[BaseModel] = VectorSearchRetrieverToolInput
    def __init__(
        self,
        index_name: str,
        num_results: int = 10,
        *,
        columns: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None,
        query_type: str = "ANN",
        tool_name: Optional[str] = None,
        tool_description: Optional[str],  # TODO: By default the UC metadata for description, how do I get this info? Call using client?
    ):
        # Use the index name as the tool name if no tool name is provided
        self.name = index_name
        if tool_name:
            self.name = tool_name
        self.num_results = num_results
        self.columns = columns
        self.filters = filters
        self.query_type = query_type
        self.description = tool_description
        self.vector_store = DatabricksVectorSearch(index_name=index_name)

    def _run(
        self,
        query: str
    ) -> str:
        """Use the tool."""
        self.vector_store.similarity_search(query, self.num_results, self.columns, self.filters, self.query_type)
