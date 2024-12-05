from databricks_langchain import DatabricksVectorSearch
from typing import (
    List,
    Optional
)
from langchain_core.embeddings import Embeddings
from langchain.tools.retriever import create_retriever_tool

class VectorSearchRetrieverTool():
    """
    A utility class to create a vector search-based retrieval tool for querying indexed embeddings.
    This class integrates with a Databricks Vector Search and provides a convenient interface
    for building a retriever tool for agents.

    Parameters:
        tool_name (str):
            The name of the retrieval tool to be created. This will be passed to the language model,
            so should be unique and somewhat descriptive.
        tool_description (str):
            A description of the tool's functionality. This will be passed to the language model,
            so should be descriptive.

        index_name (str):
            The name of the index to use. Format: “catalog.schema.index”. endpoint:
        endpoint (Optional[str]):
            The name of the Databricks Vector Search endpoint. If not specified, the endpoint name is
            automatically inferred based on the index name.
        embedding (Optional[Embeddings]):
            The embedding model. Required for direct-access index or delta-sync index with self-managed embeddings.
        text_column (Optional[str]):
            The name of the text column to use for the embeddings. Required for direct-access index or
            delta-sync index with self-managed embeddings. Make sure the text column specified is in the index.
        columns (Optional[List[str]]):
            The list of column names to get when doing the search. Defaults to [primary_key, text_column].

        search_type (Optional[str]): Defines the type of search that the Retriever should perform.
            Defaults to “similarity” (default).
        search_kwargs (Optional[Dict]): Keyword arguments to pass to the search function.
    """
    def __new__(
            self,
            tool_name = str,
            tool_description = str,
            index_name: str,
            endpoint: Optional[str] = None,
            embedding: Optional[Embeddings] = None,
            text_column: Optional[str] = None,
            columns: Optional[List[str]] = None,
            search_type: Optional[dict] = None,
    ):
        vector_store = DatabricksVectorSearch(
            endpoint=endpoint,
            index_name=index_name,
            embedding = embedding,
            text_column = text_column,
            columns = columns
        )
        retriever = vector_store.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )
        return create_retriever_tool(
            retriever,
            tool_name,
            tool_description,
        )