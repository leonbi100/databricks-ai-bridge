from typing import Any, Dict, List, Optional, Type

from openai import pydantic_function_tool
from openai.types.chat import ChatCompletionToolParam
from openai.types.chat import ChatCompletion

from pydantic import BaseModel, Field, PrivateAttr, model_validator, create_model

from databricks_ai_bridge.vector_search_retriever_tool import VectorSearchRetrieverToolMixin, VectorSearchRetrieverToolInput, DEFAULT_TOOL_DESCRIPTION
from databricks_ai_bridge.utils.vector_search import IndexDetails, DatabricksVectorSearchMixin, _validate_and_get_text_column
from databricks.vector_search.client import VectorSearchClient, VectorSearchIndex
import json

class VectorSearchRetrieverTool(VectorSearchRetrieverToolMixin, DatabricksVectorSearchMixin):
    """
    A utility class to create a vector search-based retrieval tool for querying indexed embeddings.
    This class integrates with Databricks Vector Search and provides a convenient interface
    for tool calling using the OpenAI SDK.

    Example:
        dbvs_tool = VectorSearchRetrieverTool("index_name")
        tools = [dbvs_tool.tool, ...]
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=initial_messages,
            tools=tools,
        )
        retriever_call_message = dbvs_tool.execute_retriever_calls(response)

        ### If needed, execute potential remaining tool calls here ###
        remaining_tool_call_messages = execute_remaining_tool_calls(response)

        final_response = openai.chat.completions.create(
            model="gpt-4o",
            messages=initial_messages + retriever_call_message + remaining_tool_call_messages
            tools=tools,
        )
        final_response.choices[0].message.content
    """

    embedding: Optional[Embeddings] = Field(
        None, description="Embedding model for self-managed embeddings. Used for direct "
                          "access indexes or delta-sync indexes with self-managed embeddings"
    )
    text_column: Optional[str] = Field(
        None,
        description="The name of the text column to use for the embeddings. "
                    "Required for direct-access index or delta-sync index with "
                    "self-managed embeddings. Used for direct access indexes or "
                    "delta-sync indexes with self-managed embeddings",
    )

    tool: ChatCompletionToolParam = Field(
        ..., description="The tool input used in the OpenAI chat completion SDK"
    )
    _index: VectorSearchIndex = PrivateAttr()
    _index_details: IndexDetails = PrivateAttr()

    @model_validator(mode="after")
    def _validate_tool_inputs(self):
        self._index = VectorSearchClient().get_index(index_name=self.index_name)
        self._index_details = IndexDetails(self._index)
        _validate_and_get_text_column(self.text_column, self._index_details)

        self.tool: ChatCompletionToolParam = pydantic_function_tool(
            VectorSearchRetrieverToolInput,
            name=self.tool_name or self.index_name,
            description=self.tool_description or self._get_default_tool_description(self._index_details)
        )
        return self

    def execute_retriever_calls(self,
                               response: ChatCompletion,
                               choice_index: int = 0) -> List[Dict[str, Any]]:
        """
        Execute the VectorSearchIndex tool calls from the ChatCompletions response that correspond to the
        self.tool VectorSearchRetrieverToolInput and attach the retrieved documents into toll call messages.

        Args:
            response: The chat completion response object returned by the OpenAI API.
            choice_index: The index of the choice to process. Defaults to 0. Note that multiple
                choices are not supported yet.

        Returns:
            A list of messages containing the assistant message and the retriever call results
            that correspond to the self.tool VectorSearchRetrieverToolInput.
        """

        class Document:
            def __init__(self, page_content, metadata):
                self.page_content = page_content
                self.metadata = metadata

        def get_query_text_vector(query):
            if self._index_details.is_databricks_managed_embeddings():
                return query, None

            # For non-Databricks-managed embeddings
            text = query if query_type and query_type.upper() == "HYBRID" else None
            vector = self._embeddings.embed_query(query)  # type: ignore[union-attr]
            return text, vector

        def do_arguments_match(llm_call_arguments: Dict[str, Any]):
            # TODO: Remove print statement
            print("llm_call_arguments_json: ", llm_call_arguments)
            print("retriever_tool_params: ", self.tool.function.parameters)
            retriever_tool_params: Dict[str, Any] = self.tool.function.parameters
            return set(llm_call_arguments.keys()) == set(retriever_tool_params.keys())

        message = response.choices[choice_index].message
        llm_tool_calls = message.tool_calls
        function_calls = []
        if llm_tool_calls:
            for llm_tool_call in llm_tool_calls:
                # Only process tool calls that correspond to the self.tool VectorSearchRetrieverToolInput
                llm_call_arguments_json: Dict[str, Any] = json.loads(llm_tool_call.function.arguments)
                if llm_tool_call.function.name != self.tool.function.name \
                        or not do_arguments_match(llm_call_arguments_json):
                    continue

                query_text, query_vector = get_query_text_vector(llm_call_arguments_json["query"])
                search_resp = self._index.similarity_search(
                    columns=self._columns,
                    query_text=query_text,
                    query_vector=query_vector,
                    filters=filter,
                    num_results=k,
                    query_type=query_type,
                )
                embedding_column = self._index_details.embedding_vector_column["name"]
                # Don't show the embedding column in the response
                ignore_cols: List = [embedding_column] if embedding_column not in self.columns else []
                docs_with_score: List[Tuple[Document, float]] = \
                    self.parse_vector_search_response(  # from DatabricksVectorSearchMixin
                        search_resp,
                        ignore_cols=ignore_cols,
                        document_class=Document
                    )

                function_call_result_message = {
                    "role": "tool",
                    "content": json.dumps({"content": docs_with_score}),
                    "tool_call_id": llm_tool_call.id,
                }
                function_calls.append(function_call_result_message)
        assistant_message = message.to_dict()
        return [assistant_message, *function_calls]
