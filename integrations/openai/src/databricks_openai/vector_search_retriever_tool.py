from typing import Any, Dict, List, Optional, Type

from openai import pydantic_function_tool
from openai.types import Embeddings
from openai.types.chat import ChatCompletionToolParam
from openai.types.chat import ChatCompletion

from pydantic import BaseModel, Field, PrivateAttr, model_validator, create_model

from databricks_ai_bridge.vector_search_retriever_tool import BaseVectorSearchRetrieverTool, VectorSearchRetrieverToolInput, DEFAULT_TOOL_DESCRIPTION
from databricks_ai_bridge.vectorstores import IndexDetails

class VectorSearchRetrieverTool(BaseVectorSearchRetrieverTool):
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
        new_messages = dbvs_tool.execute_retriever_call(response)
        final_response = openai.chat.completions.create(
            model="gpt-4o",
            messages=initial_messages + new_messages,
            tools=tools,
        )
        final_response.choices[0].message.content
    """

    embedding: Optional[Embeddings] = Field(
        None, description="Embedding model for self-managed embeddings."
    )
    tool: ChatCompletionToolParam = Field(
        ..., description="The tool input used in the OpenAI chat completion SDK"
    )
    _vector_store: DatabricksVectorSearch = PrivateAttr()

    @model_validator(mode="after")
    def _validate_tool_inputs(self):
        kwargs = {
            "index_name": self.index_name,
            "embedding": self.embedding,
            "text_column": self.text_column,
            "columns": self.columns,
        }
        dbvs = DatabricksVectorSearch(**kwargs)
        self._vector_store = dbvs
        self.tool = pydantic_function_tool(
            VectorSearchRetrieverToolInput,
            name=self.tool_name or self.index_name,
            description=self.tool_description or self._get_default_tool_description(),
        )
        return self

    def execute_retriever_call(self,
                               response: ChatCompletion,
                               choice_index: int = 0) -> List[Dict[str, Any]]:
        """
        Generate tool call messages from the response.

        Args:
            response: The chat completion response object returned by the OpenAI API.
            choice_index: The index of the choice to process. Defaults to 0. Note that multiple
                choices are not supported yet.

        Returns:
            A list of messages containing the assistant message and the function call results.
        """
        pass

        # client = validate_or_set_default_client(client)
        # message = response.choices[choice_index].message
        # tool_calls = message.tool_calls
        # function_calls = []
        # if tool_calls:
        #     for tool_call in tool_calls:
        #         arguments = json.loads(tool_call.function.arguments)
        #         func_name = construct_original_function_name(tool_call.function.name)
        #         result = client.execute_function(func_name, arguments)
        #         function_call_result_message = {
        #             "role": "tool",
        #             "content": json.dumps({"content": result.value}),
        #             "tool_call_id": tool_call.id,
        #         }
        #         function_calls.append(function_call_result_message)
        # assistant_message = message.to_dict()
        # return [assistant_message, *function_calls]
