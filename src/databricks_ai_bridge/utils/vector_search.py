import json
from enum import Enum
from typing import Any, Dict, Optional

class IndexType(str, Enum):
    DIRECT_ACCESS = "DIRECT_ACCESS"
    DELTA_SYNC = "DELTA_SYNC"

class IndexDetails:
    """An utility class to store the configuration details of an index."""

    def __init__(self, index: Any):
        self._index_details = index.describe()

    @property
    def name(self) -> str:
        return self._index_details["name"]

    @property
    def schema(self) -> Optional[Dict]:
        if self.is_direct_access_index():
            schema_json = self.index_spec.get("schema_json")
            if schema_json is not None:
                return json.loads(schema_json)
        return None

    @property
    def primary_key(self) -> str:
        return self._index_details["primary_key"]

    @property
    def index_spec(self) -> Dict:
        return (
            self._index_details.get("delta_sync_index_spec", {})
            if self.is_delta_sync_index()
            else self._index_details.get("direct_access_index_spec", {})
        )

    @property
    def embedding_vector_column(self) -> Dict:
        if vector_columns := self.index_spec.get("embedding_vector_columns"):
            return vector_columns[0]
        return {}

    @property
    def embedding_source_column(self) -> Dict:
        if source_columns := self.index_spec.get("embedding_source_columns"):
            return source_columns[0]
        return {}

    def is_delta_sync_index(self) -> bool:
        return self._index_details["index_type"] == IndexType.DELTA_SYNC.value

    def is_direct_access_index(self) -> bool:
        return self._index_details["index_type"] == IndexType.DIRECT_ACCESS.value

    def is_databricks_managed_embeddings(self) -> bool:
        return self.is_delta_sync_index() and self.embedding_source_column.get("name") is not None

class DatabricksVectorSearchMixin:
    def parse_vector_search_response(
        self, search_resp: Dict, ignore_cols: Optional[List[str]] = None, document_class: Any = dict
    ) -> List[Tuple[Dict, float]]:
        """
        Parse the search response into a list of Documents with score.
        The document_class parameter is used to specify the class of the document to be created.
        """
        if ignore_cols is None:
            ignore_cols = []

        columns = [col["name"] for col in search_resp.get("manifest", dict()).get("columns", [])]
        docs_with_score = []
        for result in search_resp.get("result", dict()).get("data_array", []):
            doc_id = result[columns.index(self._index_details.primary_key)]
            text_content = result[columns.index(self._text_column)]
            ignore_cols = [self._primary_key, self._text_column] + ignore_cols
            metadata = {
                col: value
                for col, value in zip(columns[:-1], result[:-1])
                if col not in ignore_cols
            }
            metadata[self._primary_key] = doc_id
            score = result[-1]
            doc = document_class(page_content=text_content, metadata=metadata)
            docs_with_score.append((doc, score))
        return docs_with_score

def _validate_and_get_text_column(text_column: Optional[str], index_details: IndexDetails) -> str:
    if index_details.is_databricks_managed_embeddings():
        index_source_column: str = index_details.embedding_source_column["name"]
        # check if input text column matches the source column of the index
        if text_column is not None:
            raise ValueError(
                f"The index '{index_details.name}' has the source column configured as "
                f"'{index_source_column}'. Do not pass the `text_column` parameter."
            )
        return index_source_column
    else:
        if text_column is None:
            raise ValueError("The `text_column` parameter is required for this index.")
        return text_column