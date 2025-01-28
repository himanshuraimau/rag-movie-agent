import json
from typing import Any, Optional, Type
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from crewai.tools import BaseTool
from pydantic import BaseModel, Field, ConfigDict

class ChromaToolSchema(BaseModel):
    """Input for ChromaTool."""

    query: str = Field(
        ...,
        description="The query to search and retrieve relevant information from the Chroma database.",
    )
    filter_by: Optional[str] = Field(
        default=None,
        description="Filter by properties. Pass only the properties, not the question.",
    )
    filter_value: Optional[str] = Field(
        default=None,
        description="Filter by value. Pass only the value, not the question.",
    )


class ChromaVectorSearchTool(BaseTool):
    """Tool to query, and if needed filter results from a Chroma database"""
    
    name: str = "ChromaVectorSearchTool"
    description: str = "A tool to search the Chroma database for relevant information on internal documents."
    args_schema: Type[BaseModel] = ChromaToolSchema
    
    # Add model_config to allow arbitrary types
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(
        self, 
        collection_name: str, 
        limit: int = 3,
        embeddings: Optional[Any] = None
    ):
        # Initialize the BaseTool first
        super().__init__()
        
        # Set up embeddings (use provided or default)
        self.embeddings = embeddings or OllamaEmbeddings(model="nomic-embed-text")
        
        # Store collection details
        self.collection_name = collection_name
        self.limit = limit
        
        # Initialize vectorstore
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory="chroma_db"  # Ensure this matches your setup
        )

    def _run(
        self,
        query: str,
        filter_by: Optional[str] = None,
        filter_value: Optional[str] = None,
    ) -> str:
        # Create retriever with search parameters
        retriever = self.vectorstore.as_retriever(
            search_kwargs={
                "k": self.limit,
                **({"filter": {filter_by: filter_value}} if filter_by and filter_value else {})
            }
        )

        # Use invoke instead of retrieve
        results = retriever.invoke(query)

        # Convert results to JSON
        json_response = []
        for doc in results:
            json_response.append({
                "page_content": doc.page_content,
                "metadata": doc.metadata
            })

        return json.dumps(json_response, indent=2)


if __name__ == "__main__":
    tool = ChromaVectorSearchTool(
        collection_name="netflix_data"  # Update with your collection name
    )
    result = tool.run("Find me similar shows to How I Met Your Mother?")
    print("result", result)
