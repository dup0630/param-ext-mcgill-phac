"""
rag.py

Provides a wrapper class for working with ChromaDB to enable retrieval-augmented generation (RAG).

The `ChromaRetriever` class manages creation and querying of a persistent vector database
backed by OpenAI's Azure-hosted embedding models. Each collection stores document chunks 
(e.g., sections from PDFs), associated with a unique `paper_id`.

Environment Variables Required:
- OPENAI_KEY: API key for Azure OpenAI
- OPENAI_EMBEDDING_ENDPOINT: Endpoint URL
- OPENAI_EMBEDDING_VERSION: API version (e.g., "2023-05-15")

Dependencies:
- chromadb
- openai
- python-dotenv
"""

from dotenv import load_dotenv
import os
import chromadb
import chromadb.utils.embedding_functions as embedding_functions

class ChromaRetriever:
    """Retrieves chunks from an existing Chroma vector database."""
    def __init__(self, db_name: str = "papers", emb_model: str = "text-embedding-3-large"):
        self.db_name = db_name
        self.emb_model = emb_model
        self.client = chromadb.PersistentClient()

    def create_db(self, dist_fn: str = "cosine") -> None:
        # Initialize OpenAI embedding function
        load_dotenv()
        key = os.getenv("OPENAI_KEY")
        endpoint = os.getenv("OPENAI_EMBEDDING_ENDPOINT")
        version = os.getenv("OPENAI_EMBEDDING_VERSION")
        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key = key,
            api_base = endpoint,
            api_type = "azure",
            api_version = version,
            model_name = self.emb_model
        )

        try:
            self.client.delete_collection(name=self.db_name)
        except:
            pass

        self.collection = self.client.create_collection(
            name = self.db_name, 
            embedding_function = openai_ef,
            metadata={
                "hnsw:space": dist_fn,
                #"hnsw:search_ef": 100 # determines the size of the dynamic candidate list used
            }
        )

    def add_paper_data(self, sections: list[str], paper_id: int, section_ids: list[int] = None) -> None:
        if not hasattr(self, "collection"):
            self.collection = self.client.get_collection(name=self.db_name)
        n = len(sections)
        template = {"paper_id" : paper_id}
        paper_ids = [template.copy() for _ in range(n)]
        
        # Include section IDs if provided; pass an enumeration otherwise
        if section_ids is None:
            section_ids = [f"paper{paper_id}section{i}" for i in range(n)]
        else:
            section_ids = [f"paper{paper_id}section{i}" for i in section_ids]
        
        self.collection.add(documents=sections, metadatas=paper_ids, ids=section_ids)
    
    def retrieve_from_paper(self, query: list[str], paper_id: str, n_results: int = 10) -> list[str]:
        results = self.collection.query(
            query_texts = query,
            n_results = n_results,
            where={"paper_id": paper_id},
            #where_document={"$contains":"search_string"} # Another possible filter
        )
        return(results)