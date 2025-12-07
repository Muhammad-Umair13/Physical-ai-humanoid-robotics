import qdrant_client
from qdrant_client import models
from sentence_transformers import SentenceTransformer
import uuid
from typing import List, Dict, Optional

class QdrantRAGStore:
    def __init__(self, host: str = "localhost", port: int = 6333):
        """
        Initialize Qdrant client and setup collection for storing textbook content
        """
        self.client = qdrant_client.QdrantClient(host=host, port=port)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight embedding model
        self.collection_name = "textbook_content"

        # Create collection if it doesn't exist
        self._create_collection()

    def _create_collection(self):
        """
        Create Qdrant collection for storing textbook content
        """
        try:
            collections = self.client.get_collections()
            collection_exists = any(col.name == self.collection_name for col in collections.collections)

            if not collection_exists:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
                )
                print(f"Created collection: {self.collection_name}")
            else:
                print(f"Collection {self.collection_name} already exists")
        except Exception as e:
            print(f"Error creating collection: {e}")

    def store_text(self, text: str, metadata: Dict = None) -> str:
        """
        Store a text snippet in the vector store
        """
        try:
            # Generate embedding for the text
            embedding = self.model.encode([text])[0].tolist()

            # Create a unique ID for this text
            text_id = str(uuid.uuid4())

            # Store in Qdrant
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=text_id,
                        vector=embedding,
                        payload={
                            "text": text,
                            "metadata": metadata or {}
                        }
                    )
                ]
            )

            return text_id
        except Exception as e:
            print(f"Error storing text: {e}")
            raise

    def retrieve_similar(self, query: str, limit: int = 5) -> List[Dict]:
        """
        Retrieve similar text snippets based on the query
        """
        try:
            # Generate embedding for the query
            query_embedding = self.model.encode([query])[0].tolist()

            # Search in Qdrant
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit
            )

            # Extract the relevant text snippets
            results = []
            for result in search_results:
                results.append({
                    "text": result.payload["text"],
                    "score": result.score,
                    "metadata": result.payload.get("metadata", {})
                })

            return results
        except Exception as e:
            print(f"Error retrieving similar texts: {e}")
            raise

    def delete_text(self, text_id: str):
        """
        Delete a text snippet by ID
        """
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(
                    points=[text_id]
                )
            )
        except Exception as e:
            print(f"Error deleting text: {e}")
            raise

# Example usage
if __name__ == "__main__":
    # Initialize the store
    store = QdrantRAGStore()

    # Example: Store some text
    text_id = store.store_text(
        "ROS 2 (Robot Operating System 2) is a flexible framework for writing robot software. It is a collection of tools, libraries, and conventions that aim to simplify the task of creating complex and robust robot behavior across a wide variety of robot platforms.",
        {"module": "module1-ros2", "chapter": "introduction-to-ros2"}
    )
    print(f"Stored text with ID: {text_id}")

    # Example: Retrieve similar content
    results = store.retrieve_similar("What is ROS 2?")
    print("Retrieved results:", results)