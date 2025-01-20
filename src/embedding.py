from typing import List
from langchain_ollama import OllamaEmbeddings

class EmbeddingModel:
    def __init__(self, model_name: str = "llama3.2"):
        """
        Initialize embedding model with LangChain OllamaEmbeddings
        
        Args:
            model_name (str): Name of the model to use
        """
        self.model_name = model_name
        self.embeddings = OllamaEmbeddings(
            model=model_name
        )

    def embed(self, text: str) -> List[float]:
        """
        Generate embeddings for a single text input
        
        Args:
            text (str): Input text to embed
            
        Returns:
            List[float]: Embedding vector
        """
        try:
            # Use embed_query for single text embedding
            return self.embeddings.embed_query(text)
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return []

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts
        
        Args:
            texts (List[str]): List of input texts to embed
            
        Returns:
            List[List[float]]: List of embedding vectors
        """
        try:
            # Use embed_documents for batch embedding
            return self.embeddings.embed_documents(texts)
        except Exception as e:
            print(f"Error generating batch embeddings: {e}")
            return []
        
if __name__ == "__main__":
        # Initialize the embedding model
    embedding_model = EmbeddingModel(model_name="llama3.2")

    # Generate embedding for a single text
    single_text = "The meaning of life is 42"
    vector = embedding_model.embed(single_text)
    print(vector[:3])  # Print first 3 dimensions

    # Generate embeddings for multiple texts
    texts = ["Document 1...", "Document 2..."]
    vectors = embedding_model.embed_batch(texts)
    print(len(vectors))  # Number of vectors
    print(vectors[0][:3])  # First 3 dimensions of first vector