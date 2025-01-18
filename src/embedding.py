from typing import List
from ollama import pull, embed
from tqdm import tqdm

class EmbeddingModel:
    def __init__(self, model_name: str = "llama3.2"):
        self.model_name = model_name
    
    def _download_model(self):
        """Download the model if not already present"""
        current_digest, bars = '', {}
        for progress in pull(self.model_name, stream=True):
            digest = progress.get('digest', '')
            if digest != current_digest and current_digest in bars:
                bars[current_digest].close()

            if not digest:
                print(progress.get('status'))
                continue

            if digest not in bars and (total := progress.get('total')):
                bars[digest] = tqdm(
                    total=total, 
                    desc=f'pulling {digest[7:19]}', 
                    unit='B', 
                    unit_scale=True
                )

            if completed := progress.get('completed'):
                bars[digest].update(completed - bars[digest].n)

            current_digest = digest

    def embed(self, text: str) -> List[float]:
        """
        Generate embeddings for a single text input
        
        Args:
            text (str): Input text to embed
            
        Returns:
            List[float]: Embedding vector
        """
        try:
            response = embed(model=self.model_name, input=text)
            return response['embeddings'][0]
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
            response = embed(model=self.model_name, input=texts)
            return response['embeddings']
        except Exception as e:
            print(f"Error generating batch embeddings: {e}")
            return []