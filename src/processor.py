from typing import List, Dict, Any
from tqdm import tqdm
import time

from src.embedding import EmbeddingModel
from src.llm import LLMChat

class TableProcessor:
    def __init__(self, llm_model: LLMChat, embedding_model: EmbeddingModel, batch_size: int = 8):
        """
        Initialize the TableProcessor with pre-initialized models.
        
        Args:
            llm_model (LLMChat): Initialized LLM model
            embedding_model (EmbeddingModel): Initialized embedding model
            batch_size (int): Batch size for processing embeddings
        """
        self.llm = llm_model
        self.embedder = embedding_model
        self.batch_size = batch_size
    
    def get_table_description(self, markdown_table: str) -> str:
        """
        Generate description for a single markdown table using Ollama chat.
        
        Args:
            markdown_table (str): Input markdown table
            
        Returns:
            str: Generated description of the table
        """
        system_prompt = """You are an AI language model. Your task is to examine the provided table, taking into account both its rows and columns, and produce a concise summary of up to 200 words. Emphasize key patterns, trends, and notable data points that provide meaningful insights into the content of the table."""
        
        try:
            # Use chat_once to avoid maintaining history between tables
            full_prompt = f"{system_prompt}\n\nTable:\n{markdown_table}"
            return self.llm.chat_once(full_prompt)
        except Exception as e:
            print(f"Error generating table description: {e}")
            return ""
    
    def process_tables(self, markdown_tables) -> List[Dict[str, Any]]:
        """
        Process a list of markdown tables: generate descriptions and embeddings.
        
        Args:
            markdown_tables (List[str]): List of markdown tables to process
            
        Returns:
            List[Dict[str, Any]]: List of dictionaries containing processed information
        """
        results = []
        descriptions = []
        
        # Generate descriptions for all tables
        with tqdm(total=len(markdown_tables), desc="Generating table descriptions") as pbar:
            for i, table in enumerate(markdown_tables):
                description = self.get_table_description(table.text)
                print(f"\nTable {i+1}:")
                print(f"Description: {description}")
                print("-" * 50)
                descriptions.append(description)
                pbar.update(1)
                time.sleep(1)  # Rate limiting
            
        # Generate embeddings in batches
        embeddings = []
        total_batches = (len(descriptions) + self.batch_size - 1) // self.batch_size
        
        with tqdm(total=total_batches, desc="Generating embeddings") as pbar:
            for i in range(0, len(descriptions), self.batch_size):
                batch = descriptions[i:i + self.batch_size]
                if len(batch) == 1:
                    batch_embeddings = [self.embedder.embed(batch[0])]
                else:
                    batch_embeddings = self.embedder.embed_batch(batch)
                embeddings.extend(batch_embeddings)
                pbar.update(1)
        
        # Combine results with progress bar
        with tqdm(total=len(markdown_tables), desc="Combining results") as pbar:
            for table, description, embedding in zip(markdown_tables, descriptions, embeddings):
                results.append({
                    "embedding": embedding,
                    "text": table,
                    "table_description": description,
                    "type": "table_chunk"
                })
                pbar.update(1)
            
        return results

    def __call__(self, markdown_tables) -> List[Dict[str, Any]]:
        """
        Make the class callable for easier use.
        
        Args:
            markdown_tables (List[str]): List of markdown tables to process
            
        Returns:
            List[Dict[str, Any]]: Processed results
        """
        return self.process_tables(markdown_tables)