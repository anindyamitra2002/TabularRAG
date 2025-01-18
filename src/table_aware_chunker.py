from typing import List, Tuple, Union
import re
from dataclasses import dataclass
from chonkie.chunker import RecursiveChunker
from chonkie.types import RecursiveChunk
from chonkie import RecursiveRules

@dataclass
class TableChunk:
    """Represents a table chunk from the markdown document."""
    text: str
    start_index: int
    end_index: int
    token_count: int

class TableRecursiveChunker(RecursiveChunker):
    """A recursive chunker that preserves markdown tables while chunking text.
    
    This chunker extends the base RecursiveChunker to handle markdown tables as special cases,
    keeping them intact rather than splitting them according to the recursive rules.
    """

    def _extract_tables(self, text: str) -> Tuple[List[TableChunk], List[Tuple[int, int, str]]]:
        """
        Extract markdown tables from text and return table chunks and remaining text segments.
        
        Args:
            text: The input text containing markdown content
            
        Returns:
            Tuple containing:
            - List of TableChunk objects for tables
            - List of (start_index, end_index, text) tuples for non-table segments
        """
        # Regular expression for markdown tables (matches header, separator, and content rows)
        table_pattern = r'(\|[^\n]+\|\n\|[-:\|\s]+\|\n(?:\|[^\n]+\|\n)+)'
        
        table_chunks = []
        non_table_segments = []
        last_end = 0
        
        for match in re.finditer(table_pattern, text):
            start, end = match.span()
            
            # Add non-table text before this table
            if start > last_end:
                non_table_segments.append((last_end, start, text[last_end:start]))
            
            # Create table chunk
            table_text = match.group()
            token_count = self._count_tokens(table_text)
            table_chunks.append(TableChunk(
                text=table_text,
                start_index=start,
                end_index=end,
                token_count=token_count
            ))
            
            last_end = end
        
        # Add remaining text after last table
        if last_end < len(text):
            non_table_segments.append((last_end, len(text), text[last_end:]))
            
        return table_chunks, non_table_segments

    def chunk(self, text: str) -> Tuple[List[RecursiveChunk], List[TableChunk]]:
        """
        Chunk the text while preserving tables.
        
        This method overrides the base chunk method to handle tables separately from
        regular text content.
        
        Args:
            text: The input text to chunk
            
        Returns:
            Tuple containing:
            - List of RecursiveChunk objects for non-table text
            - List of TableChunk objects for tables
        """
        # First extract tables
        table_chunks, non_table_segments = self._extract_tables(text)
        
        # Chunk each non-table segment using the parent class's recursive chunking
        text_chunks = []
        for start, end, segment in non_table_segments:
            if segment.strip():  # Only process non-empty segments
                # Use the parent class's recursive chunking logic
                chunks = super()._recursive_chunk(segment, level=0, full_text=text)
                text_chunks.extend(chunks)
        
        return text_chunks, table_chunks

    def chunk_batch(self, texts: List[str]) -> List[Tuple[List[RecursiveChunk], List[TableChunk]]]:
        """
        Chunk multiple texts while preserving tables in each.
        
        Args:
            texts: List of texts to chunk
            
        Returns:
            List of tuples, each containing:
            - List of RecursiveChunk objects for non-table text
            - List of TableChunk objects for tables
        """
        return [self.chunk(text) for text in texts]

    def __call__(self, texts: Union[str, List[str]]) -> Union[
        Tuple[List[RecursiveChunk], List[TableChunk]],
        List[Tuple[List[RecursiveChunk], List[TableChunk]]]
    ]:
        """Make the chunker callable for convenience."""
        if isinstance(texts, str):
            return self.chunk(texts)
        return self.chunk_batch(texts)
    
