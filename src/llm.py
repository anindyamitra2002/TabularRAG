from typing import List, Dict, Optional
from ollama import pull, chat
from tqdm import tqdm

class LLMChat:
    def __init__(self, model_name: str = "llama3.2"):
        self.model_name = model_name
        self.history: List[Dict[str, str]] = []
        # self._download_model()
    
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

    def chat_once(self, message: str) -> str:
        """
        Single chat interaction without maintaining history
        
        Args:
            message (str): User input message
            
        Returns:
            str: Model's response
        """
        try:
            messages = [{'role': 'user', 'content': message}]
            response = chat(self.model_name, messages=messages)
            return response['message']['content']
        except Exception as e:
            print(f"Error in chat: {e}")
            return ""

    def chat_with_history(self, message: str) -> str:
        """
        Chat interaction maintaining conversation history
        
        Args:
            message (str): User input message
            
        Returns:
            str: Model's response
        """
        try:
            # Add user message to history
            self.history.append({'role': 'user', 'content': message})
            
            # Get response
            response = chat(self.model_name, messages=self.history)
            assistant_message = response['message']['content']
            
            # Add assistant response to history
            self.history.append({'role': 'assistant', 'content': assistant_message})
            
            return assistant_message
        except Exception as e:
            print(f"Error in chat with history: {e}")
            return ""

    def clear_history(self):
        """Clear the conversation history"""
        self.history = []

    def get_history(self) -> List[Dict[str, str]]:
        """Return the current conversation history"""
        return self.history