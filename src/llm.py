from typing import List, Dict, Optional
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate

class LLMChat:
    def __init__(self, model_name: str = "llama3.2", temperature: float = 0):
        """
        Initialize LLMChat with LangChain ChatOllama
        
        Args:
            model_name (str): Name of the model to use
            temperature (float): Temperature parameter for response generation
        """
        self.model_name = model_name
        self.llm = ChatOllama(
            model=model_name,
            temperature=temperature
        )
        self.history: List[Dict[str, str]] = []

    def chat_once(self, message: str):
        """
        Single chat interaction without maintaining history
        
        Args:
            message (str): User input message
            
        Returns:
            str: Model's response
        """
        try:
            # Create a simple prompt template for single messages
            prompt = ChatPromptTemplate.from_messages([
                ("human", "{input}")
            ])
            
            # Create and invoke the chain
            chain = prompt | self.llm
            response = chain.invoke({"input": message})
            
            return response.content
        except Exception as e:
            print(f"Error in chat: {e}")
            return ""

    def chat_with_history(self, message: str):
        """
        Chat interaction maintaining conversation history
        
        Args:
            message (str): User input message
            
        Returns:
            str: Model's response
        """
        try:
            # Add user message to history
            self.history.append({'role': 'human', 'content': message})
            
            # Convert history to LangChain message format
            messages = [
                HumanMessage(content=msg['content']) if msg['role'] == 'human'
                else AIMessage(content=msg['content'])
                for msg in self.history
            ]
            
            # Get response using chat method
            response = self.llm.invoke(messages)
            assistant_message = response.content
            
            # Add assistant response to history
            self.history.append({'role': 'assistant', 'content': assistant_message})
            
            return assistant_message
        except Exception as e:
            print(f"Error in chat with history: {e}")
            return ""

    def chat_with_template(self, template_messages: List[Dict[str, str]], 
                         input_variables: Dict[str, str]):
        """
        Chat using a custom template
        
        Args:
            template_messages (List[Dict[str, str]]): List of template messages
            input_variables (Dict[str, str]): Variables to fill in the template
            
        Returns:
            str: Model's response
        """
        try:
            # Create prompt template from messages
            prompt = ChatPromptTemplate.from_messages([
                (msg['role'], msg['content'])
                for msg in template_messages
            ])
            
            # Create and invoke the chain
            chain = prompt | self.llm
            response = chain.invoke(input_variables)
            
            return response.content
        except Exception as e:
            print(f"Error in template chat: {e}")
            return ""

    def clear_history(self):
        """Clear the conversation history"""
        self.history = []

    def get_history(self) -> List[Dict[str, str]]:
        """Return the current conversation history"""
        return self.history
    
if __name__ == "__main__":
    # Initialize the chat
    chat = LLMChat(model_name="llama3.1", temperature=0)

    # Example of using a template for translation
    template_messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that translates {input_language} to {output_language}."
        },
        {
            "role": "human",
            "content": "{input}"
        }
    ]

    input_vars = {
        "input_language": "English",
        "output_language": "German",
        "input": "I love programming."
    }

    response = chat.chat_with_template(template_messages, input_vars)
    # Simple chat without history
    response = chat.chat_once("Hello!")

    # Chat with history
    response = chat.chat_with_history("How are you?")