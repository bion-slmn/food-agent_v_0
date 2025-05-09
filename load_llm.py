from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()


groq_api_key = os.environ.get("GROQ_API_KEY")

llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=groq_api_key)
