from langchain_community.document_loaders import WebBaseLoader
from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchResults

search_tool = DuckDuckGoSearchResults(output_format="json")


# Tool 2: Web Content Loader
@tool
def load_website_content(urls: list) -> list:
    """
    Loads and returns textual content from a list of URLs.
    Each URL's content is returned as a document chunk.
    """
    loader = WebBaseLoader(urls)
    docs = loader.load()
    return [doc.page_content[:800] for doc in docs]