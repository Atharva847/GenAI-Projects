from langchain.tools import tool

import requests
from bs4 import BeautifulSoup
from tavily import TavilyClient
import os
import re
import trafilatura

from rich import print 
from dotenv import load_dotenv
load_dotenv()

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

#CREATE TOOLS USING DECORATORS

@tool
def web_search(query : str) -> str:

    """Search the web for recent and reliable information. Returns Titles, URLs and snippets"""
    results = tavily.search(query=query, max_results=5)
    output = []

    for i in results['results']:
        output.append(
            f"Title : {i['title']}\nURL : {i['url']}\nSnippet : {i['content'][:300]}\n"
        )
    return "\n----\n".join(output)


@tool
def scrape_url(url: str) -> str:
    """Scrape and return clean text content from a given URL for deeper reading."""
    try:
        resp = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        return soup.get_text(separator=" ", strip=True)[:3500]
    except Exception as e:
        return f"Could not scrape URL: {str(e)}"
    
