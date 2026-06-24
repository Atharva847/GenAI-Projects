from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tools import web_search , scrape_url 
from dotenv import load_dotenv

load_dotenv()

def get_llm(
    model_name,
    api_key=None,
    provider="openrouter",
    ollama_base_url="http://localhost:11434/v1",
):

    if provider == "ollama":
        return ChatOpenAI(
            model=model_name,
            api_key=api_key or "ollama",
            base_url=ollama_base_url,
        )

    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )

#1st agent 
def build_search_agent(llm):
    return create_agent(
        model = llm,
        tools= [web_search]
    )

#2nd agent 

def build_reader_agent(llm):
    return create_agent(
        model = llm,
        tools = [scrape_url]
    )


#writer chain 
writer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert research writer. Write clear, structured and insightful reports using clean Markdown formatting."),
    ("human", """Write a detailed research report on the topic below.

Topic: {topic}

Research Gathered:
{research}

You are an expert research writer.

Generate a professional, well-structured, and visually appealing research report.

Instructions:

- Use Markdown formatting.
- Use # for the report title.
- Use ## for major sections.
- Use ### for subsections.
- Keep paragraphs concise and easy to read.
- Use bullet points instead of long paragraphs whenever possible.
- Use tables when comparison or numerical data is useful.
- Highlight important concepts using **bold** text.
- Avoid repeating information.
- Maintain a factual, professional, and engaging tone.
- Do NOT force sections that are irrelevant to the topic.

The report should dynamically include ONLY relevant sections from the following:

## 📌 Executive Summary
Include when the topic is broad, technical, or requires an overview.

## 🌍 Introduction
Provide background, importance, and real-world relevance.

## 🔍 Key Findings
Include 3-8 important findings depending on the complexity of the topic.

For each finding:

### Finding Title

**Overview**

Short explanation.

**Key Details**

- Point 1
- Point 2
- Point 3

**Why It Matters**

Brief explanation.

## 📊 Statistics & Data
Include ONLY if numerical data, trends, benchmarks, or measurements exist.

Use a table whenever appropriate.

| Metric | Value | Notes |
|--------|-------|-------|

## ⚖️ Advantages & Challenges
Include ONLY when the topic involves comparisons, systems, technologies, products, or methodologies.

| Advantages | Challenges |
|------------|------------|
| Advantage | Challenge |

## 🔄 Comparisons
Include ONLY when comparing multiple technologies, products, tools, approaches, or concepts.

## 🚀 Future Outlook
Include ONLY if future developments, predictions, trends, or opportunities are relevant.

## 🎯 Key Takeaways
Provide 3-5 concise takeaway points.

- Takeaway 1
- Takeaway 2
- Takeaway 3

## 📚 Sources

Provide sources in Markdown format.

1. [Source title](URL)
2. [Source title](URL)

Rules:

- Never invent sources.
- Never invent statistics.
- Only include sections that add value to the topic.
- Prioritize readability over length.
- Use tables only when beneficial.
- Adapt the report structure to the topic instead of forcing every section.
"""),
])

def build_writer_chain(llm):

    return (
        writer_prompt
        | llm
        | StrOutputParser()
    )

#critic_chain 

critic_prompt = ChatPromptTemplate.from_messages([
     ("system", "You are a sharp and constructive research critic. Be honest and specific."),
    ("human", """Review the research report below and evaluate it strictly.

Report:
{report}

Respond in this exact format:

Score: X/10

Strengths:
- ...
- ...

Areas to Improve:
- ...
- ...

One line verdict:
..."""),
])

def build_critic_chain(llm):

    return (
        critic_prompt
        | llm
        | StrOutputParser()
    )