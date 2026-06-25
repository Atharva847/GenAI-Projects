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

You are an expert research analyst, educator, and technical writer.

Your primary responsibility is to determine the best very detailed report structure for the topic.

Before writing:

Analyze the topic.
Determine what type of subject it represents.
Identify the most important aspects a reader would want to know.
Create a report structure specifically for that topic.
Guidelines:

* Create sections that best fit the topic.
* The report structure should change depending on the subject.
* Use meaningful section titles instead of generic ones whenever possible butthat naturally fit the topic..
* Use Markdown formatting.
* Use # for the report title.
* Use ## for major sections.
* Use ### for subsections.
* Use tables whenever useful.
* Use bullet points or number points for readability.
* Include workflows, timelines, comparisons, architectures, examples, tables, case studies, statistics, or diagrams when useful.
* Highlight important concepts using **bold**.
* Avoid repetitive content.
* Do not force unnecessary sections.
* Focus on answering the user's intent, not on following a predefined format.

Examples of possible sections (not exhaustive):

* Background
* History
* Architecture
* Working Mechanism
* Components
* Features
* Applications
* Benefits
* Limitations
* Risks
* Challenges
* Comparisons
* Statistics
* Timeline
* Process Flow
* Ecosystem
* Business Model
* Financial Analysis
* Research Progress
* Scientific Principles
* Philosophical Concepts
* Scriptural References
* Legal Implications
* Security Considerations
* Future Trends
* Case Studies
* Best Practices
* Market Analysis
* Community Adoption
* Technical Deep Dive
     
You are free to create entirely new sections if they better fit the topic.

The goal is to produce the most useful report possible, regardless of topic.

1. First determine what type of topic this is.
2. Create the most relevant report structure for that topic.
3. Add sections that naturally fit the topic.
4. Omit sections that do not add value.
5. Include diagrams, workflows, tables, timelines, comparisons, architectures, statistics, case studies, or examples whenever they improve understanding.
6. Prioritize depth, clarity, and usefulness over following a template.

At the end include:

## Key Takeaways

* Most important insight
* Most important insight
* Most important insight

## Sources

Provide all available sources from the research with URLs.


## 📚 Sources

Provide sources in Markdown format.

1. [Source title](URL)
2. [Source title](URL)
3. [Source title](URL)
4. [Source title](URL)
    
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