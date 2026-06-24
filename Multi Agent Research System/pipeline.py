from agents import (
    get_llm,
    build_search_agent,
    build_reader_agent,
    build_writer_chain,
    build_critic_chain,
)


def run_research_pipeline(
    topic: str,
    model_name: str,
    api_key: str = None,
    provider: str = "openrouter",
    ollama_base_url: str = "http://localhost:11434/v1",
    on_step=None,
    on_stream=None,
) -> dict:

    state = {}

    # ==========================================
    # CREATE LLM
    # ==========================================

    llm = get_llm(
        model_name=model_name,
        api_key=api_key,
        provider=provider,
        ollama_base_url=ollama_base_url,
    )

    # ==========================================
    # BUILD AGENTS & CHAINS
    # ==========================================

    search_agent = build_search_agent(llm)
    reader_agent = build_reader_agent(llm)
    writer_chain = build_writer_chain(llm)
    critic_chain = build_critic_chain(llm)

    # ==========================================
    # STEP 1 - SEARCH AGENT
    # ==========================================

    print("\n" + "=" * 50)
    print("Step 1 - Search Agent")
    print("=" * 50)

    search_result = search_agent.invoke({
        "messages": [
            (
                "user",
                f"Find recent, reliable and detailed information about: {topic}"
            )
        ]
    })

    state["search_results"] = (
        search_result["messages"][-1].content
    )

    print(
        "\nSearch Results:\n",
        state["search_results"]
    )

    if on_step:
        on_step("search", state)

    # ==========================================
    # STEP 2 - READER AGENT
    # ==========================================

    print("\n" + "=" * 50)
    print("Step 2 - Reader Agent")
    print("=" * 50)

    reader_result = reader_agent.invoke({
        "messages": [
            (
                "user",
                f"Based on the following search results about '{topic}', "
                f"pick the most relevant URL and scrape it for deeper content.\n\n"
                f"Search Results:\n"
                f"{state['search_results'][:800]}"
            )
        ]
    })

    state["scraped_content"] = (
        reader_result["messages"][-1].content
    )
    print(
        "\nScraped Content:\n",
        state["scraped_content"]
    )

    if on_step:
        on_step("scrape", state)

    # ==========================================
    # STEP 3 - WRITER
    # ==========================================

    print("\n" + "=" * 50)
    print("Step 3 - Writer")
    print("=" * 50)
    research_combined = (
        f"SEARCH RESULTS:\n"
        f"{state['search_results']}\n\n"
        f"DETAILED SCRAPED CONTENT:\n"
        f"{state['scraped_content']}"
    )
    state["report"] = ""
    for chunk in writer_chain.stream({
        "topic": topic,
        "research": research_combined,
    }):
        state["report"] += chunk
        if on_stream:
            on_stream(state["report"])
    print(
        "\nFinal Report:\n", state["report"])

    if on_step:
        on_step("report", state)

    # ==========================================
    # STEP 4 - CRITIC
    # ==========================================

    print("\n" + "=" * 50)
    print("Step 4 - Critic")
    print("=" * 50)

    state["feedback"] = critic_chain.invoke({
        "report": state["report"]
    })

    print(
        "\nCritic Report:\n",

        state["feedback"]

    )

    if on_step:
        on_step("feedback", state)

    return state