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
    api_key: str,
) -> dict:

    state = {}

    # ==========================================
    # CREATE LLM
    # ==========================================

    llm = get_llm(
        model_name=model_name,
        api_key=api_key,
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

    state["report"] = writer_chain.invoke({

        "topic": topic,

        "research": research_combined,

    })

    print(

        "\nFinal Report:\n",

        state["report"]

    )

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

    return state