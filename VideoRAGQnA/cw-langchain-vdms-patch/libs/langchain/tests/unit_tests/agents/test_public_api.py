from langchain.agents import __all__ as agents_all

_EXPECTED = [
    "Agent",
    "AgentExecutor",
    "AgentExecutorIterator",
    "AgentOutputParser",
    "AgentType",
    "BaseMultiActionAgent",
    "BaseSingleActionAgent",
    "ConversationalAgent",
    "ConversationalChatAgent",
    "LLMSingleActionAgent",
    "MRKLChain",
    "OpenAIFunctionsAgent",
    "OpenAIMultiFunctionsAgent",
    "ReActChain",
    "ReActTextWorldAgent",
    "SelfAskWithSearchChain",
    "StructuredChatAgent",
    "Tool",
    "XMLAgent",
    "ZeroShotAgent",
    "create_json_agent",
    "create_openapi_agent",
    "create_pbi_agent",
    "create_pbi_chat_agent",
    "create_spark_sql_agent",
    "create_sql_agent",
    "create_vectorstore_agent",
    "create_vectorstore_router_agent",
    "get_all_tool_names",
    "initialize_agent",
    "load_agent",
    "load_huggingface_tool",
    "load_tools",
    "tool",
    "create_openai_functions_agent",
    "create_xml_agent",
    "create_react_agent",
    "create_openai_tools_agent",
    "create_self_ask_with_search_agent",
    "create_json_chat_agent",
    "create_structured_chat_agent",
    "create_tool_calling_agent",
]


def test_public_api() -> None:
    """Test for regressions or changes in the agents public API."""
    assert sorted(agents_all) == sorted(_EXPECTED)