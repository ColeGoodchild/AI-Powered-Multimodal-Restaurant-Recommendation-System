Built a complete FastMCP server that exposes California restaurant data over the Model Context Protocol. This can now effectively configure data and tools in an MCP server!

## What This Step Accomplished

# MCP Server Setup
1. Defined an MCP resource to serve raw text data at a named URL
2. Written three MCP tools covering exact name lookup, vibe-based search, and review retrieval
3. Run the server locally and tested it with CLI commands

This server is the data layer that the upcoming labs will connect to — an agent can now discover these tools automatically and call them to answer questions about California restaurants.

## MCP Client Setup
Built a complete MCP client that connects to the Connoisseur server above, declares filesystem roots, handles delegated LLM sampling requests, and calls all three server tools.

1. Connected to an MCP server over stdio using ClientSession
2. Registered a roots callback to declare permitted filesystem paths
3. Implemented a sampling callback that proxies LLM calls through the Anthropic API
4. Called get_restaurant_info, recommend_by_vibe, and get_review via the MCP protocol

Together, the server and client form a complete MCP application. I'll build on this basic client-server app in order to enhance the demo functions for intelligent tool selection and create a full-fledged MCP application!

## MCP Host Setup

Built a complete MCP host application—a Gradio chat interface backed by a WatsonX LLM that discovers and calls MCP tools through a ReAct agent loop.

1. Connected to an MCP server at runtime and discovered its tools dynamically
2. Converted MCP tool schemas into LLM-compatible tool definitions and bound them to the model
3. Implemented a ReAct loop that iterates tool calls until the LLM produces a final answer
4. Built a Gradio chat UI with a streaming placeholder and quick-start prompt buttons

Together with the server built in the first lab, this application forms a complete end-to-end MCP system: a data layer exposed over the Model Context Protocol, and an intelligent host that uses it to answer natural language questions.
