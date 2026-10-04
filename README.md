# 🍽️ AI-Powered Multimodal Restaurant Recommendation System

### An end-to-end AI application combining LLMs, multimodal RAG, vector search, multi-agent reasoning, and MCP.

An end-to-end AI restaurant recommendation system that combines **LLM-powered data structuring, multimodal processing, vector search, metadata filtering, multi-agent reasoning, and the Model Context Protocol (MCP)**.

The project starts with messy, heterogeneous restaurant data and progressively transforms it into a searchable multimodal knowledge base, exposes that knowledge through MCP tools, and connects those tools to an LLM-powered application capable of understanding natural-language questions and generating personalized restaurant recommendations.

---

## 🚀 Project Overview

Modern recommendation systems often have access to large amounts of data, but that data may exist across multiple formats and modalities: unstructured text, reviews, recipes, images, URLs, and user histories.

This project demonstrates how to build an AI system that turns that heterogeneous information into an **intelligent, queryable knowledge base**.

The system follows an end-to-end pipeline:

```text
Raw Multimodal Data
        ↓
LLM-Powered Data Structuring
        ↓
Multimodal Data Enrichment
        ↓
Structured Knowledge Base
        ↓
Data Management Interface
        ↓
Multimodal Vector Index
        ↓
Similarity Search + Metadata Filtering
        ↓
Multi-Agent Recommendation Framework
        ↓
MCP Server
        ↓
MCP Client
        ↓
LLM-Powered MCP Host
        ↓
Natural-Language Restaurant Recommendations
```

The result is a complete AI application in which an LLM can interpret a user's request, determine which tools are relevant, retrieve restaurant information, and synthesize the retrieved information into a useful response.

---

# 🧠 What I Built

The project is organized into seven major stages, with each stage adding another layer to the final AI system.

## 1. Structure Unstructured Data with LLMs

**Directory:** `01_Structure-Unstructured-Data-with-LLM`

The project begins with a heterogeneous dataset containing:

* Restaurant descriptions
* User reviews
* Recipes
* Recipe metadata
* Food images
* User restaurant visit histories
* URLs

The raw information is not immediately suitable for efficient search or retrieval because it lacks a consistent schema and spans multiple modalities.

LLMs are used to transform this information into a **structured, machine-readable knowledge base**.

### Key capabilities

* Extract semantic information from unstructured restaurant descriptions and reviews
* Convert heterogeneous records into consistent JSON structures
* Organize restaurant information into a query-friendly schema
* Prepare text, image, and URL information for downstream AI processing

This establishes the structured data layer used throughout the remainder of the project.

---

## 2. Process Multimodal Data with LLMs

**Directory:** `02_Process-Multimodal-Data-with-LLMs`

Text alone does not capture all of the information contained in a restaurant dataset.

The second stage enriches the structured data with information derived from food images.

### Key capabilities

* Generate image captions for food images
* Use multimodal models to identify visual information
* Associate image-derived information with the corresponding restaurant and recipe records
* Merge visual and textual information into a unified schema

This creates a **multimodal knowledge base** in which images can contribute searchable semantic information alongside traditional text.

---

## 3. Command-Line Data Management UI

**Directory:** `03_Command-Line Data Management UI`

Once the data has been structured and enriched, manually managing raw JSON files becomes cumbersome and error-prone.

I built a Python command-line data management application that acts as an interface to the structured restaurant knowledge base.

### Key capabilities

* Create, read, update, and manage restaurant records
* Read and write structured restaurant data
* Perform data modifications through a controlled interface
* Integrate generative AI functions for automated data processing and enrichment
* Implement safeguards against accidental data loss

This creates a practical data-management layer between the underlying files and the AI pipeline.

---

## 4. Multimodal Vector Index

**Directory:** `04_Multimodal_Vector_Index`

The structured restaurant data is transformed into retrieval-ready documents and indexed for semantic search.

### Key capabilities

* Construct retrieval documents from structured restaurant data
* Generate dense embeddings for text
* Generate embeddings for food images
* Create separate vector indexes for different modalities
* Persist vector databases using **Chroma**
* Prepare the retrieval layer for downstream RAG applications

The result is the foundation of a **multimodal Retrieval-Augmented Generation (RAG) pipeline**.

---

## 5. Similarity Retrieval + Metadata Filtering

**Directory:** `05_Similarity-Retrieval-with-Metadata-Filtering`

The vector indexes are then used to perform semantic retrieval while maintaining control over hard constraints.

### Key capabilities

* Top-k similarity retrieval
* Metadata-based filtering with Chroma
* Hybrid similarity + metadata retrieval
* Image-to-image similarity search using **CLIP embeddings**
* Cosine similarity scoring
* Similarity score normalization across embedding spaces
* Weighted fusion of text and image retrieval results
* Constraint-aware reranking

This allows the system to combine different sources of evidence rather than relying exclusively on text similarity.

For example, a recommendation can incorporate both:

```text
"What restaurants serve spicy food?"
              +
"What food images are visually similar?"
              +
"Does the restaurant satisfy the required metadata constraints?"
```

The result is a more controllable and explainable multimodal retrieval pipeline.

---

# 🤖 6. Multi-Agent Recommendation Framework

**Directory:** `06_Multi-agent-framework`

The project then moves beyond retrieval into **multi-agent reasoning**.

I designed six specialized agents, each responsible for a specific component of the recommendation process:

| Agent                         | Responsibility                                                   |
| ----------------------------- | ---------------------------------------------------------------- |
| 👤 **User Profile Generator** | Extracts user preferences and relevant profile information       |
| 🔎 **RAG Retriever**          | Retrieves relevant restaurant and food information               |
| 📈 **Food Trend Analyst**     | Identifies relevant food trends                                  |
| 🍜 **Food Style Expert**      | Analyzes cuisines, flavors, and food styles                      |
| 🥗 **Nutrition Expert**       | Evaluates nutritional content and dietary compatibility          |
| ⭐ **Recommendation Expert**   | Synthesizes the available information into final recommendations |

Each agent has a defined role, goal, background, and expected output.

This architecture separates individual reasoning tasks into specialized components before combining their outputs into a final recommendation.

---

# 🔌 7. Model Context Protocol (MCP)

**Directory:** `07_MCP-Server-Setup`

The final stage transforms the restaurant data and recommendation capabilities into an **MCP-based application**.

I built the MCP infrastructure from the ground up, including:

### MCP Server

A **FastMCP server** exposes restaurant information and functionality as discoverable tools.

The server provides tools for:

* Exact restaurant lookup
* Vibe-based restaurant recommendations
* Restaurant review retrieval

The server also exposes restaurant data as an MCP resource.

### MCP Client

I then built a client capable of:

* Connecting to the MCP server over `stdio`
* Establishing an MCP `ClientSession`
* Declaring permitted filesystem roots
* Handling delegated LLM sampling requests
* Discovering and calling MCP tools

### MCP Host

Finally, I built an LLM-powered host application with a **Gradio chat interface**.

The host:

1. Connects to the MCP server
2. Dynamically discovers available tools
3. Converts MCP tool schemas into LLM-compatible tool definitions
4. Provides those tools to the language model
5. Runs a ReAct-style agent loop
6. Allows the model to decide when tools should be called
7. Retrieves relevant restaurant information
8. Generates a final natural-language response

This completes the end-to-end architecture:

```text
                    ┌─────────────────────┐
                    │    User Question    │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │   LLM / AI Host     │
                    │     ReAct Loop      │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │    MCP Client       │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │     MCP Server      │
                    │                     │
                    │  Restaurant Tools   │
                    └──────────┬──────────┘
                               ↓
              ┌────────────────────────────────┐
              │ Multimodal Knowledge Base      │
              │                                │
              │ Structured Data                │
              │ Vector Databases               │
              │ Text Embeddings                │
              │ Image Embeddings               │
              │ Metadata                       │
              └────────────────────────────────┘
```

---

# 🛠️ Technologies & Concepts

This project combines several modern AI engineering technologies and architectural patterns.

### Artificial Intelligence

* Large Language Models (LLMs)
* Generative AI
* Multimodal LLMs
* Agentic AI
* ReAct agent loops

### Retrieval & Search

* Retrieval-Augmented Generation (RAG)
* Dense vector embeddings
* Semantic similarity search
* Metadata filtering
* Multimodal retrieval
* Hybrid retrieval
* Weighted multimodal fusion
* CLIP embeddings
* Chroma vector databases

### AI Agents

* Multi-agent architectures
* Specialized AI agents
* Agent orchestration
* Tool calling
* Agent-based recommendation systems

### Model Context Protocol

* FastMCP
* MCP resources
* MCP tools
* MCP clients
* MCP hosts
* Dynamic tool discovery
* LLM tool integration

### Application Development

* Python
* JSON
* Command-line interfaces
* Gradio
* API integration
* Structured data management

---

# 🏗️ Repository Structure

```text
Restaurant-Recommendation-System/
│
├── 01_Structure-Unstructured-Data-with-LLM/
│   └── LLM-powered data structuring
│
├── 02_Process-Multimodal-Data-with-LLMs/
│   └── Image captioning and multimodal enrichment
│
├── 03_Command-Line Data Management UI/
│   └── CLI-based restaurant data management
│
├── 04_Multimodal_Vector_Index/
│   └── Text and image embedding + vector indexing
│
├── 05_Similarity-Retrieval-with-Metadata-Filtering/
│   └── Multimodal retrieval and metadata filtering
│
├── 06_Multi-agent-framework/
│   └── Specialized recommendation agents
│
├── 07_MCP-Server-Setup/
│   └── MCP server, client, and host application
│
└── README.md
```

---

# 🎯 What This Project Demonstrates

This project was designed to demonstrate the complete lifecycle of building a modern AI application rather than focusing on a single machine-learning model.

It demonstrates how to:

* Work with messy, real-world multimodal data
* Use LLMs for structured data extraction
* Enrich datasets with multimodal information
* Build maintainable data-management tooling
* Generate and store multimodal embeddings
* Build vector search infrastructure
* Combine semantic retrieval with deterministic metadata constraints
* Fuse information from multiple modalities
* Design specialized AI agents
* Build tool-using AI systems
* Implement MCP servers and clients
* Connect MCP tools to an LLM
* Build an interactive AI application around the resulting architecture

---

# 🔄 End-to-End Architecture

At a high level, the complete system can be viewed as five layers:

### 1. Data Layer

Raw restaurant descriptions, reviews, recipes, images, URLs, and user histories.

↓

### 2. Knowledge Layer

LLM-structured JSON + multimodal enrichment + managed restaurant records.

↓

### 3. Retrieval Layer

Text embeddings + image embeddings + Chroma + CLIP + metadata filtering.

↓

### 4. Reasoning Layer

Specialized agents analyze user preferences, retrieved information, food styles, trends, nutrition, and recommendations.

↓

### 5. Application Layer

MCP exposes the underlying capabilities as tools that an LLM can dynamically discover and use through a conversational interface.

---

# 💡 Why MCP?

A major goal of this project was to explore how AI applications can move beyond simply prompting an LLM.

Instead of giving the model a static collection of information, the system gives the model access to **external capabilities through MCP tools**.

The LLM can therefore:

```text
Understand the user's request
        ↓
Determine what information it needs
        ↓
Discover available tools
        ↓
Select appropriate tools
        ↓
Retrieve external information
        ↓
Reason over the results
        ↓
Generate a final response
```

This architecture provides a foundation for building AI systems that can interact with real data sources and external tools rather than operating exclusively within the model's context window.

---

# 📌 Key Takeaway

This repository demonstrates an end-to-end approach to building an AI-powered recommendation application—from **raw multimodal data to an LLM-driven, tool-using application**.

Rather than implementing a single recommendation algorithm, the project explores the broader AI engineering stack required to build a production-style intelligent application:

**Data → LLM Processing → Multimodal Knowledge → Vector Retrieval → Agents → MCP → LLM Application**

---

## 👨‍💻 Author

**Cole Goodchild**

Data Scientist & AI/ML Engineer

[GitHub](https://github.com/ColeGoodchild)
