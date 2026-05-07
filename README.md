# Hybrid RAG

Hybrid Retrieval-Augmented Generation (Hybrid RAG) system combining semantic vector search with knowledge graph retrieval for more accurate and context-aware question answering.

This project integrates:
- ChromaDB for vector similarity search
- NetworkX for knowledge graph traversal
- Gemini 2.5 Flash for routing and answer generation
- Query routing logic for intelligent retrieval selection

The system dynamically decides whether a query should use:
- Vector retrieval
- Graph retrieval
- Hybrid retrieval combining both approaches

Built to address limitations of standard RAG pipelines where semantic retrieval alone fails to capture relationships between entities. :contentReference[oaicite:0]{index=0}

---

## Architecture

```text
User Query
    │
    ▼
┌────────────────────┐
│ Query Router (LLM) │
└─────────┬──────────┘
          │
   ┌──────┼──────┐
   │      │      │
   ▼      ▼      ▼
Vector  Graph  Hybrid
Search  Search  Merge
   │      │
   ▼      ▼
ChromaDB  NetworkX
(Vector)  (Knowledge Graph)
   └──────┬──────┘
          ▼
   Context Assembly
          ▼
   Gemini 2.5 Flash
          ▼
        Answer
```

The architecture routes queries based on intent:
- Factual lookups → Vector retrieval
- Relationship reasoning → Graph traversal
- Complex contextual queries → Hybrid retrieval

The routing workflow and retrieval architecture are described in the project document. :contentReference[oaicite:1]{index=1}

---

## Key Features

### Hybrid Retrieval Pipeline
- Combines semantic retrieval with graph reasoning
- Handles both factual and relational queries
- Improves contextual grounding

### Query Router
- LLM-powered classification layer
- Dynamically selects:
  - Vector mode
  - Graph mode
  - Hybrid mode

### Vector Search
- ChromaDB semantic retrieval
- SentenceTransformer embeddings
- Efficient top-k document chunk retrieval

### Knowledge Graph Retrieval
- Entity and relationship extraction
- Directed graph traversal using NetworkX
- Multi-entity relationship reasoning

### LLM-Powered Answer Generation
- Gemini 2.5 Flash based response synthesis
- Grounded generation using retrieved context
- Reduced hallucination through constrained prompts

### Structured Graph Construction
- Automatic entity extraction
- Relationship mapping
- Directed edge creation from unstructured text

---

## Tech Stack

| Category | Technologies |
|---|---|
| LLM | Gemini 2.5 Flash |
| Vector Database | ChromaDB |
| Knowledge Graph | NetworkX |
| Embeddings | SentenceTransformers |
| Framework | Python |
| Environment | uv |
| Console UI | Rich |
| Configuration | python-dotenv |

---

## Project Structure

```bash
Hybrid-RAG/
│
├── main.py
├── .env
├── pyproject.toml
├── README.md
│
├── data/
├── docs/
└── notebooks/
```

---

## How Hybrid RAG Works

### 1. Vector Retrieval
Semantic search retrieves document chunks similar to the query.

Best for:
- Definitions
- Pricing questions
- Direct factual lookups

Example:
```text
"What is the Premium Plan cost?"
```

### 2. Graph Retrieval
Knowledge graph traversal retrieves connected entities and relationships.

Best for:
- Relational reasoning
- Multi-hop questions
- Policy relationships

Example:
```text
"How does Premium Plan relate to support?"
```

### 3. Hybrid Retrieval
Combines vector chunks and graph relationships into a unified context.

Best for:
- Complex reasoning
- Cross-document synthesis
- Multi-context answers

Example:
```text
"Explain Premium Plan benefits including support and refund policies."
```

The document explains how the router classifies these retrieval modes dynamically at runtime. :contentReference[oaicite:2]{index=2}

---

## Query Router Logic

The query router acts as the decision-making layer.

### Routing Modes

| Mode | Use Case |
|---|---|
| Vector | Direct factual retrieval |
| Graph | Relationship reasoning |
| Hybrid | Combined contextual reasoning |

The router uses Gemini with a lightweight classification prompt to determine the optimal retrieval path before executing expensive retrieval operations. :contentReference[oaicite:3]{index=3}

---

## Knowledge Graph Pipeline

### Entity Extraction
Gemini extracts:
- Entities
- Relationships
- Targets

Example:
```json
{
  "entity": "Premium Plan",
  "relation": "includes",
  "target": "Priority Support"
}
```

### Graph Construction
NetworkX stores:
- Nodes → Entities
- Edges → Relationships

### Traversal
The graph retriever:
- Identifies query entities
- Traverses incoming and outgoing edges
- Builds relationship-aware context

---

## Installation

### Clone Repository

```bash
git clone git@github.com:Sudhanshud98/Hybrid-RAG.git
cd Hybrid-RAG
```

### Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

Or using uv:

```bash
uv sync
```

---

## Environment Variables

Create a `.env` file:

```env
GEMINI_API_KEY=your_api_key
```

Get your Gemini API key from:
https://aistudio.google.com

---

## Running the Project

```bash
uv run main.py
```

---

## Example Queries

### Vector Query

```text
What is the Premium Plan cost?
```

### Graph Query

```text
How does the Premium Plan relate to support?
```

### Hybrid Query

```text
Give me a complete overview of Premium Plan benefits.
```

---

## Example Output

```json
{
  "query": "How does Premium Plan relate to support?",
  "mode": "graph",
  "answer": "Premium Plan includes priority support with a guaranteed 2-hour response time."
}
```

---

## Production Enhancements

### Persistent Vector Storage
- ChromaDB PersistentClient
- Durable vector indexing

### PDF Ingestion
- PDF parsing support
- Page-level chunk ingestion

### Production Graph Database
Replace NetworkX with:
- Neo4j
- FalkorDB

The uploaded project document also outlines additional production-ready improvements and scaling considerations. :contentReference[oaicite:4]{index=4}

---

## Use Cases

- Enterprise document QA
- Policy analysis systems
- Customer support copilots
- Legal and compliance search
- Multi-document reasoning
- Relationship-aware RAG systems

---

## Contributors

Sudhanshu Deshpande

---
