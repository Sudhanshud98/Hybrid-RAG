import os
import json
import networkx as nx
import chromadb
from chromadb.utils import embedding_functions
from google import genai
from google.genai import types
from rich.console import Console
from rich.panel import Panel

console = Console()

# Load API key
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not set in environment")

# Gemini client
client = genai.Client(api_key=GEMINI_API_KEY)
MODEL = "gemini-2.5-flash"

# Step 1 — Initialise the ChromaDB vector store
def build_vector_store():
    """
    Creates an in-memory ChromaDB collection with a SentenceTransformer 
    embedding function.
    """
    chroma_client = chromadb.Client()
    
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    collection = chroma_client.create_collection(
        name="hybrid_rag_docs",
        embedding_function=ef,
    )
    return collection

# Step 2 — Initialise the knowledge graph
def build_knowledge_graph():
    """
    Returns an empty directed graph.
    Nodes = entities (concepts, products, policies, people).
    Edges = labelled relationships between them.
    """
    graph = nx.DiGraph()
    return graph

# Step 3 — Ingest a document into both systems
def extract_entities_and_relations(text: str) -> list[dict]:
    """
    Sends a document chunk to Gemini and asks it to extract
    (entity, relation, target) triples as JSON.
    These triples become directed edges in the knowledge graph.
    """
    prompt = f"""Extract the key entities and relationships from the text below.
    Return ONLY a JSON array. No explanation, no markdown, no code fences.
    Each item must have exactly these keys:
        "entity" — the subject (e.g. "Premium Plan")
        "relation" — the relationship (e.g. "includes")
        "target" — the object (e.g. "Priority Support")
    Text:
    {text}

    JSON array:"""
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.0),
    )
    raw = response.text.strip()

    # Strip markdown fences if the model added them despite instructions
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    try:
        triples = json.loads(raw)
        return triples if isinstance(triples, list) else []
    except json.JSONDecodeError:
        return []

def ingest_documents(documents: list[dict], collection, graph: nx.DiGraph):
    """
    Ingests a list of documents into both ChromaDB and the knowledge graph.
    Each document is a dict with:
        "id" — unique string identifier
        "text" — the document content
    ChromaDB receives the raw text as a chunk.
    The knowledge graph receives extracted (entity, relation, target) triples.
    """
    console.print("[bold blue]Ingesting documents...[/bold blue]")
    for doc in documents:
        doc_id = doc["id"]
        text = doc["text"]

        # 1. Add to ChromaDB for vector retrieval
        collection.add(
            documents=[text],
            ids=[doc_id],
        )

        # 2. Extract entities and add to knowledge graph
        triples = extract_entities_and_relations(text)
        for triple in triples:
            entity = triple.get("entity", "").strip()
            relation = triple.get("relation", "").strip()
            target = triple.get("target", "").strip()

            if entity and relation and target:
                # Add nodes if they don't exist
                if not graph.has_node(entity):
                    graph.add_node(entity)
                if not graph.has_node(target):
                    graph.add_node(target)

                # Add directed edge with relationship label
                graph.add_edge(entity, target, relation=relation, source_doc=doc_id)
            
            console.print(
                f" [green]✓[/green] {doc_id} — "
                f"{len(triples)} relations extracted"
            )
        console.print(
            f"\n[bold green]Ingestion complete.[/bold green] "
            f"Graph: {graph.number_of_nodes()} nodes, "
            f"{graph.number_of_edges()} edges."
        )

# Step 4 — Vector retriever
def vector_retrieve(query: str, collection, top_k: int = 3) -> list[str]:
    """
    Queries ChromaDB for the top_k most semantically similar
    document chunks.
    Returns a list of text strings ready to be injected into
    a prompt.
    """
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
    )

    # results["documents"] is a list of lists — one per query
    chunks = results["documents"][0] if results["documents"] else []
    return chunks

# Step 5 — Graph retriever
def identify_query_entities(query: str) -> list[str]:
    """
    Asks Gemini to identify which entities from the query
    should be looked up in the knowledge graph.
    Returns a list of entity name strings.
    """
    prompt = f"""Identify the key entities (nouns, concepts,
    products, policies) in this query.
    Return ONLY a JSON array of strings. No explanation, no markdown.
    Query: {query}
    JSON array:"""
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.0),
    )
    raw = response.text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    try:
        entities = json.loads(raw)
        return [str(e) for e in entities] if isinstance(entities, list) else []
    except json.JSONDecodeError:
        return []
    
def graph_retrieve(query: str, graph: nx.DiGraph) -> list[str]:
    """
    Identifies entities in the query, finds them in the grap
    h,
    then collects all their direct neighbours and edge relati
    onships.
    Returns a list of plain-English relationship strings:
    "PremiumPlan --[includes]--> PrioritySupport"
    """
    query_entities = identify_query_entities(query)
    context_lines = []
    for entity in query_entities:
        # Case-insensitive node matching
        matched_nodes = [
            n for n in graph.nodes()
            if entity.lower() in n.lower() or n.lower() in entity.lower()
        ]
        for node in matched_nodes:
            # Outgoing edges — what this entity connects to
            for _, target, data in graph.out_edges(node, data = True):
                relation = data.get("relation", "relates to")
                context_lines.append(f"{node} --[{relation}]--> {target}")
            # Incoming edges — what connects to this entity
            for source, _, data in graph.in_edges(node, data = True):
                relation = data.get("relation", "relates to")
                context_lines.append(f"{source} --[{relation}]--> {node}")
    return context_lines

# Step 6 — Query router
def route_query(query: str) -> str:
    """
    Classifies the incoming query and returns the retrieval mode to use.
    Returns one of:
        "vector" — direct factual lookup, best answered by semantic similarity
        "graph" — relational or multi-hop question, best answered by graph traversal
        "hybrid" — complex question needing both types of context
    The routing decision is based on question structure, not topic.
    """
    prompt = f"""You are a query router for a hybrid RAG system.
    Classify the query below into exactly one retrieval mode.
    Modes:
    vector — Direct factual question. Asks for a specific fact, value, or definition.
    Examples: "What is the return window?", "What does the Premium plan cost?"
    graph — Relational question. Asks how things connect, compare, or affect each other.
    Examples: "How does the pricing tier affect support options?",
    "What is the relationship between refund policy and subscription type?"
    hybrid — Complex question needing both specific facts ANDrelational context.
    Examples: "Explain how the cancellation policy works and what plans it applies to.",
    "Give me a full overview of what Premium subscribers get."
    Return ONLY one word: vector, graph, or hybrid. Nothing else.
    Query: {query}
    Mode:"""
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.0),
    )
    decision = response.text.strip().lower()

    # Validate and fallback to hybrid if the model returns something unexpected
    if decision not in ("vector", "graph", "hybrid"):
        decision = "hybrid"
        
    return decision