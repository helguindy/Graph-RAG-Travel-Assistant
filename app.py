"""
03_STREAMLIT_APP.PY
=====================================================================
Milestone 3: Graph-RAG Travel Assistant - Interactive Demo

A Streamlit UI that demonstrates the complete Graph-RAG pipeline:
  1. Input Preprocessing: Intent classification, entity extraction, embedding
  2. Graph Retrieval: Baseline and embedding-based retrieval from Neo4j
  3. LLM Layer: (Coming soon) Response generation from retrieved context

Allows users to enter queries and see results at each stage.
=====================================================================
"""

import streamlit as st
import json
import sys
from typing import Dict, Any
import importlib.util
try:
    from pyvis.network import Network
    PYVIS_AVAILABLE = True
except ImportError:
    PYVIS_AVAILABLE = False

# Import the preprocessing module using importlib (module name can't start with digit)
spec = importlib.util.spec_from_file_location("input_preprocessing", "01_input_preprocessing.py")
input_preprocessing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(input_preprocessing)
InputPreprocessor = input_preprocessing.InputPreprocessor

# Import the graph retrieval module using importlib (filename starts with digit)
spec2 = importlib.util.spec_from_file_location("graph_retrieval", "02_graph_retrieval.py")
graph_retrieval = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(graph_retrieval)
BaselineRetriever = getattr(graph_retrieval, 'BaselineRetriever', None)
EmbeddingRetriever = getattr(graph_retrieval, 'EmbeddingRetriever', None)
HybridRetriever = getattr(graph_retrieval, 'HybridRetriever', None)

st.set_page_config(page_title="Graph-RAG Travel Assistant", layout="wide")

# =====================================================================
# GRAPH VISUALIZATION HELPER
# =====================================================================
def visualize_knowledge_graph(retrieval_result: Dict[str, Any], title: str = "Knowledge Graph") -> str:
    """
    Create an interactive network graph visualization from retrieval results.
    
    Args:
        retrieval_result: Dict with baseline_results, embedding_results from HybridRetriever
        title: Title for the graph
    
    Returns:
        HTML string of the network visualization (or fallback message if pyvis unavailable)
    """
    if not PYVIS_AVAILABLE:
        return "<div style='color:red;'>pyvis not installed. Run: pip install pyvis</div>"
    
    net = Network(directed=True, height="600px")
    net.physics(enabled=True)
    
    # Collect all hotels from results
    baseline_results = retrieval_result.get('baseline_results', [])
    embedding_results = retrieval_result.get('embedding_results', [])
    
    hotels = []
    for res in baseline_results:
        hotel_id = res.get('h.hotel_id') or res.get('hotel_id')
        hotel_name = res.get('h.name') or res.get('hotel_name')
        if hotel_id and hotel_name:
            hotels.append({
                'id': hotel_id,
                'name': hotel_name,
                'source': 'baseline',
                'score': res.get('h.average_reviews_score', 0),
            })
    
    for res in embedding_results:
        hotel_id = res.get('hotel_id')
        hotel_name = res.get('hotel_name')
        score = res.get('score', 0)
        if hotel_id and hotel_name:
            # Check if already added from baseline
            if not any(h['id'] == hotel_id for h in hotels):
                hotels.append({
                    'id': hotel_id,
                    'name': hotel_name,
                    'source': 'embedding',
                    'score': score,
                })
    
    # Add hotel nodes
    for hotel in hotels:
        color = '#FF6B6B' if hotel['source'] == 'baseline' else '#4ECDC4'
        title_text = f"{hotel['name']}\nScore: {hotel['score']:.2f}\nSource: {hotel['source']}"
        net.add_node(
            f"hotel_{hotel['id']}",
            label=hotel['name'],
            title=title_text,
            color=color,
            size=25,
            font={'size': 12},
        )
    
    # Add city/country nodes (extract from hotels if possible)
    cities = set()
    for res in baseline_results + embedding_results:
        city = res.get('h.city') or res.get('city') or res.get('city')
        if city:
            cities.add(city)
    
    for city in cities:
        net.add_node(
            f"city_{city}",
            label=city,
            color='#95E1D3',
            size=15,
            shape='diamond',
            font={'size': 10},
        )
        # Connect hotels to city
        for hotel in hotels:
            net.add_edge(f"hotel_{hotel['id']}", f"city_{city}", arrows='to', title="LOCATED_IN")
    
    # Add query context node
    net.add_node(
        "query",
        label="Your Query",
        color='#FFD93D',
        size=20,
        shape='box',
        font={'size': 11, 'bold': True},
    )
    
    # Connect query to hotels
    for hotel in hotels:
        net.add_edge("query", f"hotel_{hotel['id']}", title="Retrieves", color='#FFD93D', width=2)
    
    net.show_buttons(filter_=['physics'])
    net.toggle_physics(True)
    
    return net.show()

st.set_page_config(page_title="Graph-RAG Travel Assistant", layout="wide")

st.title("🏨 Graph-RAG Travel Assistant")
st.markdown("""
An end-to-end system that combines:
- **Symbolic Reasoning** (Knowledge Graph / Neo4j) for factual accuracy
- **Statistical Reasoning** (LLM) for natural language understanding
- **Retrieval Augmentation** for reduced hallucination and interpretability
""")

# Initialize preprocessor
@st.cache_resource
def load_preprocessor():
    return InputPreprocessor(csv_dir="csv", embedding_model="all-MiniLM-L6-v2")

preprocessor = load_preprocessor()

# Main input
st.sidebar.header("Query Input")
user_query = st.sidebar.text_area(
    "Enter your travel-related question:",
    value="Find hotels in Cairo with rating > 4",
    height=100
)

if st.sidebar.button("Process Query", key="process_btn"):
    st.session_state.process = True
else:
    st.session_state.process = st.session_state.get("process", False)

# Process the query if button is clicked
if user_query.strip() and st.session_state.process:
    result = preprocessor.process(user_query)

    # Create four tabs for each stage (including knowledge graph visualization)
    tab1, tab2, tab2b, tab3 = st.tabs([
        "1️⃣ Input Preprocessing",
        "2️⃣ Graph Retrieval",
        "📊 Knowledge Graph Viz",
        "3️⃣ LLM Response"
    ])

    # =====================================================================
    # TAB 1: INPUT PREPROCESSING
    # =====================================================================
    with tab1:
        st.header("Stage 1: Input Preprocessing")
        st.markdown("""
        **Goal:** Convert raw user input into structured intent, entities, and embeddings.

        **Components:**
        - **Intent Classification:** What does the user want? (search, recommend, book, etc.)
        - **Entity Extraction:** What entities are mentioned? (cities, hotels, traveller types, ratings, dates)
        - **Input Embedding:** Convert query to vector for semantic search
        """)

        col1, col2 = st.columns(2)

        # Intent Classification
        with col1:
            st.subheader("Intent Classification")
            intent = result.get('intent', 'N/A')
            confidence = result.get('intent_confidence', 'N/A')

            st.metric("Detected Intent", intent)
            st.metric("Confidence", confidence)

            with st.expander("View Intent Details"):
                st.json({
                    'intent': intent,
                    'confidence': confidence,
                    'theme': result.get('theme', 'hotel'),
                })

        # Entity Extraction
        with col2:
            st.subheader("Entity Extraction")
            entities = result.get('entities', {})
            extracted_count = sum(1 for v in entities.values() if v is not None)

            st.metric("Entities Found", extracted_count)

            with st.expander("View Extracted Entities"):
                entity_display = {k: v for k, v in entities.items() if v is not None}
                if entity_display:
                    st.json(entity_display)
                else:
                    st.info("No entities extracted from this query.")

        # Input Embedding
        st.subheader("Input Embedding")
        embedding = result.get('embedding')
        embedding_dim = result.get('embedding_dim')

        if embedding:
            col1, col2, col3 = st.columns(3)
            col1.metric("Embedding Dimension", embedding_dim)
            col2.metric("First 5 values", f"{embedding[:5]}")
            col3.metric("Vector norm", f"{sum(x**2 for x in embedding)**0.5:.4f}")

            with st.expander("View Full Embedding Vector"):
                st.write(embedding)
        else:
            st.warning("Embedding not available. Install sentence-transformers: `pip install sentence-transformers`")

        # Summary
        st.subheader("Preprocessing Summary")
        summary_data = {
            "Raw Query": result.get('raw_input'),
            "Detected Intent": result.get('intent'),
            "Intent Confidence": result.get('intent_confidence'),
            "Entities Extracted": {k: v for k, v in result.get('entities', {}).items() if v is not None},
            "Embedding Available": embedding is not None,
            "Embedding Dimensions": embedding_dim,
        }
        st.json(summary_data)

    # =====================================================================
    # TAB 2: GRAPH RETRIEVAL (Placeholder - Coming Soon)
    # =====================================================================
    with tab2:
        st.header("Stage 2: Graph Retrieval")
        st.markdown("""
        **Goal:** Retrieve relevant information from the Neo4j Knowledge Graph.

        **Methods:**
        - **Baseline Retrieval:** Execute deterministic Cypher queries (exact matches, filters)
        - **Embedding-Based Retrieval:** Semantic similarity search using query embeddings

        **Status:** Retrieval templates defined (12 Cypher queries), executor ready.
        """)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Baseline Retrieval")
            st.info("""
            Cypher Query Template Selected:
            ```cypher
            MATCH (h:Hotel)-[:LOCATED_IN]->(c:City {name:$city})
            WHERE h.average_reviews_score >= $min_rating
            RETURN h LIMIT 50
            ```

            Status: Waiting for Neo4j connection...
            (Configure Neo4j in config.txt)
            """)

        with col2:
            st.subheader("Embedding-Based Retrieval")
            st.info("""
            Query Embedding: ✓ Generated
            Dimensions: 384

            Status: Placeholder
            (Full implementation requires Neo4j vector index)
            """)

        # Retrieval options
        st.subheader("Retrieval Method Selection")
        retrieval_method = st.radio(
            "Choose retrieval method:",
            ["baseline", "embeddings", "hybrid"],
            index=0
        )

        if st.button("Execute Retrieval"):
            st.info(f"Executing {retrieval_method}...")

            # Build entities and embedding from preprocessing result
            entities = result.get('entities', {}) or {}
            query_embedding = result.get('embedding')

            # Use local CSV retriever for baseline (cloud Neo4j unreachable in this environment)
            baseline = None
            try:
                LocalBaseline = getattr(graph_retrieval, 'LocalBaselineRetriever', None)
                if LocalBaseline is not None:
                    baseline = LocalBaseline(csv_dir='csv')
                    st.info("✓ Using local CSV baseline retrieval (cloud Neo4j unreachable).")
            except Exception as e:
                st.warning(f"Local baseline init failed: {e}")
            else:
                st.info("Neo4j config incomplete. Baseline retrieval will be unavailable.")

            # Initialize embedding retriever (use preprocessor embedder if available)
            embedding = None
            try:
                embedder = getattr(preprocessor, 'embedder', None)
                embedding = EmbeddingRetriever(embedder=embedder)
            except Exception as e:
                st.warning(f"Embedding retriever init failed: {e}")

            hybrid = HybridRetriever(baseline_retriever=baseline, embedding_retriever=embedding)

            # Execute retrieval
            try:
                r = hybrid.retrieve(intent=result.get('intent', 'hotel_search'), entities=entities, query_embedding=query_embedding, method=retrieval_method)
            except Exception as e:
                st.error(f"Retrieval failed: {e}")
                r = None

            if r is None:
                st.info("No results returned.")
            else:
                # Store result in session state so graph viz tab can access it
                st.session_state['retrieval_result'] = r
                
                st.success(f"Retrieval completed. Baseline status: {r.get('baseline_status')}, Embedding status: {r.get('embedding_status')}")

                # Show baseline results
                if r.get('baseline_results'):
                    st.subheader("Baseline Results")
                    try:
                        st.dataframe(r.get('baseline_results'))
                    except Exception:
                        st.json(r.get('baseline_results'))
                else:
                    st.info("No baseline results.")

                # Show embedding results
                if r.get('embedding_results'):
                    st.subheader("Embedding Results")
                    st.dataframe(r.get('embedding_results'))
                else:
                    st.info("No embedding results.")

                # Show merged results for hybrid
                if retrieval_method == 'hybrid':
                    if r.get('merged_results'):
                        st.subheader("Merged Results")
                        st.dataframe(r.get('merged_results'))
                    else:
                        st.info("No merged results.")

                with st.expander('Full Retrieval Debug JSON'):
                    st.json(r)

    # =====================================================================
    # TAB 2b: KNOWLEDGE GRAPH VISUALIZATION
    # =====================================================================
    with tab2b:
        st.header("Knowledge Graph Visualization")
        st.markdown("""
        **Visualize the Knowledge Graph structure** showing:
        - **Hotels** (retrieved from your query) in red/teal
        - **Cities** (where hotels are located) in green diamond
        - **Connections** (LOCATED_IN relationships) showing graph structure
        """)
        
        if 'retrieval_result' not in st.session_state:
            st.info("👈 Run a retrieval query in the 'Graph Retrieval' tab first to see the knowledge graph visualization.")
        else:
            r = st.session_state['retrieval_result']
            
            if not PYVIS_AVAILABLE:
                st.error("⚠️ Graph visualization requires `pyvis`. Install it with:")
                st.code("pip install pyvis")
            else:
                if not r.get('baseline_results') and not r.get('embedding_results'):
                    st.warning("No results to visualize. Try a different query.")
                else:
                    st.success(f"Visualizing {len(r.get('baseline_results', [])) + len(r.get('embedding_results', []))} hotels")
                    
                    # Generate and display graph
                    html_graph = visualize_knowledge_graph(r, title="Hotel Knowledge Graph")
                    st.components.v1.html(html_graph, height=700)
                    
                    # Legend
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown("🔴 **Red nodes** = Baseline retrieval results")
                    with col2:
                        st.markdown("🔵 **Teal nodes** = Embedding retrieval results")
                    with col3:
                        st.markdown("🟢 **Green diamonds** = Cities")

    # =====================================================================
    # TAB 3: LLM RESPONSE (Placeholder - Coming Soon)
    # =====================================================================
    with tab3:
        st.header("Stage 3: LLM Response Generation")
        st.markdown("""
        **Goal:** Generate a natural language response grounded in KG context.

        **Components:**
        - **Context Merging:** Combine baseline and embedding results
        - **Prompt Engineering:** Structure context + persona + task
        - **LLM Selection:** Choose from multiple models (GPT, Claude, Gemini, Open-source)
        - **Response Generation:** Generate final answer

        **Status:** Coming soon (see 03_llm_layer.py)
        """)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("LLM Model Selection")
            model = st.selectbox(
                "Choose an LLM:",
                ["GPT-3.5-turbo (OpenAI)", "GPT-4 (OpenAI)", "Claude-3 (Anthropic)",
                 "Gemini Pro (Google)", "Llama-2 (Meta)", "Mistral-7B (Mistral)"]
            )
            st.info(f"Selected: {model}")

        with col2:
            st.subheader("Prompt Structure")
            st.markdown("""
            **Context:** Retrieved KG information
            **Persona:** "You are a helpful travel assistant"
            **Task:** "Answer using only provided information"
            """)

        st.subheader("Generated Response (Mock)")
        st.info("""
        Based on the retrieved information from the Knowledge Graph:

        **Cairo Hotels with Rating > 4.0:**
        1. **Nile Grandeur** - 5-star, Rating: 4.2
           Located in Cairo, Egypt. Known for great location and staff service.

        2. **The Azure Tower** - 5-star, Rating: 4.6
           Top-rated hotel in New York (not Cairo - different results may apply).

        Would you like more details about any of these hotels?
        """)

        st.subheader("LLM Comparison Metrics (Placeholder)")
        comparison_df = {
            "Model": ["GPT-3.5", "GPT-4", "Claude-3", "Llama-2"],
            "Response Time (ms)": [250, 800, 400, 150],
            "Accuracy": [0.78, 0.92, 0.88, 0.75],
            "Cost ($/1K tokens)": [0.0015, 0.03, 0.008, 0.0],
        }
        st.dataframe(comparison_df)

else:
    st.info("👈 Enter a query in the sidebar and click 'Process Query' to see the pipeline in action.")


