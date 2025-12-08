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
import tempfile
import os
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

# Import the LLM layer module using importlib (filename starts with digit)
spec3 = importlib.util.spec_from_file_location("llm_layer", "03_llm_layer.py")
llm_layer = importlib.util.module_from_spec(spec3)
spec3.loader.exec_module(llm_layer)
LLMOrchestrator = getattr(llm_layer, 'LLMOrchestrator', None)
HuggingFaceProvider = getattr(llm_layer, 'HuggingFaceProvider', None)
OpenAIProvider = getattr(llm_layer, 'OpenAIProvider', None)
AnthropicProvider = getattr(llm_layer, 'AnthropicProvider', None)
ModelEvaluator = getattr(llm_layer, 'ModelEvaluator', None)

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
    
    # Collect all hotels from results
    baseline_results = retrieval_result.get('baseline_results', [])
    embedding_results = retrieval_result.get('embedding_results', [])
    intent = retrieval_result.get('intent', '')
    
    # Check if this is a visa query (different visualization)
    is_visa_query = intent == 'visa_check' or any('visa_type' in str(res) or 'from.name' in str(res) for res in baseline_results)
    
    if is_visa_query:
        # Handle visa query visualization
        countries_seen = set()
        visa_relationships = []
        
        for res in baseline_results:
            from_country = res.get('from.name') or res.get('from_name') or res.get('from_country')
            to_country = res.get('to.name') or res.get('to_name') or res.get('to_country')
            visa_type = res.get('v.visa_type') or res.get('visa_type')
            visa_status = res.get('visa_status', 'Unknown')
            
            if from_country:
                countries_seen.add(from_country)
            if to_country:
                countries_seen.add(to_country)
            if from_country and to_country:
                visa_relationships.append({
                    'from': from_country,
                    'to': to_country,
                    'visa_type': visa_type or visa_status or 'Unknown',
                    'visa_status': visa_status
                })
        
        # Add country nodes
        for country in countries_seen:
            net.add_node(
                f"country_{country}",
                label=country,
                color='#FF6B6B',
                size=30,
                shape='ellipse',
                font={'size': 14, 'bold': True},
            )
        
        # Add visa relationship edges
        for rel in visa_relationships:
            label = rel.get('visa_type', rel.get('visa_status', 'Unknown'))
            title_text = f"Visa Status: {rel.get('visa_status', 'Unknown')}"
            if rel.get('visa_type'):
                title_text += f"\nVisa Type: {rel['visa_type']}"
            # Color based on visa status
            edge_color = '#FF6B6B' if 'required' in str(rel.get('visa_status', '')).lower() else '#4ECDC4'
            net.add_edge(
                f"country_{rel['from']}",
                f"country_{rel['to']}",
                label=label,
                title=title_text,
                color=edge_color,
                width=3,
                arrows='to'
            )
        
        # Add query node
        net.add_node(
            "query",
            label="Your Query",
            color='#FFD93D',
            size=20,
            shape='box',
            font={'size': 11, 'bold': True},
        )
        
        # Connect query to countries
        for country in countries_seen:
            net.add_edge("query", f"country_{country}", title="Queries", color='#95E1D3', width=2)
    
    else:
        # Original hotel visualization
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
    
    # Generate HTML - use temporary file approach for better compatibility
    try:
        # Try generate_html() first (newer pyvis versions)
        html = net.generate_html()
        return html
    except (AttributeError, TypeError):
        # Fallback: write to temporary file and read it back
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        try:
            net.show(tmp_path)
            with open(tmp_path, 'r', encoding='utf-8') as f:
                html = f.read()
            os.unlink(tmp_path)  # Clean up temp file
            return html
        except Exception as e:
            # If that fails too, return error message
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            return f"<div style='color:red;'>Error generating graph: {e}</div>"

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
            # Count entities - lists count as multiple values
            extracted_count = 0
            for v in entities.values():
                if v is not None:
                    if isinstance(v, list):
                        extracted_count += len(v)
                    else:
                        extracted_count += 1

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
            Query Embedding: [OK] Generated
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
            # Add rating_types from intent classification to entities
            if result.get('rating_types'):
                entities['rating_types'] = result.get('rating_types')
            query_embedding = result.get('embedding')

            # Try to connect to Neo4j first, fall back to local CSV if unavailable
            baseline = None
            load_neo4j_config = getattr(graph_retrieval, 'load_neo4j_config', None)
            BaselineRetriever = getattr(graph_retrieval, 'BaselineRetriever', None)
            LocalBaseline = getattr(graph_retrieval, 'LocalBaselineRetriever', None)
            
            if load_neo4j_config and BaselineRetriever:
                config = load_neo4j_config('config.txt')
                if config.get('uri') and config.get('username') and config.get('password'):
                    try:
                        baseline = BaselineRetriever(config['uri'], config['username'], config['password'])
                        if baseline.connected:
                            st.success("[OK] Connected to Neo4j")
                        else:
                            raise Exception("Neo4j connection failed")
                    except Exception as e:
                        st.warning(f"Neo4j connection failed: {e}. Falling back to local CSV retrieval.")
                        baseline = None
            
            # Fall back to local CSV retriever if Neo4j unavailable
            if baseline is None and LocalBaseline is not None:
                try:
                    baseline = LocalBaseline(csv_dir='csv')
                    st.info("[OK] Using local CSV baseline retrieval")
                except Exception as e:
                    st.warning(f"Local baseline init failed: {e}")

            # Initialize embedding retriever (use preprocessor embedder if available)
            embedding = None
            try:
                embedder = getattr(preprocessor, 'embedder', None)
                if embedder and getattr(embedder, 'available', False):
                    embedding = EmbeddingRetriever(embedder=embedder, csv_dir='csv')
                    st.success("[OK] Embedding retriever initialized")
                else:
                    st.warning("Embedder not available. Embedding retrieval will be disabled.")
            except Exception as e:
                st.warning(f"Embedding retriever init failed: {e}")

            hybrid = HybridRetriever(baseline_retriever=baseline, embedding_retriever=embedding)

            # Execute retrieval
            try:
                r = hybrid.retrieve(intent=result.get('intent', 'hotel_search'), entities=entities, query_embedding=query_embedding, method=retrieval_method)
            except Exception as e:
                st.error(f"Retrieval failed: {e}")
                import traceback
                st.code(traceback.format_exc())
                r = None

            if r is None:
                st.info("No results returned.")
            else:
                # Store result in session state so graph viz tab and LLM tab can access it
                st.session_state['retrieval_result'] = r
                st.session_state['user_query'] = user_query
                
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
    # TAB 3: LLM RESPONSE GENERATION
    # =====================================================================
    with tab3:
        st.header("Stage 3: LLM Response Generation")
        st.markdown("""
        **Goal:** Generate a natural language response grounded in KG context.

        **Components:**
        - **Context Merging:** Combine baseline and embedding results
        - **Prompt Engineering:** Structure context + persona + task
        - **LLM Selection:** Choose from multiple models (GPT, Claude, HuggingFace)
        - **Response Generation:** Generate final answer
        """)
        
        if 'retrieval_result' not in st.session_state:
            st.info("👈 Run a retrieval query in the 'Graph Retrieval' tab first to generate LLM responses.")
        else:
            r = st.session_state['retrieval_result']
            stored_query = st.session_state.get('user_query', user_query)
            
            # Initialize LLM orchestrator
            @st.cache_resource
            def load_llm_orchestrator():
                if LLMOrchestrator is None:
                    return None
                
                # Determine theme from intent
                intent = r.get('intent', 'hotel_search')
                theme = 'visa' if intent == 'visa_check' else 'hotel'
                orchestrator = LLMOrchestrator(theme=theme)
                
                # Add available providers
                if HuggingFaceProvider:
                    # Try Inference API first (better models) if token available
                    import os
                    hf_token = os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_API_TOKEN') or "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"
                    
                    # Focus on Gemma 2 2B model only
                    # Try different model name variations
                    gemma_models_to_try = [
                        "google/gemma-2-2b-it",  # Instruction-tuned version
                        "google/gemma-2b-it",     # Alternative name
                        "google/gemma-2-2b",      # Base model
                    ]
                    
                    gemma_provider = None
                    for model_name in gemma_models_to_try:
                        try:
                            test_provider = HuggingFaceProvider(
                                model_name=model_name,
                                use_inference_api=True,
                                hf_token=hf_token
                            )
                            if test_provider.available:
                                gemma_provider = test_provider
                                gemma_provider.display_name = "Gemma 2 2B"
                                gemma_provider.is_instruction_model = True
                                orchestrator.add_provider(gemma_provider)
                                print(f"[OK] Loaded Gemma model: {model_name}")
                                break
                        except Exception as e:
                            print(f"[SKIP] Failed to load {model_name}: {e}")
                            continue
                    
                    if gemma_provider is None:
                        print("[WARN] Could not load any Gemma model variant")
                
                if OpenAIProvider:
                    openai_provider = OpenAIProvider(model_name="gpt-3.5-turbo")
                    if openai_provider.available:
                        orchestrator.add_provider(openai_provider)
                
                if AnthropicProvider:
                    anthropic_provider = AnthropicProvider(model_name="claude-3-haiku-20240307")
                    if anthropic_provider.available:
                        orchestrator.add_provider(anthropic_provider)
                
                return orchestrator
            
            orchestrator = load_llm_orchestrator()
            
            if orchestrator is None or not orchestrator.providers:
                st.warning("⚠️ LLM layer not available. Ensure 03_llm_layer.py exists and required packages are installed.")
                st.code("pip install transformers torch openai anthropic")
            else:
                available_models = list(orchestrator.providers.keys())
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("LLM Model Selection")
                    
                    # Show model selection options
                    selection_mode = st.radio(
                        "Selection mode:",
                        ["Single model", "Compare multiple models"],
                        key="model_selection_mode"
                    )
                    
                    if selection_mode == "Single model":
                        selected_model = st.selectbox(
                            "Choose an LLM:",
                            available_models,
                            key="llm_model_select"
                        )
                        compare_models = False
                        selected_models = None
                    else:
                        # Multi-select for comparison
                        selected_models = st.multiselect(
                            "Choose models to compare:",
                            available_models,
                            default=available_models[:min(3, len(available_models))],  # Default to first 3
                            key="llm_models_multiselect"
                        )
                        compare_models = len(selected_models) > 0
                        selected_model = selected_models[0] if selected_models else None
                
                with col2:
                    st.subheader("Prompt Structure")
                    with st.expander("View Prompt Components"):
                        # Show merged context
                        if orchestrator.context_merger:
                            merged_context = orchestrator.context_merger.merge_results(r)
                            formatted_context = orchestrator.context_merger.format_context(merged_context)
                            
                            st.markdown("**Context:**")
                            st.text_area("", formatted_context, height=150, disabled=True, key="context_view")
                            
                            st.markdown("**Persona:**")
                            st.text(orchestrator.prompt_builder.persona)
                            
                            st.markdown("**Task:**")
                            st.text("Answer the user's question using ONLY the information provided in the context above.")
                
                # Generate response
                if st.button("Generate Response", key="generate_llm_response"):
                    if selection_mode == "Compare multiple models" and selected_models and len(selected_models) > 1:
                        # Compare selected models
                        models_to_compare = selected_models
                    elif compare_models and len(available_models) > 1:
                        # Compare all available models
                        models_to_compare = available_models
                    else:
                        models_to_compare = None
                    
                    if models_to_compare:
                        # Compare selected models
                        with st.spinner(f"Comparing {len(models_to_compare)} models..."):
                            responses = {}
                            for model_name in models_to_compare:
                                if model_name in orchestrator.providers:
                                    responses[model_name] = orchestrator.generate_response(
                                        stored_query, r, model_name=model_name
                                    )
                        
                        st.subheader("Model Comparison Results")
                        
                        # Quantitative metrics
                        st.markdown("**Quantitative Metrics:**")
                        metrics_data = []
                        for model_name, response in responses.items():
                            metrics_data.append({
                                "Model": model_name,
                                "Response Time (s)": f"{response.response_time:.2f}",
                                "Tokens": response.tokens_used or "N/A",
                                "Cost": f"${response.cost:.4f}" if response.cost else "Free",
                                "Response Length": len(response.text),
                                "Error": "Yes" if response.error else "No"
                            })
                        st.dataframe(metrics_data)
                        
                        # Qualitative evaluation
                        st.markdown("**Qualitative Evaluation:**")
                        for model_name, response in responses.items():
                            with st.expander(f"{model_name} Response"):
                                if response.error:
                                    st.error(f"Error: {response.error}")
                                else:
                                    st.markdown(response.text)
                                    st.caption(f"Generated in {response.response_time:.2f}s | Tokens: {response.tokens_used or 'N/A'} | Cost: ${response.cost:.4f}" if response.cost else f"Generated in {response.response_time:.2f}s | Tokens: {response.tokens_used or 'N/A'} | Free")
                    else:
                        # Single model response
                        with st.spinner(f"Generating response with {selected_model}..."):
                            response = orchestrator.generate_response(stored_query, r, model_name=selected_model)
                        
                        st.subheader("Generated Response")
                        if response.error:
                            st.error(f"Error: {response.error}")
                        else:
                            st.markdown(response.text)
                            
                            # Show metadata
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Response Time", f"{response.response_time:.2f}s")
                            with col2:
                                st.metric("Tokens Used", response.tokens_used or "N/A")
                            with col3:
                                st.metric("Cost", f"${response.cost:.4f}" if response.cost else "Free")
                            with col4:
                                st.metric("Response Length", len(response.text))
                            
                            # Show full prompt if requested
                            with st.expander("View Full Prompt"):
                                merged_context = orchestrator.context_merger.merge_results(r)
                                formatted_context = orchestrator.context_merger.format_context(merged_context)
                                full_prompt = orchestrator.prompt_builder.build_prompt(stored_query, formatted_context)
                                st.text_area("", full_prompt, height=300, disabled=True)

else:
    st.info("👈 Enter a query in the sidebar and click 'Process Query' to see the pipeline in action.")


