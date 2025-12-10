"""
Graph-RAG Travel Assistant - ChatGPT-like Interface
=====================================================================
A clean, user-friendly chat interface for the Graph-RAG Travel Assistant.
Users can chat naturally and view technical details only when needed.
=====================================================================
"""

import streamlit as st
import json
from typing import Dict, Any, List
import importlib.util
import os

# Import modules
spec = importlib.util.spec_from_file_location("input_preprocessing", "01_input_preprocessing.py")
input_preprocessing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(input_preprocessing)
InputPreprocessor = input_preprocessing.InputPreprocessor

spec2 = importlib.util.spec_from_file_location("graph_retrieval", "02_graph_retrieval.py")
graph_retrieval = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(graph_retrieval)
BaselineRetriever = getattr(graph_retrieval, 'BaselineRetriever', None)
EmbeddingRetriever = getattr(graph_retrieval, 'EmbeddingRetriever', None)
HybridRetriever = getattr(graph_retrieval, 'HybridRetriever', None)
load_neo4j_config = getattr(graph_retrieval, 'load_neo4j_config', None)

spec3 = importlib.util.spec_from_file_location("llm_layer", "03_llm_layer.py")
llm_layer = importlib.util.module_from_spec(spec3)
spec3.loader.exec_module(llm_layer)
LLMOrchestrator = getattr(llm_layer, 'LLMOrchestrator', None)
HuggingFaceProvider = getattr(llm_layer, 'HuggingFaceProvider', None)
OpenAIProvider = getattr(llm_layer, 'OpenAIProvider', None)
AnthropicProvider = getattr(llm_layer, 'AnthropicProvider', None)

# Page config
st.set_page_config(
    page_title="Travel Assistant",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for ChatGPT-like styling
st.markdown("""
<style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Chat message styling */
    .user-message {
        background-color: #f0f0f0;
        padding: 12px 16px;
        border-radius: 18px;
        margin: 8px 0;
        margin-left: 20%;
        text-align: left;
    }
    
    .assistant-message {
        background-color: #e8f4f8;
        padding: 12px 16px;
        border-radius: 18px;
        margin: 8px 0;
        margin-right: 20%;
        text-align: left;
    }
    
    /* Chat container */
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
        padding: 20px;
    }
    
    /* Input area */
    .stTextInput > div > div > input {
        border-radius: 24px;
        padding: 12px 20px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'orchestrator' not in st.session_state:
    st.session_state.orchestrator = None

if 'available_models' not in st.session_state:
    st.session_state.available_models = []

if 'selected_model' not in st.session_state:
    st.session_state.selected_model = None
if 'retrieval_method' not in st.session_state:
    st.session_state.retrieval_method = 'hybrid'
if 'compare_models' not in st.session_state:
    st.session_state.compare_models = False
if 'comparison_models' not in st.session_state:
    st.session_state.comparison_models = []


# Helper to present a user-friendly model name
def get_model_display_name(model_name: str) -> str:
    if not model_name:
        return ""
    if st.session_state.orchestrator:
        provider = st.session_state.orchestrator.providers.get(model_name)
        display_name = getattr(provider, 'display_name', None) if provider else None
        if display_name:
            return display_name
    # Fallback to known labels
    fallback = {
        "google/gemma-2-2b-it": "Gemma 2 2B",
        "mistralai/Mistral-7B-Instruct-v0.2": "Mistral 7B",
        "meta-llama/Llama-3.2-1B-Instruct": "Llama 3.2 1B",
        "gpt-3.5-turbo": "GPT-3.5 Turbo",
        "claude-3-haiku-20240307": "Claude 3 Haiku",
    }
    return fallback.get(model_name, model_name)


def format_visa_response(from_country: str, visa_rows: List[Dict[str, Any]]) -> str:
    if not visa_rows:
        return "I couldn't find visa information for your route."

    def normalize_status(status: str) -> str:
        return status.lower() if status else ""

    sentences = []
    origin = from_country or "your country"

    for row in visa_rows:
        country = row.get('country_name') or row.get('to_name') or row.get('to') or row.get('to_country')
        status = row.get('visa_status') or row.get('v.visa_type') or row.get('visa_type') or 'No visa required'
        norm = normalize_status(status)

        if not country:
            continue

        if 'no visa' in norm or 'visa free' in norm or 'visa-free' in norm:
            sentences.append(f"You can visit {country} from {origin} without a visa.")
        elif 'on arrival' in norm:
            sentences.append(f"{country} offers visa on arrival when traveling from {origin}.")
        elif 'e-visa' in norm or 'electronic' in norm:
            sentences.append(f"You need an e-visa to visit {country} from {origin}.")
        else:
            sentences.append(f"You need {status} to visit {country} from {origin}.")

    if not sentences:
        return "I couldn't find visa information for your route."

    if len(sentences) == 1:
        return sentences[0]

    bullets = "\n- " + "\n- ".join(sentences)
    return f"Here is what I found:\n{bullets}"

# Initialize components
@st.cache_resource
def initialize_components(embedding_model='minilm', use_features=True, feature_weight=0.3):
    """Initialize preprocessing, retrieval, and LLM components.
    
    Args:
        embedding_model: Embedding model to use ('minilm', 'mpnet', 'bge')
        use_features: Whether to use feature-based embeddings
        feature_weight: Weight for feature embeddings (0-1)
    """
    # Preprocessor
    preprocessor = InputPreprocessor()
    
    # Load Neo4j config
    if load_neo4j_config:
        neo4j_config = load_neo4j_config('config.txt')
    else:
        # Fallback: manual config loading
        config_path = 'config.txt'
        neo4j_config = {'uri': None, 'username': None, 'password': None}
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if '=' in line:
                            key, value = line.strip().split('=', 1)
                            key = key.strip().lower()
                            if key == 'uri':
                                neo4j_config['uri'] = value.strip()
                            elif key == 'username':
                                neo4j_config['username'] = value.strip()
                            elif key == 'password':
                                neo4j_config['password'] = value.strip()
            except Exception as e:
                print(f"Warning: could not load config.txt: {e}")
    
    # Retrievers
    baseline = None
    if BaselineRetriever and neo4j_config['uri']:
        try:
            baseline = BaselineRetriever(
                uri=neo4j_config['uri'],
                username=neo4j_config['username'],
                password=neo4j_config['password']
            )
        except Exception as e:
            print(f"Warning: Could not connect to Neo4j: {e}")
            baseline = None
    
    embedding = None
    if EmbeddingRetriever:
        try:
            # Get embedder from preprocessor
            embedder = getattr(preprocessor, 'embedder', None)
            if embedder and hasattr(embedder, 'model'):
                embedding = EmbeddingRetriever(
                    embedder=embedder,
                    embedding_model=embedding_model,
                    use_features=use_features,
                    feature_weight=feature_weight
                )
        except Exception as e:
            print(f"Warning: Could not initialize embedding retriever: {e}")
    
    hybrid = HybridRetriever(baseline, embedding, baseline_weight=0.6, embedding_weight=0.4) if HybridRetriever else None
    
    # LLM Orchestrator
    orchestrator = None
    if LLMOrchestrator:
        orchestrator = LLMOrchestrator(theme='hotel')

        # Add three free HuggingFace models (Gemma, Mistral, Llama) if available
        if HuggingFaceProvider:
            hf_token = os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_API_TOKEN') or "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"
            hf_models = [
                ("google/gemma-2-2b-it", "Gemma 2 2B", "Fast & smart"),
                ("mistralai/Mistral-7B-Instruct-v0.2", "Mistral 7B", "Balanced & capable"),
                ("meta-llama/Llama-3.2-1B-Instruct", "Llama 3.2 1B", "Lightweight & quick"),
            ]

            for model_name, display_name, _desc in hf_models:
                try:
                    provider = HuggingFaceProvider(
                        model_name=model_name,
                        use_inference_api=True,
                        hf_token=hf_token
                    )
                    if provider.available:
                        provider.display_name = display_name
                        provider.is_instruction_model = True
                        orchestrator.add_provider(provider)
                except Exception as e:
                    print(f"Warning: could not load {model_name}: {e}")

        # Add OpenAI provider
        if OpenAIProvider:
            try:
                provider = OpenAIProvider(model_name="gpt-3.5-turbo")
                if provider.available:
                    orchestrator.add_provider(provider)
            except Exception as e:
                print(f"Warning: OpenAI provider error: {e}")

        # Add Anthropic provider
        if AnthropicProvider:
            try:
                provider = AnthropicProvider(model_name="claude-3-haiku-20240307")
                if provider.available:
                    orchestrator.add_provider(provider)
            except Exception as e:
                print(f"Warning: Anthropic provider error: {e}")
    
    return preprocessor, hybrid, orchestrator

# Session state for embedding settings
if 'embedding_model' not in st.session_state:
    st.session_state.embedding_model = 'minilm'
if 'use_features' not in st.session_state:
    st.session_state.use_features = True
if 'feature_weight' not in st.session_state:
    st.session_state.feature_weight = 0.3

# Initialize with current settings
preprocessor, hybrid_retriever, orchestrator = initialize_components(
    embedding_model=st.session_state.embedding_model,
    use_features=st.session_state.use_features,
    feature_weight=st.session_state.feature_weight
)

if orchestrator:
    st.session_state.orchestrator = orchestrator
    st.session_state.available_models = list(orchestrator.providers.keys())
    # Only set a default if nothing is selected yet
    if st.session_state.available_models and not st.session_state.selected_model:
        st.session_state.selected_model = st.session_state.available_models[0]

# Sidebar for settings
with st.sidebar:
    st.title("⚙️ Settings")
    st.divider()
    
    # Embedding model selection
    st.subheader("🔍 Embedding Settings")
    
    embedding_models = {
        'minilm': 'MiniLM-L6 (384-dim, Fast)',
        'mpnet': 'MPNet (768-dim, High Quality)',
        'bge': 'BGE-Small (384-dim, Retrieval-Optimized)'
    }
    
    selected_embedding = st.selectbox(
        "Embedding Model:",
        options=list(embedding_models.keys()),
        format_func=lambda x: embedding_models[x],
        index=list(embedding_models.keys()).index(st.session_state.embedding_model),
        key="embedding_selector"
    )
    
    use_features = st.checkbox(
        "Use Feature Embeddings",
        value=st.session_state.use_features,
        help="Combine text embeddings with numerical hotel attributes (star rating, quality scores)"
    )
    
    if use_features:
        feature_weight = st.slider(
            "Feature Weight:",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.feature_weight,
            step=0.1,
            help="Balance between text (0.0) and feature (1.0) embeddings"
        )
    else:
        feature_weight = 0.0

    # Retrieval method selection
    st.subheader("🧭 Retrieval Method")
    retrieval_options = {
        'hybrid': 'Hybrid (baseline + embeddings)',
        'baseline': 'Baseline only (Cypher)',
        'embeddings': 'Embeddings only'
    }
    selected_retrieval = st.selectbox(
        "Retrieval Mode:",
        options=list(retrieval_options.keys()),
        format_func=lambda x: retrieval_options[x],
        index=list(retrieval_options.keys()).index(st.session_state.retrieval_method),
        key="retrieval_selector"
    )
    
    # Check if settings changed and need reinitialization
    settings_changed = (
        selected_embedding != st.session_state.embedding_model or
        use_features != st.session_state.use_features or
        abs(feature_weight - st.session_state.feature_weight) > 0.01
    )
    retrieval_changed = selected_retrieval != st.session_state.retrieval_method
    
    if settings_changed or retrieval_changed:
        if st.button("Apply Embedding Settings", use_container_width=True):
            st.session_state.embedding_model = selected_embedding
            st.session_state.use_features = use_features
            st.session_state.feature_weight = feature_weight
            st.session_state.retrieval_method = selected_retrieval
            st.cache_resource.clear()  # Clear cache to reinitialize
            st.rerun()
        st.caption("⚠️ Click to apply changes")
    else:
        st.caption("✅ Settings active")
    
    st.divider()
    
    # Clear chat button
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    
    # About section
    with st.expander("ℹ️ About"):
        st.markdown("""
        **Graph-RAG Travel Assistant**
        
        Ask questions about hotels, travel, and visa requirements.
        
        The AI uses a knowledge graph to provide accurate, 
        context-aware answers.
        """)

# Main chat interface
st.title("✈️ Travel Assistant")
st.caption("Ask me anything about hotels, travel, or visa requirements")

# Model selector in main UI
if st.session_state.available_models:
    col1, col2, col3 = st.columns([2, 3, 2])
    
    with col1:
        st.markdown("**🤖 AI Model:**")
    
    with col2:
        # Model descriptions for user-friendly display
        model_descriptions = {
            "google/gemma-2-2b-it": ("✨ Gemma", "Fast & smart"),
            "mistralai/Mistral-7B-Instruct-v0.2": ("🌊 Mistral", "Balanced & capable"),
            "meta-llama/Llama-3.2-1B-Instruct": ("🦙 Llama", "Lightweight & quick"),
            "gpt-3.5-turbo": ("⚡ GPT-3.5", "Powerful & reliable"),
            "claude-3-haiku-20240307": ("🧠 Claude", "Thoughtful & nuanced"),
        }
        
        # Create button-style selection
        selected_idx = 0
        if st.session_state.selected_model in st.session_state.available_models:
            selected_idx = st.session_state.available_models.index(st.session_state.selected_model)
        
        model_cols = st.columns(len(st.session_state.available_models))
        for idx, model_name in enumerate(st.session_state.available_models):
            with model_cols[idx]:
                # Prefer provider display name if available
                provider_display = None
                if st.session_state.orchestrator:
                    provider = st.session_state.orchestrator.providers.get(model_name)
                    provider_display = getattr(provider, 'display_name', None) if provider else None

                default_name, default_desc = model_descriptions.get(model_name, (model_name[:15], ""))
                short_name = provider_display or default_name
                description = default_desc
                is_selected = (idx == selected_idx)

                if st.button(
                    f"{short_name}\n{description}",
                    key=f"model_btn_{model_name}",
                    use_container_width=True
                ):
                    st.session_state.selected_model = model_name
                    st.rerun()
    
    with col3:
        pass

    # Model comparison controls
    st.markdown("**🔁 Model Comparison**")
    compare_enabled = st.checkbox(
        "Compare multiple models for each query",
        value=st.session_state.compare_models,
        key="compare_models"
    )

    if compare_enabled:
        default_models = st.session_state.comparison_models or st.session_state.available_models
        st.session_state.comparison_models = st.multiselect(
            "Models to compare:",
            options=st.session_state.available_models,
            default=default_models,
            key="comparison_models"
        )
    else:
        st.session_state.comparison_models = []

st.divider()

# Template questions section
st.markdown("### 💡 Example Questions")
st.markdown("Click a question below or write your own:")

# Template questions organized by category (matching Cypher query examples)
template_questions = {
    "🏨 Hotel Search & Details": [
        "Tell me about The Azure Tower hotel",
        "Find hotels in France",
    ],
    "🏆 Quality & Amenities": [
        "What's the best facilities hotels in Egypt?",
        "What are the facilities in The Golden Oasis?",
        "Where's the cleanest hotel in Egypt",
        "Find clean hotels in Japan with score above 8",
        "Best located hotel in New Zealand",
        "The most comfortable hotel in Turkey",
        "Hotel with the best staff in India",
        "Best facilities in Brazil",
    ],
    "💰 Value & Pricing": [
        "What are the best value hotels in Singapore?",
    ],
    "👥 Traveller Preferences": [
        "Best hotels for solo travellers in Dubai",
        "Best hotels for business travellers in Egypt",
        "Recommend hotels for business travelers in Russia",
    ],
    "🍽️ Specific Amenities": [
        "Is there hotels with a breakfast buffet in Russia?",
        "Is there hotels with a laundry services in Thailand?",
    ],
    "✈️ Visa & Travel Requirements": [
        "Do I need a visa to visit France from United States?",
        "Visa requirements from India to Dubai",
        "Where can I go without a visa traveling from Japan",
        "Can I travel from China to Singapore without a visa",
    ],
}

# Create tabs for different question categories
col1, col2, col3 = st.columns(3)
cols = [col1, col2, col3]
col_idx = 0

for category, questions in template_questions.items():
    with cols[col_idx % 3]:
        st.markdown(f"**{category}**")
        for question in questions:
            if st.button(question, key=f"template_{question}", use_container_width=True):
                # Set the question and trigger processing
                user_input = question
                st.session_state.messages.append({"role": "user", "content": user_input})
                
                # Process query
                with st.spinner("Thinking..."):
                    # Preprocessing
                    preprocessed = preprocessor.process(user_input)
                    intent = preprocessed.get('intent', 'hotel_search')
                    entities = preprocessed.get('entities', {})
                    query_embedding = preprocessed.get('query_embedding')
                    
                    # Retrieval
                    retrieval_method = st.session_state.retrieval_method
                    retrieval_result = {}
                    if hybrid_retriever:
                        try:
                            retrieval_result = hybrid_retriever.retrieve(
                                intent=intent,
                                entities=entities,
                                query_embedding=query_embedding,
                                method=retrieval_method
                            )
                            if not isinstance(retrieval_result, dict):
                                retrieval_result = {}
                            retrieval_result.setdefault('intent', intent)
                            retrieval_result.setdefault('method', retrieval_method)
                            retrieval_result.setdefault('baseline_results', [])
                            retrieval_result.setdefault('embedding_results', [])
                            retrieval_result.setdefault('merged_results', [])
                        except Exception as e:
                            st.error(f"Retrieval error: {str(e)}")
                            retrieval_result = {
                                'intent': intent,
                                'method': retrieval_method,
                                'baseline_results': [],
                                'embedding_results': [],
                                'merged_results': [],
                                'error': str(e)
                            }
                    else:
                        retrieval_result = {
                            'intent': intent,
                            'method': retrieval_method,
                            'baseline_results': [],
                            'embedding_results': [],
                            'merged_results': [],
                            'error': 'No retriever available'
                        }
                    
                    # Generate response
                    response_text = "I'm sorry, I couldn't generate a response."
                    technical_details = {}

                    # Fast-path for visa queries
                    if intent == 'visa_check' and retrieval_result.get('baseline_results'):
                        visa_rows = retrieval_result.get('baseline_results', [])
                        from_country = None
                        if isinstance(entities.get('country'), list):
                            from_country = entities.get('country')[0]
                        else:
                            from_country = entities.get('country')

                        response_text = format_visa_response(from_country, visa_rows)
                        formatted_context = "\n".join([
                            f"- {(row.get('country_name') or row.get('to_name') or row.get('to') or row.get('to_country') or 'Unknown')}: "
                            f"{row.get('visa_status') or row.get('v.visa_type') or row.get('visa_type') or 'No visa required'}"
                            for row in visa_rows
                        ])

                        technical_details = {
                            "query": user_input,
                            "intent": intent,
                            "entities": entities,
                            "retrieval_method": retrieval_method,
                            "query_number": retrieval_result.get('query_number'),
                            "chosen_intent": retrieval_result.get('chosen_intent'),
                            "cypher_template": retrieval_result.get('cypher_template'),
                            "baseline_results": visa_rows,
                            "embedding_results": [],
                            "merged_results": visa_rows,
                            "baseline_status": retrieval_result.get('baseline_status', 'unknown'),
                            "embedding_status": 'skipped',
                            "formatted_context": formatted_context,
                            "context_item_count": len(visa_rows),
                            "knowledge_graph_data": {
                                "baseline_count": len(visa_rows),
                                "embedding_count": 0,
                                "merged_count": len(visa_rows),
                                "method": retrieval_method,
                                "has_results": True,
                            },
                        }

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response_text,
                            "technical_details": technical_details,
                            "show_details": False,
                        })
                        st.rerun()
                    
                    has_results = (
                        len(retrieval_result.get('baseline_results', [])) > 0 or
                        len(retrieval_result.get('embedding_results', [])) > 0 or
                        len(retrieval_result.get('merged_results', [])) > 0
                    )
                    
                    # Decide which models to run (single or comparison)
                    models_to_run = []
                    if st.session_state.orchestrator:
                        if st.session_state.compare_models and st.session_state.comparison_models:
                            models_to_run = st.session_state.comparison_models
                        elif st.session_state.selected_model:
                            models_to_run = [st.session_state.selected_model]

                    # Prepare shared context formatting once
                    merged_context = None
                    formatted_context = None
                    if st.session_state.orchestrator and st.session_state.orchestrator.context_merger:
                        try:
                            merged_context = st.session_state.orchestrator.context_merger.merge_results(retrieval_result)
                            formatted_context = st.session_state.orchestrator.context_merger.format_context(merged_context)
                        except:
                            pass

                    base_details = {
                        "query": user_input,
                        "intent": intent,
                        "entities": entities,
                        "retrieval_method": retrieval_method,
                        "baseline_results": retrieval_result.get('baseline_results', []),
                        "embedding_results": retrieval_result.get('embedding_results', []),
                        "merged_results": retrieval_result.get('merged_results', []),
                        "baseline_status": retrieval_result.get('baseline_status', 'unknown'),
                        "embedding_status": retrieval_result.get('embedding_status', 'unknown'),
                        "embedding_model": st.session_state.embedding_model,
                        "retrieval_mode": f"text+features (w={st.session_state.feature_weight:.1f})" if st.session_state.use_features else "text_only",
                        "merge_method": retrieval_result.get('merge_method', 'N/A'),
                        "formatted_context": formatted_context,
                        "context_item_count": merged_context.get('total_count', 0) if merged_context else 0,
                        "knowledge_graph_data": {
                            "baseline_count": len(retrieval_result.get('baseline_results', [])),
                            "embedding_count": len(retrieval_result.get('embedding_results', [])),
                            "merged_count": len(retrieval_result.get('merged_results', [])),
                            "method": retrieval_method,
                            "has_results": has_results
                        }
                    }

                    if models_to_run:
                        for model_name in models_to_run:
                            model_used = None
                            response_text_model = response_text
                            try:
                                llm_response = st.session_state.orchestrator.generate_response(
                                    user_query=user_input,
                                    retrieval_result=retrieval_result,
                                    model_name=model_name
                                )

                                if not llm_response.error:
                                    response_text_model = llm_response.text
                                    model_used = get_model_display_name(model_name)
                                else:
                                    response_text_model = f"I encountered an error: {llm_response.error}"
                                    if not has_results:
                                        response_text_model += "\n\nNote: No results were found in the knowledge graph for your query."
                            except Exception as e:
                                import traceback
                                error_msg = str(e)
                                response_text_model = f"I encountered an error: {error_msg}"
                                if not has_results:
                                    response_text_model += "\n\nNote: No results were found in the knowledge graph. This might be why the response failed."
                                technical_details = {
                                    "query": user_input,
                                    "intent": intent,
                                    "entities": entities,
                                    "error": error_msg,
                                    "traceback": traceback.format_exc()
                                }
                                st.session_state.messages.append({
                                    "role": "assistant",
                                    "content": response_text_model,
                                    "model_used": model_used,
                                    "technical_details": technical_details,
                                    "show_details": False
                                })
                                continue

                            technical_details = dict(base_details)
                            technical_details["model_name"] = model_name

                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": response_text_model,
                                "model_used": model_used,
                                "technical_details": technical_details,
                                "show_details": False
                            })
                    else:
                        if has_results:
                            hotels = []
                            if retrieval_result.get('merged_results'):
                                hotels = retrieval_result['merged_results'][:3]
                            elif retrieval_result.get('baseline_results'):
                                hotels = retrieval_result['baseline_results'][:3]
                            elif retrieval_result.get('embedding_results'):
                                hotels = retrieval_result['embedding_results'][:3]
                            
                            if hotels:
                                hotel_names = []
                                for h in hotels:
                                    name = h.get('h.name') or h.get('hotel_name') or h.get('name', 'Unknown')
                                    hotel_names.append(name)
                                response_text = f"I found {len(hotels)} hotel(s). Here are some: {', '.join(hotel_names)}"
                            else:
                                response_text = "I found some results but couldn't format them properly."
                        else:
                            response_text = "I couldn't find relevant information in the knowledge graph. Please try rephrasing your question or check if the data contains information about your query."
                        technical_details = dict(base_details)

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response_text,
                            "technical_details": technical_details,
                            "show_details": False
                        })
                
                st.rerun()
        col_idx += 1

st.divider()

# Display chat history
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]
        technical_details = message.get("technical_details", {})
        show_details = message.get("show_details", False)
        
        if role == "user":
            with st.chat_message("user"):
                st.markdown(content)
        else:  # assistant
            with st.chat_message("assistant"):
                model_used = message.get("model_used")
                if model_used:
                    st.caption(f"Model: {model_used}")
                st.markdown(content)
                
                # Technical details button
                if technical_details:
                    with st.expander("🔍 View Technical Details", expanded=show_details):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Query Number and Intent
                            if technical_details.get("query_number"):
                                st.markdown(f"**Query #{technical_details['query_number']}** - {technical_details.get('chosen_intent', 'N/A')}")
                            
                            st.markdown("**Cypher Query Used:**")
                            st.code(technical_details.get("cypher_template", "N/A"), language="cypher")
                            
                            st.markdown("**Intent:**")
                            st.info(technical_details.get("intent", "N/A"))
                            
                            st.markdown("**Entities Extracted:**")
                            st.json(technical_details.get("entities", {}))
                            
                            # Show embedding model info
                            if technical_details.get("embedding_model"):
                                st.markdown("**Embedding Model:**")
                                st.info(f"{technical_details['embedding_model']} | Mode: {technical_details.get('retrieval_mode', 'N/A')}")
                        
                        with col2:
                            st.markdown("**Retrieval Method:**")
                            st.info(technical_details.get("retrieval_method", "N/A"))
                            
                            # Show merge method if hybrid
                            if technical_details.get("merge_method"):
                                st.markdown("**Merge Strategy:**")
                                st.info(technical_details["merge_method"])
                            
                            st.markdown("**Results Found:**")
                            baseline_count = len(technical_details.get("baseline_results", []))
                            embedding_count = len(technical_details.get("embedding_results", []))
                            merged_count = len(technical_details.get("merged_results", []))
                            st.metric("Baseline Results", baseline_count)
                            st.metric("Embedding Results", embedding_count)
                            st.metric("Merged Results", merged_count)
                            
                            if technical_details.get("formatted_context"):
                                st.markdown("**Context Sent to LLM:**")
                                st.text_area("Context", technical_details["formatted_context"], height=200, disabled=True, key=f"context_display_{hash(str(technical_details))}", label_visibility="collapsed")
                            
                            if technical_details.get("knowledge_graph_data"):
                                st.markdown("**Knowledge Graph:**")
                                st.json(technical_details["knowledge_graph_data"])

# Chat input
user_input = st.chat_input("Type your message here...")

if user_input:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Process query
    with st.spinner("Thinking..."):
        # Preprocessing
        preprocessed = preprocessor.process(user_input)
        intent = preprocessed.get('intent', 'hotel_search')
        entities = preprocessed.get('entities', {})
        query_embedding = preprocessed.get('query_embedding')
        
        # Retrieval
        retrieval_method = st.session_state.retrieval_method
        retrieval_result = {}
        if hybrid_retriever:
            try:
                retrieval_result = hybrid_retriever.retrieve(
                    intent=intent,
                    entities=entities,
                    query_embedding=query_embedding,
                    method=retrieval_method
                )
                # Ensure retrieval_result has required keys
                if not isinstance(retrieval_result, dict):
                    retrieval_result = {}
                retrieval_result.setdefault('intent', intent)
                retrieval_result.setdefault('method', retrieval_method)
                retrieval_result.setdefault('baseline_results', [])
                retrieval_result.setdefault('embedding_results', [])
                retrieval_result.setdefault('merged_results', [])
            except Exception as e:
                st.error(f"Retrieval error: {str(e)}")
                retrieval_result = {
                    'intent': intent,
                    'method': retrieval_method,
                    'baseline_results': [],
                    'embedding_results': [],
                    'merged_results': [],
                    'error': str(e)
                }
        else:
            retrieval_result = {
                'intent': intent,
                'method': retrieval_method,
                'baseline_results': [],
                'embedding_results': [],
                'merged_results': [],
                'error': 'No retriever available'
            }
        
        # Generate response
        response_text = "I'm sorry, I couldn't generate a response."
        technical_details = {}

        # Fast-path for visa queries: format baseline results into a clear, natural answer
        if intent == 'visa_check' and retrieval_result.get('baseline_results'):
            visa_rows = retrieval_result.get('baseline_results', [])
            from_country = None
            # Try to get source country from params/entities
            if isinstance(entities.get('country'), list):
                from_country = entities.get('country')[0]
            else:
                from_country = entities.get('country')

            response_text = format_visa_response(from_country, visa_rows)
            formatted_context = "\n".join([
                f"- {(row.get('country_name') or row.get('to_name') or row.get('to') or row.get('to_country') or 'Unknown')}: "
                f"{row.get('visa_status') or row.get('v.visa_type') or row.get('visa_type') or 'No visa required'}"
                for row in visa_rows
            ])

            technical_details = {
                "query": user_input,
                "intent": intent,
                "entities": entities,
                "retrieval_method": retrieval_method,
                "query_number": retrieval_result.get('query_number'),
                "chosen_intent": retrieval_result.get('chosen_intent'),
                "cypher_template": retrieval_result.get('cypher_template'),
                "baseline_results": visa_rows,
                "embedding_results": [],
                "merged_results": visa_rows,
                "baseline_status": retrieval_result.get('baseline_status', 'unknown'),
                "embedding_status": 'skipped',
                "formatted_context": formatted_context,
                "context_item_count": len(visa_rows),
                "knowledge_graph_data": {
                    "baseline_count": len(visa_rows),
                    "embedding_count": 0,
                    "merged_count": len(visa_rows),
                    "method": retrieval_method,
                    "has_results": True,
                },
            }

            # Emit assistant message and rerun to show it in chat
            st.session_state.messages.append({
                "role": "assistant",
                "content": response_text,
                "technical_details": technical_details,
                "show_details": False,
            })
            st.rerun()
        
        # Check if we have any results
        has_results = (
            len(retrieval_result.get('baseline_results', [])) > 0 or
            len(retrieval_result.get('embedding_results', [])) > 0 or
            len(retrieval_result.get('merged_results', [])) > 0
        )
        
        # Decide which models to run (single or comparison)
        models_to_run = []
        if st.session_state.orchestrator:
            if st.session_state.compare_models and st.session_state.comparison_models:
                models_to_run = st.session_state.comparison_models
            elif st.session_state.selected_model:
                models_to_run = [st.session_state.selected_model]

        # Prepare shared context formatting once
        merged_context = None
        formatted_context = None
        if st.session_state.orchestrator and st.session_state.orchestrator.context_merger:
            try:
                merged_context = st.session_state.orchestrator.context_merger.merge_results(retrieval_result)
                formatted_context = st.session_state.orchestrator.context_merger.format_context(merged_context)
            except:
                pass

        base_details = {
            "query": user_input,
            "intent": intent,
            "entities": entities,
            "retrieval_method": retrieval_method,
            "query_number": retrieval_result.get('query_number'),
            "chosen_intent": retrieval_result.get('chosen_intent'),
            "cypher_template": retrieval_result.get('cypher_template'),
            "baseline_results": retrieval_result.get('baseline_results', []),
            "embedding_results": retrieval_result.get('embedding_results', []),
            "merged_results": retrieval_result.get('merged_results', []),
            "baseline_status": retrieval_result.get('baseline_status', 'unknown'),
            "embedding_status": retrieval_result.get('embedding_status', 'unknown'),
            "embedding_model": st.session_state.embedding_model,
            "retrieval_mode": f"text+features (w={st.session_state.feature_weight:.1f})" if st.session_state.use_features else "text_only",
            "merge_method": retrieval_result.get('merge_method', 'N/A'),
            "formatted_context": formatted_context,
            "context_item_count": merged_context.get('total_count', 0) if merged_context else 0,
            "knowledge_graph_data": {
                "baseline_count": len(retrieval_result.get('baseline_results', [])),
                "embedding_count": len(retrieval_result.get('embedding_results', [])),
                "merged_count": len(retrieval_result.get('merged_results', [])),
                "method": retrieval_method,
                "has_results": has_results
            }
        }

        if models_to_run:
            for model_name in models_to_run:
                model_used = None
                response_text_model = response_text
                try:
                    llm_response = st.session_state.orchestrator.generate_response(
                        user_query=user_input,
                        retrieval_result=retrieval_result,
                        model_name=model_name
                    )

                    if not llm_response.error:
                        response_text_model = llm_response.text
                        model_used = get_model_display_name(model_name)
                    else:
                        response_text_model = f"I encountered an error: {llm_response.error}"
                        if not has_results:
                            response_text_model += "\n\nNote: No results were found in the knowledge graph for your query."
                except Exception as e:
                    import traceback
                    error_msg = str(e)
                    response_text_model = f"I encountered an error: {error_msg}"
                    if not has_results:
                        response_text_model += "\n\nNote: No results were found in the knowledge graph. This might be why the response failed."
                    technical_details = {
                        "query": user_input,
                        "intent": intent,
                        "entities": entities,
                        "error": error_msg,
                        "traceback": traceback.format_exc()
                    }
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response_text_model,
                        "model_used": model_used,
                        "technical_details": technical_details,
                        "show_details": False
                    })
                    continue

                technical_details = dict(base_details)
                technical_details["model_name"] = model_name

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_text_model,
                    "model_used": model_used,
                    "technical_details": technical_details,
                    "show_details": False
                })
        else:
            # Fallback response when LLM not available
            if has_results:
                hotels = []
                if retrieval_result.get('merged_results'):
                    hotels = retrieval_result['merged_results'][:3]
                elif retrieval_result.get('baseline_results'):
                    hotels = retrieval_result['baseline_results'][:3]
                elif retrieval_result.get('embedding_results'):
                    hotels = retrieval_result['embedding_results'][:3]
                
                if hotels:
                    hotel_names = []
                    for h in hotels:
                        name = h.get('h.name') or h.get('hotel_name') or h.get('name', 'Unknown')
                        hotel_names.append(name)
                    response_text = f"I found {len(hotels)} hotel(s). Here are some: {', '.join(hotel_names)}"
                else:
                    response_text = "I found some results but couldn't format them properly."
            else:
                response_text = "I couldn't find relevant information in the knowledge graph. Please try rephrasing your question or check if the data contains information about your query."

            technical_details = dict(base_details)

            # Add assistant message
            st.session_state.messages.append({
                "role": "assistant",
                "content": response_text,
                "technical_details": technical_details,
                "show_details": False
            })
    
    st.rerun()

# Footer
st.markdown("---")
st.caption("Powered by Graph-RAG | Technical details available in each response")
