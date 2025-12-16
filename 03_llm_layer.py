"""
03_LLM_LAYER.PY
=====================================================================
Milestone 3: LLM Response Generation Layer

This module implements the third stage of the Graph-RAG pipeline: generating
natural language responses from retrieved Knowledge Graph context using LLMs.

Components:
  1. Context Merging: Combine baseline and embedding retrieval results
  2. Structured Prompts: Context + Persona + Task format
  3. Multi-LLM Support: OpenAI, Anthropic, HuggingFace, etc.
  4. Model Comparison: Quantitative and qualitative evaluation
=====================================================================
"""

import json
import time
import requests
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime


# =====================================================================
#  HUGGINGFACE MODELS CONFIGURATION
# =====================================================================
# Three accessible models for testing and comparison

FREE_HF_MODELS = {
    'llama-3.2-1b': {
        'model_id': 'meta-llama/Llama-3.2-1B-Instruct',
        'display_name': 'Llama 3.2 1B (Instruct)',
        'type': 'instruct',
        'description': 'Meta\'s efficient 1B parameter instruction-tuned model',
        'cost_per_1k': 0.0,  # Free via Inference API
    },
    'mistral-7b': {
        'model_id': 'mistralai/Mistral-7B-Instruct-v0.2',
        'display_name': 'Mistral 7B (Instruct)',
        'type': 'instruct',
        'description': 'Mistral AI\'s 7B instruction-tuned model',
        'cost_per_1k': 0.0,  # Free via Inference API
    },
    'gemma-2b': {
        'model_id': 'google/gemma-2-2b-it',
        'display_name': 'Gemma 2 2B (IT)',
        'type': 'instruct',
        'description': 'Google\'s lightweight 2B instruction-tuned model',
        'cost_per_1k': 0.0,  # Free via Inference API
    },
}


# =====================================================================
# SECTION 1: CONTEXT MERGER
# =====================================================================
# Combine and structure KG results from baseline and embedding retrieval

class ContextMerger:
    """Merge and structure retrieval results into unified context for LLM."""
    
    def __init__(self):
        """Initialize context merger."""
        pass
    
    def merge_results(
        self,
        retrieval_result: Dict[str, Any],
        max_items: int = 20
    ) -> Dict[str, Any]:
        """
        Merge baseline and embedding results into unified context.
        
        Args:
            retrieval_result: Dict from HybridRetriever with baseline_results,
                            embedding_results, merged_results, intent, method
            max_items: Maximum number of items to include in context
        
        Returns:
            Dict with structured context ready for LLM prompt
        """
        intent = retrieval_result.get('intent', 'hotel_search')
        baseline_results = retrieval_result.get('baseline_results', [])
        embedding_results = retrieval_result.get('embedding_results', [])
        merged_results = retrieval_result.get('merged_results', [])
        
        # Use merged_results if available, otherwise combine manually
        if merged_results:
            combined = merged_results[:max_items]
        else:
            # Combine and deduplicate
            seen = set()
            combined = []
            
            # Add baseline results first (higher priority)
            for item in baseline_results[:max_items]:
                key = self._get_item_key(item, intent)
                if key and key not in seen:
                    combined.append(item)
                    seen.add(key)
            
            # Add embedding results (fill remaining slots)
            remaining = max_items - len(combined)
            for item in embedding_results[:remaining]:
                key = self._get_item_key(item, intent)
                if key and key not in seen:
                    combined.append(item)
                    seen.add(key)
        
        return {
            'intent': intent,
            'items': combined,
            'baseline_count': len(baseline_results),
            'embedding_count': len(embedding_results),
            'total_count': len(combined),
        }
    
    def _get_item_key(self, item: Dict[str, Any], intent: str) -> Optional[str]:
        """Generate unique key for deduplication."""
        if intent == 'visa_check':
            # For visa queries, use country pair
            from_country = item.get('from_name') or item.get('from_country')
            to_country = item.get('to_name') or item.get('to_country')
            if from_country and to_country:
                return f"{from_country}->{to_country}"
        else:
            # For hotel queries, try multiple ways to get hotel_id
            # Neo4j returns keys like 'h.hotel_id' (with dots), not nested dicts
            hotel_id = (
                item.get('hotel_id') or 
                item.get('h.hotel_id') or
                (item.get('h', {}).get('hotel_id') if isinstance(item.get('h'), dict) else None)
            )
            if hotel_id:
                return str(hotel_id)
            # Fallback: use hotel name if available
            hotel_name = (
                item.get('hotel_name') or 
                item.get('h.name') or
                (item.get('h', {}).get('name') if isinstance(item.get('h'), dict) else None) or
                item.get('name')
            )
            if hotel_name:
                return f"name_{hotel_name}"
        return None
    
    def format_context(self, merged_context: Dict[str, Any]) -> str:
        """
        Format merged context into readable text for LLM prompt.
        
        Args:
            merged_context: Output from merge_results()
        
        Returns:
            Formatted context string
        """
        intent = merged_context.get('intent', 'hotel_search')
        items = merged_context.get('items', [])
        
        if not items:
            return "No relevant information found in the knowledge graph."
        
        if intent == 'visa_check' or intent == 'visa_free_destinations':
            # Format visa information
            for item in items:
                # Check if this is a visa-free destinations list (has country_name)
                if 'country_name' in item:
                    # Visa-free destinations format
                    country_name = item.get('country_name', 'Unknown')
                    visa_status = item.get('visa_status', 'No visa required')
                    lines.append(f"{country_name}: {visa_status}")
                else:
                    # Regular visa check format (from X to Y)
                    from_name = item.get('from_name') or item.get('from_country', 'Unknown')
                    to_name = item.get('to_name') or item.get('to_country', 'Unknown')
                    visa_type = item.get('v.visa_type') or item.get('visa_type', None)
                    visa_status = item.get('visa_status', 'Unknown')
                    
                    # Format as a readable sentence
                    if visa_status == 'No visa required':
                        lines.append(f"Visa Status: No visa is required for citizens of {from_name} to visit {to_name}.")
                    else:
                        if visa_type and visa_type != 'Not specified':
                            lines.append(f"Visa Status: A {visa_type} visa is required for citizens of {from_name} to visit {to_name}.")
                        else:
                            lines.append(f"Visa Status: A visa is required for citizens of {from_name} to visit {to_name}.")
        else:
            # Format hotel information
            lines = ["Hotel Information from Knowledge Graph:"]
            
            # Add scoring explanation for traveller-type queries or facility-filtered queries
            traveller_type = merged_context.get('traveller_type')
            facility = merged_context.get('facility')
            
            if traveller_type:
                scoring_explanations = {
                    'couple': "Hotels ranked by: Comfort (35%) + Location (35%) + Staff (20%) + Facilities (10%), with bonus for Concierge/Laundry",
                    'family': "Hotels ranked by: Facilities (40%) + Cleanliness (30%) + Location (20%) + Comfort (10%), with bonus for Pool/Breakfast",
                    'business': "Hotels ranked by: Location (40%) + Staff (35%) + Comfort (15%) + Cleanliness (10%), with bonus for Concierge",
                    'solo': "Hotels ranked by: Comfort (30%) + Location (25%) + Value (20%) + Facilities (15%) + Cleanliness (10%), with bonus for Gym/Laundry/Concierge"
                }
                if traveller_type in scoring_explanations:
                    lines.append(f"✓ Optimized for {traveller_type.capitalize()} Travellers")
                    lines.append(f"Ranking Criteria: {scoring_explanations[traveller_type]}")
                    lines.append("")
            
            if facility:
                lines.append(f"✓ Filtered by Facility: {facility.capitalize()}")
                lines.append("")
            
            for i, item in enumerate(items, 1):
                # Handle Neo4j dot notation keys (e.g., 'h.name', 'h.hotel_id')
                # Also handle direct keys returned from queries (e.g., 'city', 'country')
                hotel_name = (
                    item.get('hotel_name') or 
                    item.get('h.name') or
                    (item.get('h', {}).get('name') if isinstance(item.get('h'), dict) else None) or
                    item.get('name') or
                    'Unknown Hotel'
                )
                # City can be returned directly as 'city' or in dot notation
                city = (
                    item.get('city') or  # Direct key from query RETURN clause
                    item.get('h.city') or  # Dot notation
                    (item.get('h', {}).get('city') if isinstance(item.get('h'), dict) else None) or
                    'Unknown City'
                )
                # Country can be returned directly as 'country' or in dot notation
                country = (
                    item.get('country') or  # Direct key from query RETURN clause
                    item.get('h.country') or  # Dot notation
                    (item.get('h', {}).get('country') if isinstance(item.get('h'), dict) else None) or
                    'Unknown Country'
                )
                
                lines.append(f"\n{i}. {hotel_name}")
                lines.append(f"   Location: {city}, {country}")
                
                # Add ratings/scores if available (handle dot notation)
                score = (
                    item.get('average_reviews_score') or 
                    item.get('h.average_reviews_score') or
                    (item.get('h', {}).get('average_reviews_score') if isinstance(item.get('h'), dict) else None)
                )
                if score:
                    try:
                        lines.append(f"   Average Rating: {float(score):.2f}")
                    except (ValueError, TypeError):
                        lines.append(f"   Average Rating: {score}")
                
                stars = (
                    item.get('star_rating') or 
                    item.get('h.star_rating') or
                    (item.get('h', {}).get('star_rating') if isinstance(item.get('h'), dict) else None)
                )
                if stars:
                    lines.append(f"   Star Rating: {stars}")

                # Quality dimensions
                def _fmt(val):
                    try:
                        return f"{float(val):.2f}"
                    except (ValueError, TypeError):
                        return val if val is not None else None

                cleanliness = item.get('cleanliness_base') or item.get('h.cleanliness_base') or (item.get('h', {}).get('cleanliness_base') if isinstance(item.get('h'), dict) else None)
                comfort = item.get('comfort_base') or item.get('h.comfort_base') or (item.get('h', {}).get('comfort_base') if isinstance(item.get('h'), dict) else None)
                facilities = item.get('facilities_base') or item.get('h.facilities_base') or (item.get('h', {}).get('facilities_base') if isinstance(item.get('h'), dict) else None)
                location = item.get('location_base') or item.get('h.location_base') or (item.get('h', {}).get('location_base') if isinstance(item.get('h'), dict) else None)
                staff = item.get('staff_base') or item.get('h.staff_base') or (item.get('h', {}).get('staff_base') if isinstance(item.get('h'), dict) else None)
                value_money = item.get('value_for_money_base') or item.get('h.value_for_money_base') or (item.get('h', {}).get('value_for_money_base') if isinstance(item.get('h'), dict) else None)

                quality_lines = []
                if cleanliness is not None:
                    quality_lines.append(f"Cleanliness: {_fmt(cleanliness)}")
                if comfort is not None:
                    quality_lines.append(f"Comfort: {_fmt(comfort)}")
                if facilities is not None:
                    quality_lines.append(f"Facilities: {_fmt(facilities)}")
                if location is not None:
                    quality_lines.append(f"Location: {_fmt(location)}")
                if staff is not None:
                    quality_lines.append(f"Staff: {_fmt(staff)}")
                if value_money is not None:
                    quality_lines.append(f"Value for money: {_fmt(value_money)}")

                if quality_lines:
                    lines.append("   Quality Scores:")
                    for ql in quality_lines:
                        lines.append(f"     - {ql}")
                
                # Add facilities list if available (from hotel_facilities query)
                facilities_list = item.get('facilities_list') or item.get('h.facilities_list')
                relationship_facilities = item.get('relationship_facilities')
                
                if facilities_list or relationship_facilities:
                    lines.append("   Facilities:")
                    if facilities_list:
                        # facilities_list is pipe-separated string
                        fac_items = [f.strip() for f in str(facilities_list).split('|') if f.strip()]
                        for fac in fac_items:
                            lines.append(f"     - {fac.capitalize()}")
                    if relationship_facilities and isinstance(relationship_facilities, list) and relationship_facilities:
                        # Add relationship-based facilities (from HAS_FACILITY edges)
                        for fac in relationship_facilities:
                            if fac and fac not in (facilities_list or ''):
                                lines.append(f"     - {fac.capitalize()}")
                
                # Add embedding score if available
                if 'score' in item:
                    try:
                        lines.append(f"   Relevance Score: {float(item['score']):.3f}")
                    except (ValueError, TypeError):
                        lines.append(f"   Relevance Score: {item['score']}")
                
                # Add review text if available (from embedding results)
                review_text = item.get('review_text', '')
                if review_text:
                    # Truncate long review text to first 200 characters
                    if len(review_text) > 200:
                        review_text = review_text[:200] + "..."
                    lines.append(f"   Customer Reviews: {review_text}")
                
                lines.append("---")
        
        return "\n".join(lines)


# =====================================================================
# SECTION 2: STRUCTURED PROMPT BUILDER
# =====================================================================
# Build prompts with Context + Persona + Task structure

class PromptBuilder:
    """Build structured prompts with Context, Persona, and Task components."""
    
    PERSONAS = {
        'hotel': "You are a helpful travel assistant specializing in hotel recommendations and travel information.",
        'visa': "You are a knowledgeable travel document and visa requirements assistant.",
        'general': "You are a helpful travel assistant.",
    }
    
    def __init__(self, theme: str = 'hotel'):
        """
        Initialize prompt builder.
        
        Args:
            theme: Theme for persona selection ('hotel', 'visa', 'general')
        """
        self.theme = theme
        self.persona = self.PERSONAS.get(theme, self.PERSONAS['general'])
    
    def build_prompt(
        self,
        user_query: str,
        context: str,
        task_instructions: Optional[str] = None
    ) -> str:
        """
        Build structured prompt with Context, Persona, and Task.
        
        Args:
            user_query: Original user query
            context: Formatted context from ContextMerger
            task_instructions: Optional custom task instructions
        
        Returns:
            Complete prompt string
        """
        if task_instructions is None:
            task_instructions = (
                "Answer the user's question using ONLY the information provided in the context above. "
                "If the context does not contain enough information to answer the question, "
                "say so clearly. Do not make up or hallucinate information. "
                "Be concise, helpful, and accurate."
            )
        
        prompt = f"""# Persona
{self.persona}

# Context
{context}

# Task
{task_instructions}

# User Question
{user_query}

# Response
"""
        return prompt


# =====================================================================
# SECTION 3: LLM PROVIDERS
# =====================================================================
# Support for multiple LLM providers

@dataclass
class LLMResponse:
    """Structured LLM response with metadata."""
    text: str
    model_name: str
    response_time: float
    tokens_used: Optional[int] = None
    cost: Optional[float] = None
    error: Optional[str] = None


class BaseLLMProvider:
    """Base class for LLM providers."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.available = False
    
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response from prompt. Must be implemented by subclasses."""
        raise NotImplementedError


class HuggingFaceProvider(BaseLLMProvider):
    """HuggingFace Transformers provider (free, local or Inference API)."""
    
    def __init__(self, model_name: str = "gpt2", use_inference_api: bool = False, hf_token: Optional[str] = None):
        """
        Initialize HuggingFace provider.
        
        Args:
            model_name: HuggingFace model ID
            use_inference_api: If True, use HuggingFace Inference API (better models, requires token)
            hf_token: HuggingFace token (or set HF_TOKEN env var)
        """
        super().__init__(model_name)
        self.model = None
        self.tokenizer = None
        self.use_inference_api = use_inference_api
        self.hf_token = hf_token
        self.api_client = None
        
        if use_inference_api:
            # Use Inference API (better models, requires token)
            try:
                import os
                from huggingface_hub import InferenceClient
                
                token = hf_token or os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_API_TOKEN')
                if token:
                    # Initialize client with token - model specified per request
                    self.api_client = InferenceClient(token=token)
                    self.api_token = token
                    self.available = True
                    print(f"[OK] Using HuggingFace Inference API with model: {model_name}")
                else:
                    print("Warning: HuggingFace token not found. Set HF_TOKEN env var or pass hf_token parameter.")
                    print("Falling back to local model...")
                    use_inference_api = False
            except ImportError:
                print("Warning: huggingface_hub not installed. Install with: pip install huggingface_hub")
                print("Falling back to local model...")
                use_inference_api = False
            except Exception as e:
                print(f"Warning: Inference API initialization failed: {e}")
                use_inference_api = False
        
        if not use_inference_api:
            # Use local transformers
            try:
                from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
                import torch
                import os
                
                token = hf_token or os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_API_TOKEN')
                
                # Use smaller, faster models for local execution
                if "gpt2" in model_name.lower() or "dialo" in model_name.lower():
                    self.generator = pipeline(
                        "text-generation", 
                        model=model_name, 
                        device=-1,  # CPU
                        token=token  # For gated models
                    )
                    self.available = True
                else:
                    # For larger models, try to load with limited resources
                    try:
                        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
                        self.model = AutoModelForCausalLM.from_pretrained(
                            model_name,
                            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                            device_map="auto" if torch.cuda.is_available() else None,
                            token=token
                        )
                        self.available = True
                    except Exception as e:
                        print(f"Warning: Could not load {model_name}: {e}")
                        print("Falling back to GPT-2")
                        self.generator = pipeline("text-generation", model="gpt2", device=-1)
                        self.model_name = "gpt2"
                        self.available = True
            except ImportError:
                print("Warning: transformers not installed. Install with: pip install transformers torch")
            except Exception as e:
                print(f"Warning: HuggingFace provider initialization failed: {e}")
    
    def generate(self, prompt: str, max_length: int = 200, **kwargs) -> LLMResponse:
        """Generate response using HuggingFace model."""
        if not self.available:
            return LLMResponse(
                text="HuggingFace provider not available. Install transformers: pip install transformers",
                model_name=self.model_name,
                response_time=0.0,
                error="Provider not available"
            )
        
        start_time = time.time()
        try:
            if self.use_inference_api and self.api_client:
                # Use Inference API (better quality, faster)
                # Check if model is instruction-tuned (needs conversational format)
                is_instruct_model = any(x in self.model_name.lower() for x in [
                    "instruct", "chat", "llama", "mistral", "gemma"
                ])
                
                if is_instruct_model:
                    # For instruction-tuned models, use InferenceClient (handles routing automatically)
                    response_text = None
                    
                    # Format prompt based on model type
                    if "gemma" in self.model_name.lower():
                        # Use InferenceClient directly (as user confirmed this works)
                        # Match the working example: client.chat_completion()
                        try:
                            from huggingface_hub import InferenceClient
                            # Create client with model and token (handles routing automatically)
                            client = InferenceClient(model=self.model_name, token=self.api_token)
                            
                            # Use chat_completion method (as shown in working example)
                            response = client.chat_completion(
                                messages=[{"role": "user", "content": prompt}],
                                max_tokens=max_length,
                                temperature=0.7
                            )
                            
                            # Extract response (as in working example)
                            response_text = response.choices[0].message["content"]
                                
                        except (AttributeError, Exception) as e1:
                            # If chat.completion doesn't exist or fails, use conversational API format
                            try:
                                # Use router endpoint (new, required) with conversational API format
                                api_url = f"https://router.huggingface.co/models/{self.model_name}"
                                headers = {
                                    "Authorization": f"Bearer {self.api_token}",
                                    "Content-Type": "application/json"
                                }
                                
                                # Conversational API format (required for Gemma)
                                payload = {
                                    "inputs": {
                                        "past_user_inputs": [],
                                        "generated_responses": [],
                                        "text": prompt
                                    },
                                    "parameters": {
                                        "max_new_tokens": max_length,
                                        "temperature": 0.7,
                                        "return_full_text": False
                                    }
                                }
                                
                                response = requests.post(api_url, headers=headers, json=payload, timeout=60)
                                
                                if response.status_code == 200:
                                    result = response.json()
                                    # Conversational API returns conversation object
                                    if isinstance(result, dict):
                                        if "generated_text" in result:
                                            response_text = result["generated_text"]
                                        elif "conversation" in result:
                                            conv = result["conversation"]
                                            responses = conv.get("generated_responses", [])
                                            response_text = responses[-1] if responses else ""
                                        else:
                                            response_text = str(result)
                                    elif isinstance(result, list) and len(result) > 0:
                                        response_text = result[0].get("generated_text", "") if isinstance(result[0], dict) else str(result[0])
                                    else:
                                        response_text = str(result)
                                elif response.status_code == 503:
                                    # Model loading - wait and retry
                                    time.sleep(10)
                                    response = requests.post(api_url, headers=headers, json=payload, timeout=60)
                                    if response.status_code == 200:
                                        result = response.json()
                                        if isinstance(result, dict) and "conversation" in result:
                                            response_text = result["conversation"].get("generated_responses", [""])[-1]
                                        elif isinstance(result, dict):
                                            response_text = result.get("generated_text", str(result))
                                        else:
                                            response_text = str(result)
                                    else:
                                        raise Exception(f"Model loading timeout: {response.status_code}")
                                else:
                                    raise Exception(f"API returned {response.status_code}: {response.text[:200]}")
                                    
                            except Exception as e2:
                                raise Exception(
                                    f"Failed to generate with {self.model_name}. "
                                    f"Tried chat completion ({str(e1)[:100]}) and conversational API ({str(e2)[:100]})."
                                )
                    else:
                        # For other instruction models (Mistral, Llama), use chat_completion like Gemma
                        try:
                            from huggingface_hub import InferenceClient
                            # Create client with model and token
                            client = InferenceClient(model=self.model_name, token=self.api_token)
                            
                            # Use chat_completion for all instruction models
                            response = client.chat_completion(
                                messages=[{"role": "user", "content": prompt}],
                                max_tokens=max_length,
                                temperature=0.7
                            )
                            
                            # Extract response
                            response_text = response.choices[0].message["content"]
                                
                        except Exception as e:
                            raise Exception(f"Failed to generate with {self.model_name}: {str(e)[:200]}")
                    
                    if not response_text or len(response_text.strip()) < 10:
                        raise Exception(f"Model {self.model_name} returned empty or very short response.")
                else:
                    # Use standard text generation for non-instruct models
                    response_text = self.api_client.text_generation(
                        prompt,
                        model=self.model_name,
                        max_new_tokens=max_length,
                        temperature=0.7,
                        return_full_text=False  # Don't include prompt in response
                    )
                
                response_time = time.time() - start_time
                tokens_used = len(response_text.split())
                
                return LLMResponse(
                    text=response_text,
                    model_name=self.model_name,
                    response_time=response_time,
                    tokens_used=tokens_used,
                    cost=0.0  # Free Inference API
                )
            elif hasattr(self, 'generator'):
                # Use local pipeline
                # For GPT-2, we need to be more careful with prompt length and add stop sequences
                prompt_words = prompt.split()
                max_prompt_length = min(400, len(prompt_words))  # Limit prompt length
                truncated_prompt = " ".join(prompt_words[-max_prompt_length:]) if len(prompt_words) > max_prompt_length else prompt
                
                # Add a clear instruction to stop at reasonable points
                stop_sequences = ["\n\n\n", "User:", "Question:", "Context:", "Persona:", "Task:"]
                
                result = self.generator(
                    truncated_prompt,
                    max_length=min(len(truncated_prompt.split()) + max_length, 1024),  # GPT-2 has 1024 token limit
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.generator.tokenizer.eos_token_id,
                    truncation=True,
                    repetition_penalty=1.2  # Reduce repetition
                )
                generated_text = result[0]['generated_text']
                # Extract only the new part (after prompt)
                if truncated_prompt in generated_text:
                    response_text = generated_text[len(truncated_prompt):].strip()
                else:
                    # If prompt not found, take the last part
                    response_text = generated_text.strip()
                
                # Clean up response - remove repetitive patterns and stop at reasonable points
                if response_text:
                    # Remove excessive repetition
                    lines = response_text.split('\n')
                    seen = set()
                    cleaned_lines = []
                    for line in lines[:30]:  # Limit to 30 lines max
                        line_stripped = line.strip()
                        # Stop if we hit a stop sequence
                        if any(stop in line_stripped for stop in stop_sequences):
                            break
                        if line_stripped and line_stripped not in seen:
                            cleaned_lines.append(line)
                            seen.add(line_stripped)
                        # Stop if we see too much repetition
                        if len(seen) > 20 and len(cleaned_lines) > 15:
                            break
                    response_text = '\n'.join(cleaned_lines)
                    
                    # Additional cleanup: remove common GPT-2 artifacts
                    response_text = response_text.replace("Trip Advisor", "").strip()
                    response_text = response_text.replace("Yelp", "").strip()
                    # Remove lines that are just URLs or navigation
                    final_lines = []
                    for line in response_text.split('\n'):
                        line_clean = line.strip()
                        if line_clean and not line_clean.startswith('http') and len(line_clean) > 10:
                            final_lines.append(line)
                    response_text = '\n'.join(final_lines[:15])  # Limit to 15 meaningful lines
            else:
                # Use model directly
                inputs = self.tokenizer(prompt, return_tensors="pt")
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=max_length,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                response_text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            response_time = time.time() - start_time
            tokens_used = len(response_text.split())
            
            return LLMResponse(
                text=response_text,
                model_name=self.model_name,
                response_time=response_time,
                tokens_used=tokens_used,
                cost=0.0  # Free
            )
        except Exception as e:
            return LLMResponse(
                text=f"Error generating response: {str(e)}",
                model_name=self.model_name,
                response_time=time.time() - start_time,
                error=str(e)
            )


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API provider (requires API key)."""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", api_key: Optional[str] = None):
        """
        Initialize OpenAI provider.
        
        Args:
            model_name: OpenAI model name (gpt-3.5-turbo, gpt-4, etc.)
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
        """
        super().__init__(model_name)
        self.api_key = api_key
        self.client = None
        
        try:
            import os
            from openai import OpenAI
            
            api_key = api_key or os.getenv('OPENAI_API_KEY')
            if api_key:
                self.client = OpenAI(api_key=api_key)
                self.available = True
            else:
                print("Warning: OpenAI API key not found. Set OPENAI_API_KEY env var or pass api_key parameter.")
        except ImportError:
            print("Warning: openai package not installed. Install with: pip install openai")
        except Exception as e:
            print(f"Warning: OpenAI provider initialization failed: {e}")
    
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response using OpenAI API."""
        if not self.available or not self.client:
            return LLMResponse(
                text="OpenAI provider not available. Set OPENAI_API_KEY environment variable.",
                model_name=self.model_name,
                response_time=0.0,
                error="Provider not available"
            )
        
        start_time = time.time()
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=kwargs.get('max_tokens', 500)
            )
            
            response_time = time.time() - start_time
            response_text = response.choices[0].message.content
            tokens_used = response.usage.total_tokens if hasattr(response, 'usage') else None
            
            # Estimate cost (rough estimates)
            cost = None
            if tokens_used:
                if "gpt-4" in self.model_name.lower():
                    cost = (tokens_used / 1000) * 0.03  # $0.03 per 1K tokens
                elif "gpt-3.5" in self.model_name.lower():
                    cost = (tokens_used / 1000) * 0.002  # $0.002 per 1K tokens
            
            return LLMResponse(
                text=response_text,
                model_name=self.model_name,
                response_time=response_time,
                tokens_used=tokens_used,
                cost=cost
            )
        except Exception as e:
            return LLMResponse(
                text=f"Error generating response: {str(e)}",
                model_name=self.model_name,
                response_time=time.time() - start_time,
                error=str(e)
            )


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude API provider (requires API key)."""
    
    def __init__(self, model_name: str = "claude-3-haiku-20240307", api_key: Optional[str] = None):
        """
        Initialize Anthropic provider.
        
        Args:
            model_name: Claude model name
            api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var)
        """
        super().__init__(model_name)
        self.api_key = api_key
        self.client = None
        
        try:
            import os
            from anthropic import Anthropic
            
            api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
            if api_key:
                self.client = Anthropic(api_key=api_key)
                self.available = True
            else:
                print("Warning: Anthropic API key not found. Set ANTHROPIC_API_KEY env var.")
        except ImportError:
            print("Warning: anthropic package not installed. Install with: pip install anthropic")
        except Exception as e:
            print(f"Warning: Anthropic provider initialization failed: {e}")
    
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response using Anthropic API."""
        if not self.available or not self.client:
            return LLMResponse(
                text="Anthropic provider not available. Set ANTHROPIC_API_KEY environment variable.",
                model_name=self.model_name,
                response_time=0.0,
                error="Provider not available"
            )
        
        start_time = time.time()
        try:
            message = self.client.messages.create(
                model=self.model_name,
                max_tokens=kwargs.get('max_tokens', 500),
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            response_time = time.time() - start_time
            response_text = message.content[0].text
            tokens_used = message.usage.input_tokens + message.usage.output_tokens if hasattr(message, 'usage') else None
            
            # Estimate cost (rough estimates for Claude)
            cost = None
            if tokens_used:
                if "claude-3-opus" in self.model_name.lower():
                    cost = (tokens_used / 1000) * 0.015  # $0.015 per 1K tokens
                elif "claude-3-sonnet" in self.model_name.lower():
                    cost = (tokens_used / 1000) * 0.003  # $0.003 per 1K tokens
                elif "claude-3-haiku" in self.model_name.lower():
                    cost = (tokens_used / 1000) * 0.00025  # $0.00025 per 1K tokens
            
            return LLMResponse(
                text=response_text,
                model_name=self.model_name,
                response_time=response_time,
                tokens_used=tokens_used,
                cost=cost
            )
        except Exception as e:
            return LLMResponse(
                text=f"Error generating response: {str(e)}",
                model_name=self.model_name,
                response_time=time.time() - start_time,
                error=str(e)
            )


# =====================================================================
# SECTION 4: LLM ORCHESTRATOR
# =====================================================================
# Main class that combines everything

class LLMOrchestrator:
    """Orchestrate context merging, prompt building, and LLM generation."""
    
    def __init__(self, theme: str = 'hotel'):
        """
        Initialize LLM orchestrator.
        
        Args:
            theme: Theme for persona ('hotel', 'visa', 'general')
        """
        self.context_merger = ContextMerger()
        self.prompt_builder = PromptBuilder(theme=theme)
        self.providers: Dict[str, BaseLLMProvider] = {}
    
    def add_provider(self, provider: BaseLLMProvider):
        """Add an LLM provider."""
        self.providers[provider.model_name] = provider
    
    def generate_response(
        self,
        user_query: str,
        retrieval_result: Dict[str, Any],
        model_name: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate LLM response from retrieval results.
        
        Args:
            user_query: Original user query
            retrieval_result: Output from HybridRetriever
            model_name: Name of model to use (uses first available if None)
            **kwargs: Additional arguments for LLM generation
        
        Returns:
            LLMResponse with generated text and metadata
        """
        #  Merge baseline + embedding results
        merged_context = self.context_merger.merge_results(retrieval_result)
				#Format into readable text
        formatted_context = self.context_merger.format_context(merged_context)
        
        # Build structured prompt (Persona + Context + Task)
        prompt = self.prompt_builder.build_prompt(user_query, formatted_context)
        
        # Select provider
        if model_name and model_name in self.providers:
            provider = self.providers[model_name]
        elif self.providers:
            provider = list(self.providers.values())[0]
        else:
            return LLMResponse(
                text="No LLM providers available. Add providers using add_provider().",
                model_name="none",
                response_time=0.0,
                error="No providers"
            )
        
        # Generate response,    # STEP 4: Send to LLM and get response
        return provider.generate(prompt, **kwargs)
    
    def compare_models(
        self,
        user_query: str,
        retrieval_result: Dict[str, Any],
        model_names: Optional[List[str]] = None
    ) -> Dict[str, LLMResponse]:
        """
        Compare multiple models on the same query.
        
        Args:
            user_query: Original user query
            retrieval_result: Output from HybridRetriever
            model_names: List of model names to compare (uses all if None)
        
        Returns:
            Dict mapping model_name to LLMResponse
        """
        if model_names is None:
            model_names = list(self.providers.keys())
        
        results = {}
        for model_name in model_names:
            if model_name in self.providers:
                results[model_name] = self.generate_response(
                    user_query,
                    retrieval_result,
                    model_name=model_name
                )
        
        return results


# =====================================================================
# SECTION 5: MODEL COMPARISON & EVALUATION
# =====================================================================

@dataclass
class ModelComparison:
    """Results from comparing multiple models."""
    model_name: str
    response: LLMResponse
    accuracy_score: Optional[float] = None
    relevance_score: Optional[float] = None
    naturalness_score: Optional[float] = None
    correctness_score: Optional[float] = None
    grounding_score: Optional[float] = None
    notes: Optional[str] = None


class QualitativeEvaluator:
    """Evaluate LLM responses with detailed qualitative rubric."""
    
    RUBRIC = {
        'accuracy': {
            'description': 'Response contains factually correct information from context',
            'scores': {
                5: 'All information accurate, no hallucinations',
                4: 'Mostly accurate with minor irrelevant details',
                3: 'Partially accurate, some errors',
                2: 'Many inaccuracies or contradictions',
                1: 'Completely inaccurate or made-up information'
            }
        },
        'relevance': {
            'description': 'Response directly addresses the user query',
            'scores': {
                5: 'Directly answers query with perfect relevance',
                4: 'Mostly relevant with minor tangents',
                3: 'Somewhat relevant but includes off-topic content',
                2: 'Barely addresses the query',
                1: 'Completely irrelevant response'
            }
        },
        'naturalness': {
            'description': 'Response is fluent, coherent, and human-like',
            'scores': {
                5: 'Perfect fluency, natural conversation',
                4: 'Good fluency with minor awkwardness',
                3: 'Understandable but somewhat robotic',
                2: 'Awkward phrasing, hard to follow',
                1: 'Incoherent or nonsensical'
            }
        },
        'correctness': {
            'description': 'Response correctly interprets and uses context',
            'scores': {
                5: 'Perfect use of context, no misinterpretation',
                4: 'Good use with minor omissions',
                3: 'Uses context but misses key details',
                2: 'Misinterprets context significantly',
                1: 'Ignores or contradicts context'
            }
        },
        'grounding': {
            'description': 'Response stays grounded in provided context',
            'scores': {
                5: 'All claims backed by context, no hallucination',
                4: 'Mostly grounded with minimal speculation',
                3: 'Some ungrounded claims',
                2: 'Significant hallucination',
                1: 'Completely fabricated information'
            }
        }
    }
    
    def __init__(self):
        """Initialize qualitative evaluator."""
        pass
    
    def evaluate(self, response_text: str, context: str, query: str) -> Dict[str, float]:
        """Evaluate response using automated heuristics.
        
        Args:
            response_text: Generated response
            context: Original context provided
            query: User query
            
        Returns:
            Dict with scores for each rubric dimension
        """
        scores = {}
        
        # Accuracy: Check for context word overlap
        context_words = set(context.lower().split())
        response_words = set(response_text.lower().split())
        overlap = len(context_words.intersection(response_words))
        overlap_ratio = overlap / max(len(context_words), 1)
        scores['accuracy'] = min(5, max(1, int(overlap_ratio * 5) + 1))
        
        # Relevance: Check query word presence
        query_words = set(query.lower().split()) - {'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for'}
        query_in_response = len(query_words.intersection(response_words)) / max(len(query_words), 1)
        scores['relevance'] = min(5, max(1, int(query_in_response * 5) + 1))
        
        # Naturalness: Basic heuristics
        has_complete_sentences = response_text.count('.') > 0 or response_text.count('!') > 0
        reasonable_length = 20 < len(response_text.split()) < 500
        no_excessive_repetition = len(set(response_text.split())) / max(len(response_text.split()), 1) > 0.5
        naturalness_score = sum([has_complete_sentences, reasonable_length, no_excessive_repetition]) + 2
        scores['naturalness'] = min(5, max(1, naturalness_score))
        
        # Correctness: Check for specific context entities
        context_entities = [w for w in context.split() if len(w) > 3 and w[0].isupper()]
        entities_mentioned = sum(1 for e in context_entities if e in response_text)
        entity_ratio = entities_mentioned / max(len(context_entities), 1)
        scores['correctness'] = min(5, max(1, int(entity_ratio * 5) + 1))
        
        # Grounding: Check for hallucination indicators
        hallucination_phrases = ['i think', 'probably', 'might be', 'maybe', 'i\'m not sure']
        has_uncertainty = any(phrase in response_text.lower() for phrase in hallucination_phrases)
        has_specific_numbers = bool(context_words.intersection(set([str(i) for i in range(10)])))
        numbers_match = has_specific_numbers and any(str(i) in response_text for i in range(10))
        grounding_score = 5 - (2 if has_uncertainty else 0) + (1 if numbers_match else 0)
        scores['grounding'] = min(5, max(1, grounding_score))
        
        return scores
    
    def manual_evaluate(self, response_text: str, context: str, query: str) -> Dict[str, float]:
        """Placeholder for manual evaluation (returns automated scores).
        
        In a real system, this would prompt a human evaluator.
        For now, returns automated scores.
        """
        return self.evaluate(response_text, context, query)
    
    def get_rubric_description(self, dimension: str) -> str:
        """Get human-readable rubric description."""
        if dimension in self.RUBRIC:
            rubric = self.RUBRIC[dimension]
            lines = [f"{dimension.upper()}: {rubric['description']}", ""]
            for score, desc in sorted(rubric['scores'].items(), reverse=True):
                lines.append(f"  {score}: {desc}")
            return "\n".join(lines)
        return "Unknown dimension"
    
    def calculate_overall_score(self, scores: Dict[str, float]) -> float:
        """Calculate weighted overall score.
        
        Weights: accuracy=0.3, relevance=0.25, correctness=0.25, grounding=0.15, naturalness=0.05
        """
        weights = {
            'accuracy': 0.3,
            'relevance': 0.25,
            'correctness': 0.25,
            'grounding': 0.15,
            'naturalness': 0.05
        }
        total = sum(scores.get(dim, 3) * weight for dim, weight in weights.items())
        return round(total, 2)


class TestCase:
    """Test case with query, context, and expected characteristics."""
    
    def __init__(self, query: str, context: str, expected_keywords: List[str], 
                 intent: str = 'hotel_search', difficulty: str = 'medium'):
        self.query = query
        self.context = context
        self.expected_keywords = expected_keywords
        self.intent = intent
        self.difficulty = difficulty


class TestSuite:
    """Collection of test cases for systematic evaluation."""
    
    STANDARD_TESTS = [
        TestCase(
            query="Find hotels in Cairo with rating > 4",
            context="Hotel Information from Knowledge Graph:\n\n1. Nile Grandeur\n   Location: Cairo, Egypt\n   Average Rating: 4.50\n   Star Rating: 5\n   Quality Scores:\n     - Cleanliness: 8.50\n     - Comfort: 8.20\n---",
            expected_keywords=['Nile Grandeur', 'Cairo', '4.5', 'rating'],
            intent='hotel_search',
            difficulty='easy'
        ),
        TestCase(
            query="Which hotels in Paris have excellent cleanliness and staff service?",
            context="Hotel Information from Knowledge Graph:\n\n1. Le Parisien Luxe\n   Location: Paris, France\n   Average Rating: 4.70\n   Quality Scores:\n     - Cleanliness: 9.20\n     - Staff: 9.10\n---\n\n2. Budget Inn Paris\n   Location: Paris, France\n   Average Rating: 3.20\n   Quality Scores:\n     - Cleanliness: 6.50\n     - Staff: 6.80\n---",
            expected_keywords=['Le Parisien', 'cleanliness', 'staff', '9.2', '9.1'],
            intent='hotel_filter',
            difficulty='medium'
        ),
        TestCase(
            query="What visa requirements are there from Egypt to France?",
            context="Visa Requirements Information:\n\nFrom: Egypt\nTo: France\nVisa Status: Visa required\nVisa Type: Schengen\n---",
            expected_keywords=['visa required', 'Schengen', 'Egypt', 'France'],
            intent='visa_check',
            difficulty='easy'
        ),
        TestCase(
            query="Recommend family-friendly hotels in Dubai with good facilities",
            context="Hotel Information from Knowledge Graph:\n\n1. Dubai Family Resort\n   Location: Dubai, UAE\n   Average Rating: 4.60\n   Quality Scores:\n     - Facilities: 9.00\n     - Comfort: 8.80\n---\n\n2. Business Tower Dubai\n   Location: Dubai, UAE\n   Average Rating: 4.40\n   Quality Scores:\n     - Facilities: 7.50\n     - Staff: 8.20\n---",
            expected_keywords=['Dubai Family Resort', 'facilities', '9.0', 'family'],
            intent='recommendation',
            difficulty='medium'
        ),
        TestCase(
            query="Compare hotels in Tokyo and Osaka for business travelers",
            context="Hotel Information from Knowledge Graph:\n\n1. Tokyo Business Center\n   Location: Tokyo, Japan\n   Average Rating: 4.55\n   Quality Scores:\n     - Location: 9.50\n     - Facilities: 8.70\n---\n\n2. Osaka Executive\n   Location: Osaka, Japan\n   Average Rating: 4.45\n   Quality Scores:\n     - Location: 8.90\n     - Facilities: 8.50\n---",
            expected_keywords=['Tokyo', 'Osaka', 'business', 'location', 'facilities'],
            intent='comparison',
            difficulty='hard'
        ),
    ]
    
    def __init__(self, custom_tests: Optional[List[TestCase]] = None):
        """Initialize test suite.
        
        Args:
            custom_tests: Optional custom test cases (uses STANDARD_TESTS if None)
        """
        self.tests = custom_tests if custom_tests else self.STANDARD_TESTS
    
    def get_test(self, index: int) -> Optional[TestCase]:
        """Get test case by index."""
        return self.tests[index] if 0 <= index < len(self.tests) else None
    
    def get_tests_by_difficulty(self, difficulty: str) -> List[TestCase]:
        """Get all tests of a specific difficulty."""
        return [t for t in self.tests if t.difficulty == difficulty]
    
    def get_tests_by_intent(self, intent: str) -> List[TestCase]:
        """Get all tests for a specific intent."""
        return [t for t in self.tests if t.intent == intent]


class ModelEvaluator:
    """Evaluate and compare LLM models with test suite support."""
    
    def __init__(self, test_suite: Optional[TestSuite] = None):
        """Initialize evaluator.
        
        Args:
            test_suite: Optional test suite (creates default if None)
        """
        self.test_suite = test_suite if test_suite else TestSuite()
        self.qual_evaluator = QualitativeEvaluator()
    
    def evaluate_response(
        self,
        response: LLMResponse,
        context: str,
        user_query: str,
        include_qualitative: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate a single response with quantitative and qualitative metrics.
        
        Args:
            response: LLMResponse to evaluate
            context: Original context used
            user_query: Original user query
            include_qualitative: Whether to include qualitative scoring
        
        Returns:
            Dict with comprehensive evaluation metrics
        """
        metrics = {
            'model_name': response.model_name,
            'response_time': response.response_time,
            'tokens_used': response.tokens_used,
            'cost': response.cost,
            'response_length': len(response.text),
            'has_error': response.error is not None,
        }
        
        # Error analysis
        if response.error:
            metrics['error_type'] = self._classify_error(response.error)
            metrics['error_severity'] = self._assess_error_severity(response.error)
            metrics['error_message'] = response.error
        
        # Calculate quantitative quality metrics
        if response.text and not response.error:
            # Check if response mentions context keywords
            context_words = set(context.lower().split())
            response_words = set(response.text.lower().split())
            overlap = len(context_words.intersection(response_words))
            metrics['context_overlap'] = overlap / max(len(context_words), 1)
            
            # Check if response is too short (might be incomplete)
            metrics['is_too_short'] = len(response.text.split()) < 10
            
            # Check for hallucination indicators
            hallucination_markers = ['i think', 'probably', 'might be', 'maybe', 'not sure']
            metrics['hallucination_risk'] = sum(1 for m in hallucination_markers if m in response.text.lower())
            
            # Qualitative evaluation
            if include_qualitative:
                qual_scores = self.qual_evaluator.evaluate(response.text, context, user_query)
                metrics['qualitative_scores'] = qual_scores
                metrics['overall_quality'] = self.qual_evaluator.calculate_overall_score(qual_scores)
        
        return metrics
    
    def _classify_error(self, error_msg: str) -> str:
        """Classify error type for analysis."""
        error_lower = error_msg.lower()
        if 'timeout' in error_lower or 'timed out' in error_lower:
            return 'timeout'
        elif 'api' in error_lower or 'key' in error_lower:
            return 'authentication'
        elif 'rate limit' in error_lower or 'quota' in error_lower:
            return 'rate_limit'
        elif 'not available' in error_lower or 'not found' in error_lower:
            return 'availability'
        elif 'token' in error_lower or 'length' in error_lower:
            return 'token_limit'
        else:
            return 'unknown'
    
    def _assess_error_severity(self, error_msg: str) -> str:
        """Assess error severity."""
        error_type = self._classify_error(error_msg)
        if error_type in ['authentication', 'availability']:
            return 'critical'
        elif error_type in ['rate_limit', 'timeout']:
            return 'moderate'
        else:
            return 'low'
    
    def compare_responses(
        self,
        responses: Dict[str, LLMResponse],
        context: str,
        user_query: str,
        include_qualitative: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare multiple responses with comprehensive metrics.
        
        Args:
            responses: Dict mapping model_name to LLMResponse
            context: Original context used
            user_query: Original user query
            include_qualitative: Whether to include qualitative evaluation
        
        Returns:
            Dict mapping model_name to evaluation metrics
        """
        comparisons = {}
        
        for model_name, response in responses.items():
            metrics = self.evaluate_response(response, context, user_query, include_qualitative)
            comparisons[model_name] = metrics
        
        return comparisons
    
    def run_test_suite(
        self,
        orchestrator: 'LLMOrchestrator',
        model_names: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Run all test cases and aggregate results.
        
        Args:
            orchestrator: LLMOrchestrator with providers
            model_names: Models to test (uses all if None)
            
        Returns:
            Dict with aggregated results per model
        """
        if model_names is None:
            model_names = list(orchestrator.providers.keys())
        
        results = {model: {'tests': [], 'aggregate': {}} for model in model_names}
        
        for i, test in enumerate(self.test_suite.tests):
            print(f"Running test {i+1}/{len(self.test_suite.tests)}: {test.query[:50]}...")
            
            # Create mock retrieval result
            mock_retrieval = {
                'intent': test.intent,
                'method': 'baseline',
                'baseline_results': [],
                'embedding_results': [],
                'merged_results': []
            }
            
            # Generate responses from all models
            for model_name in model_names:
                if model_name not in orchestrator.providers:
                    continue
                
                response = orchestrator.generate_response(
                    test.query,
                    mock_retrieval,
                    model_name=model_name
                )
                
                # Evaluate response
                metrics = self.evaluate_response(response, test.context, test.query)
                metrics['test_index'] = i
                metrics['difficulty'] = test.difficulty
                metrics['expected_keywords_found'] = sum(
                    1 for kw in test.expected_keywords 
                    if kw.lower() in response.text.lower()
                ) if not response.error else 0
                
                results[model_name]['tests'].append(metrics)
        
        # Aggregate statistics
        for model_name in model_names:
            tests = results[model_name]['tests']
            if tests:
                results[model_name]['aggregate'] = {
                    'avg_response_time': sum(t['response_time'] for t in tests) / len(tests),
                    'avg_tokens': sum(t.get('tokens_used', 0) for t in tests if t.get('tokens_used')) / max(1, len([t for t in tests if t.get('tokens_used')])),
                    'total_cost': sum(t.get('cost', 0) for t in tests if t.get('cost')),
                    'error_rate': sum(1 for t in tests if t['has_error']) / len(tests),
                    'avg_quality': sum(t.get('overall_quality', 0) for t in tests if t.get('overall_quality')) / max(1, len([t for t in tests if t.get('overall_quality')])),
                }
        
        return results
    
    def generate_comparison_report(
        self,
        comparisons: Dict[str, Dict[str, Any]],
        responses: Dict[str, LLMResponse],
        include_cost_analysis: bool = True,
        include_qualitative: bool = True
    ) -> str:
        """
        Generate comprehensive comparison report with cost analysis.
        
        Args:
            comparisons: Output from compare_responses()
            responses: Dict mapping model_name to LLMResponse
            include_cost_analysis: Whether to include detailed cost breakdown
            include_qualitative: Whether to include qualitative scores
        
        Returns:
            Formatted comparison report string
        """
        lines = ["=" * 100]
        lines.append("LLM MODEL COMPARISON REPORT")
        lines.append("=" * 100)
        lines.append("")
        
        # Quantitative metrics table
        lines.append("QUANTITATIVE METRICS:")
        lines.append("-" * 100)
        lines.append(f"{'Model':<30} {'Time(s)':<10} {'Tokens':<10} {'Cost($)':<12} {'Length':<10} {'Errors':<10}")
        lines.append("-" * 100)
        
        for model_name, metrics in comparisons.items():
            time_str = f"{metrics.get('response_time', 0):.2f}"
            tokens_str = str(metrics.get('tokens_used', 'N/A'))
            cost_str = f"${metrics.get('cost', 0):.4f}" if metrics.get('cost') else "Free"
            length_str = str(metrics.get('response_length', 0))
            error_str = "Yes" if metrics.get('has_error') else "No"
            
            lines.append(f"{model_name:<30} {time_str:<10} {tokens_str:<10} {cost_str:<12} {length_str:<10} {error_str:<10}")
        
        # Cost comparison analysis
        if include_cost_analysis:
            lines.append("")
            lines.append("COST ANALYSIS:")
            lines.append("-" * 100)
            
            # Calculate cost statistics
            paid_models = {k: v for k, v in comparisons.items() if v.get('cost', 0) > 0}
            free_models = {k: v for k, v in comparisons.items() if v.get('cost', 0) == 0 and not v.get('has_error')}
            
            if paid_models:
                lines.append("\nPaid Models:")
                for model_name, metrics in sorted(paid_models.items(), key=lambda x: x[1].get('cost', 0), reverse=True):
                    cost = metrics.get('cost', 0)
                    tokens = metrics.get('tokens_used', 0)
                    cost_per_token = (cost / tokens * 1000) if tokens else 0
                    lines.append(f"  {model_name}: ${cost:.4f} (${cost_per_token:.6f} per 1K tokens)")
            
            if free_models:
                lines.append("\nFree Models:")
                for model_name in free_models.keys():
                    lines.append(f"  {model_name}: Free (Inference API)")
            
            # Cost savings calculation
            if paid_models and free_models:
                avg_paid_cost = sum(m.get('cost', 0) for m in paid_models.values()) / len(paid_models)
                lines.append(f"\n💰 Cost Savings: Using free models saves ~${avg_paid_cost:.4f} per query")
        
        # Qualitative scores table
        if include_qualitative:
            lines.append("")
            lines.append("QUALITATIVE SCORES (1-5 scale):")
            lines.append("-" * 100)
            lines.append(f"{'Model':<30} {'Accuracy':<12} {'Relevance':<12} {'Correct':<12} {'Ground':<12} {'Overall':<12}")
            lines.append("-" * 100)
            
            for model_name, metrics in comparisons.items():
                if 'qualitative_scores' in metrics:
                    scores = metrics['qualitative_scores']
                    acc = f"{scores.get('accuracy', 0):.1f}"
                    rel = f"{scores.get('relevance', 0):.1f}"
                    cor = f"{scores.get('correctness', 0):.1f}"
                    grd = f"{scores.get('grounding', 0):.1f}"
                    ovr = f"{metrics.get('overall_quality', 0):.2f}"
                    lines.append(f"{model_name:<30} {acc:<12} {rel:<12} {cor:<12} {grd:<12} {ovr:<12}")
                else:
                    lines.append(f"{model_name:<30} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12}")
        
        # Error analysis
        errors = {k: v for k, v in comparisons.items() if v.get('has_error')}
        if errors:
            lines.append("")
            lines.append("ERROR ANALYSIS:")
            lines.append("-" * 100)
            for model_name, metrics in errors.items():
                error_type = metrics.get('error_type', 'unknown')
                severity = metrics.get('error_severity', 'unknown')
                lines.append(f"  {model_name}: {error_type} (severity: {severity})")
                lines.append(f"    Message: {metrics.get('error_message', 'N/A')[:80]}...")
        
        # Response samples
        lines.append("")
        lines.append("RESPONSE SAMPLES:")
        lines.append("-" * 100)
        
        for model_name, response in responses.items():
            lines.append(f"\n{model_name}:")
            if response.error:
                lines.append(f"  ❌ Error: {response.error[:100]}...")
            else:
                preview = response.text[:200] if len(response.text) > 200 else response.text
                lines.append(f"  {preview}..." if len(response.text) > 200 else f"  {preview}")
        
        lines.append("")
        lines.append("=" * 100)
        
        return "\n".join(lines)


# =====================================================================
# EXAMPLE USAGE & TESTING
# =====================================================================

if __name__ == '__main__':
    print("=" * 80)
    print("LLM LAYER DEMO")
    print("=" * 80)
    
    # Mock retrieval result for testing
    mock_retrieval = {
        'intent': 'hotel_search',
        'method': 'hybrid',
        'baseline_results': [
            {
                'hotel_id': 1,
                'hotel_name': 'Nile Grandeur',
                'city': 'Cairo',
                'country': 'Egypt',
                'average_reviews_score': 4.5,
                'star_rating': 5
            }
        ],
        'embedding_results': [
            {
                'hotel_id': 2,
                'hotel_name': 'The Azure Tower',
                'city': 'New York',
                'country': 'USA',
                'score': 0.85
            }
        ],
        'merged_results': []
    }
    
    # Initialize orchestrator
    orchestrator = LLMOrchestrator(theme='hotel')
    
    # Add providers (using free models for demo)
    print("\nInitializing LLM providers...")
    
    # Add all three free HuggingFace models
    for model_key, model_config in FREE_HF_MODELS.items():
        print(f"\nLoading {model_config['display_name']}...")
        hf_provider = HuggingFaceProvider(
            model_name=model_config['model_id'],
            use_inference_api=True
        )
        if hf_provider.available:
            hf_provider.display_name = model_config['display_name']
            orchestrator.add_provider(hf_provider)
            print(f"[OK] Added {model_config['display_name']}")
        else:
            print(f"[SKIP] {model_config['display_name']} not available")
    
    # OpenAI (requires API key)
    openai_provider = OpenAIProvider(model_name="gpt-3.5-turbo")
    if openai_provider.available:
        orchestrator.add_provider(openai_provider)
        print(f"[OK] Added OpenAI provider: {openai_provider.model_name}")
    else:
        print("[SKIP] OpenAI provider not available (no API key)")
    
    # Anthropic (requires API key)
    anthropic_provider = AnthropicProvider(model_name="claude-3-haiku-20240307")
    if anthropic_provider.available:
        orchestrator.add_provider(anthropic_provider)
        print(f"[OK] Added Anthropic provider: {anthropic_provider.model_name}")
    else:
        print("[SKIP] Anthropic provider not available (no API key)")
    
    # Test query
    test_query = "Find hotels in Cairo with rating > 4"
    print(f"\n{'='*80}")
    print(f"Test Query: {test_query}")
    print(f"{'='*80}\n")
    
    # Generate response with first available model
    if orchestrator.providers:
        model_name = list(orchestrator.providers.keys())[0]
        print(f"Generating response with {model_name}...")
        response = orchestrator.generate_response(test_query, mock_retrieval, model_name=model_name)
        
        print(f"\nResponse ({response.response_time:.2f}s):")
        print(response.text)
        print(f"\nTokens: {response.tokens_used}, Cost: ${response.cost:.4f}" if response.cost else f"\nTokens: {response.tokens_used}, Cost: Free")
        
        # Compare models if multiple available
        if len(orchestrator.providers) > 1:
            print(f"\n{'='*80}")
            print("Comparing all available models...")
            print(f"{'='*80}\n")
            
            responses = orchestrator.compare_models(test_query, mock_retrieval)
            
            # Evaluate with enhanced metrics
            evaluator = ModelEvaluator()
            context = orchestrator.context_merger.format_context(
                orchestrator.context_merger.merge_results(mock_retrieval)
            )
            comparisons = evaluator.compare_responses(
                responses, 
                context, 
                test_query,
                include_qualitative=True
            )
            report = evaluator.generate_comparison_report(
                comparisons, 
                responses,
                include_cost_analysis=True,
                include_qualitative=True
            )
            print(report)
            
            # Run test suite if time permits
            print(f"\n{'='*80}")
            print("Test Suite Evaluation (first 2 tests)")
            print(f"{'='*80}\n")
            
            test_suite = TestSuite()
            for i in range(min(2, len(test_suite.tests))):
                test = test_suite.get_test(i)
                print(f"\nTest {i+1}: {test.query}")
                print(f"Difficulty: {test.difficulty}")
                print(f"Expected keywords: {', '.join(test.expected_keywords)}\n")
                
                # Quick single model test
                model_name = list(orchestrator.providers.keys())[0]
                mock_test_retrieval = {
                    'intent': test.intent,
                    'method': 'baseline',
                    'baseline_results': [],
                    'embedding_results': [],
                    'merged_results': []
                }
                response = orchestrator.generate_response(test.query, mock_test_retrieval, model_name)
                metrics = evaluator.evaluate_response(response, test.context, test.query)
                
                print(f"Response: {response.text[:150]}..." if len(response.text) > 150 else f"Response: {response.text}")
                if 'qualitative_scores' in metrics:
                    print(f"Quality Score: {metrics.get('overall_quality', 0):.2f}/5.0")
                print("-" * 80)
    else:
        print("No LLM providers available. Install required packages and set API keys.")

