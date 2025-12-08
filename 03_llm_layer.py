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
            # For hotel queries, use hotel_id
            hotel_id = item.get('hotel_id') or item.get('h', {}).get('hotel_id')
            if hotel_id:
                return str(hotel_id)
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
        
        if intent == 'visa_check':
            # Format visa information
            lines = ["Visa Requirements Information:"]
            for item in items:
                from_name = item.get('from_name') or item.get('from_country', 'Unknown')
                to_name = item.get('to_name') or item.get('to_country', 'Unknown')
                visa_type = item.get('v.visa_type') or item.get('visa_type', 'Not specified')
                visa_status = item.get('visa_status', 'Unknown')
                
                lines.append(f"\nFrom: {from_name}")
                lines.append(f"To: {to_name}")
                lines.append(f"Visa Status: {visa_status}")
                if visa_type and visa_type != 'Not specified':
                    lines.append(f"Visa Type: {visa_type}")
                lines.append("---")
        else:
            # Format hotel information
            lines = ["Hotel Information from Knowledge Graph:"]
            for i, item in enumerate(items, 1):
                hotel_name = item.get('hotel_name') or item.get('h', {}).get('name') or item.get('name', 'Unknown Hotel')
                city = item.get('city') or item.get('h', {}).get('city', 'Unknown City')
                country = item.get('country') or item.get('h', {}).get('country', 'Unknown Country')
                
                lines.append(f"\n{i}. {hotel_name}")
                lines.append(f"   Location: {city}, {country}")
                
                # Add ratings/scores if available
                if 'average_reviews_score' in item or 'h' in item:
                    score = item.get('average_reviews_score') or item.get('h', {}).get('average_reviews_score')
                    if score:
                        lines.append(f"   Average Rating: {score:.2f}")
                
                if 'star_rating' in item or 'h' in item:
                    stars = item.get('star_rating') or item.get('h', {}).get('star_rating')
                    if stars:
                        lines.append(f"   Star Rating: {stars}")
                
                # Add embedding score if available
                if 'score' in item:
                    lines.append(f"   Relevance Score: {item['score']:.3f}")
                
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
                        # For other instruction models, use similar approach
                        if "llama" in self.model_name.lower() and "3" in self.model_name.lower():
                            formatted_prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                        elif "mistral" in self.model_name.lower():
                            formatted_prompt = f"<s>[INST] {prompt} [/INST]"
                        else:
                            formatted_prompt = f"### User:\n{prompt}\n\n### Assistant:\n"
                        
                        try:
                            response_text = self.api_client.text_generation(
                                formatted_prompt,
                                model=self.model_name,
                                max_new_tokens=max_length,
                                temperature=0.7,
                                return_full_text=False
                            )
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
        # Merge context
        merged_context = self.context_merger.merge_results(retrieval_result)
        formatted_context = self.context_merger.format_context(merged_context)
        
        # Build prompt
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
        
        # Generate response
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
    notes: Optional[str] = None


class ModelEvaluator:
    """Evaluate and compare LLM models."""
    
    def __init__(self):
        """Initialize evaluator."""
        pass
    
    def evaluate_response(
        self,
        response: LLMResponse,
        context: str,
        user_query: str
    ) -> Dict[str, Any]:
        """
        Evaluate a single response (quantitative metrics).
        
        Args:
            response: LLMResponse to evaluate
            context: Original context used
            user_query: Original user query
        
        Returns:
            Dict with evaluation metrics
        """
        metrics = {
            'model_name': response.model_name,
            'response_time': response.response_time,
            'tokens_used': response.tokens_used,
            'cost': response.cost,
            'response_length': len(response.text),
            'has_error': response.error is not None,
        }
        
        # Calculate some basic quality metrics
        if response.text and not response.error:
            # Check if response mentions context keywords
            context_words = set(context.lower().split())
            response_words = set(response.text.lower().split())
            overlap = len(context_words.intersection(response_words))
            metrics['context_overlap'] = overlap / max(len(context_words), 1)
            
            # Check if response is too short (might be incomplete)
            metrics['is_too_short'] = len(response.text.split()) < 10
        
        return metrics
    
    def compare_responses(
        self,
        responses: Dict[str, LLMResponse],
        context: str,
        user_query: str
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare multiple responses and generate comparison report.
        
        Args:
            responses: Dict mapping model_name to LLMResponse
            context: Original context used
            user_query: Original user query
        
        Returns:
            Dict mapping model_name to evaluation metrics
        """
        comparisons = {}
        
        for model_name, response in responses.items():
            metrics = self.evaluate_response(response, context, user_query)
            comparisons[model_name] = metrics
        
        return comparisons
    
    def generate_comparison_report(
        self,
        comparisons: Dict[str, Dict[str, Any]],
        responses: Dict[str, LLMResponse]
    ) -> str:
        """
        Generate human-readable comparison report.
        
        Args:
            comparisons: Output from compare_responses()
            responses: Dict mapping model_name to LLMResponse
        
        Returns:
            Formatted comparison report string
        """
        lines = ["=" * 80]
        lines.append("LLM MODEL COMPARISON REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        # Quantitative metrics table
        lines.append("QUANTITATIVE METRICS:")
        lines.append("-" * 80)
        lines.append(f"{'Model':<30} {'Time(s)':<10} {'Tokens':<10} {'Cost($)':<10} {'Length':<10}")
        lines.append("-" * 80)
        
        for model_name, metrics in comparisons.items():
            time_str = f"{metrics.get('response_time', 0):.2f}"
            tokens_str = str(metrics.get('tokens_used', 'N/A'))
            cost_str = f"${metrics.get('cost', 0):.4f}" if metrics.get('cost') else "Free"
            length_str = str(metrics.get('response_length', 0))
            
            lines.append(f"{model_name:<30} {time_str:<10} {tokens_str:<10} {cost_str:<10} {length_str:<10}")
        
        lines.append("")
        lines.append("QUALITATIVE EVALUATION:")
        lines.append("-" * 80)
        
        for model_name, response in responses.items():
            lines.append(f"\n{model_name}:")
            lines.append(f"Response: {response.text[:200]}..." if len(response.text) > 200 else f"Response: {response.text}")
            if response.error:
                lines.append(f"Error: {response.error}")
            lines.append("")
        
        lines.append("=" * 80)
        
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
    
    # Add providers (using free/local models for demo)
    print("\nInitializing LLM providers...")
    
    # HuggingFace (free, local)
    hf_provider = HuggingFaceProvider(model_name="gpt2")
    if hf_provider.available:
        orchestrator.add_provider(hf_provider)
        print(f"[OK] Added HuggingFace provider: {hf_provider.model_name}")
    
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
            
            # Evaluate
            evaluator = ModelEvaluator()
            context = orchestrator.context_merger.format_context(
                orchestrator.context_merger.merge_results(mock_retrieval)
            )
            comparisons = evaluator.compare_responses(responses, context, test_query)
            report = evaluator.generate_comparison_report(comparisons, responses)
            print(report)
    else:
        print("No LLM providers available. Install required packages and set API keys.")

