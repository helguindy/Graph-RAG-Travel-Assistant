"""
01_INPUT_PREPROCESSING.PY
=====================================================================
Milestone 3: Input Preprocessing Layer

This module implements the first stage of the Graph-RAG pipeline: converting
raw user input into structured intent, extracted entities, and vector embeddings.

Three main components:
  1. INTENT CLASSIFICATION: Classify user intent (e.g., hotel search, recommendation)
  2. ENTITY EXTRACTION: Extract entities (cities, hotels, traveller types, dates, ratings)
  3. INPUT EMBEDDING: Convert user text to vector representations
=====================================================================
"""

import re
import csv
import os
from difflib import get_close_matches
from typing import Dict, List, Optional, Set, Tuple


# =====================================================================
# SECTION 1: INTENT CLASSIFICATION
# =====================================================================
# Rule-based intent classifier for Hotel theme
# Maps user input keywords to high-level intents that drive retrieval strategy

class HotelIntentClassifier:
    """Rule-based intent classifier adapted for hotel domain."""

    def __init__(self):
        """Initialize with hotel-specific intent patterns."""
        self.intents = {
            # Core search/discovery intents
            'hotel_search': ['find', 'search', 'show', 'hotels', 'stay', 'looking for', 'looking', 'where', 'which hotel'],
            'hotel_filter': ['filter', 'narrow down', 'narrow', 'by rating', 'by quality', 'filter hotels'],
            'recommendation': ['recommend', 'suggest', 'best', 'top', 'popular', 'suggest me', 'what is the best'],
            'booking': ['book', 'reserve', 'reservation', 'book a room', 'make a booking', 'check in'],
            'hotel_details': ['details', 'information about', 'tell me about', 'show details', 'info', 'describe'],
            
            # Rating/Quality-specific intents
            'rating_filter': ['rating', 'star', 'stars', 'score', 'quality', 'excellent', 'good', 'highly rated'],
            'cleanliness_filter': ['clean', 'cleanliness', 'tidy', 'hygiene', 'spotless'],
            'comfort_filter': ['comfort', 'comfortable', 'cozy', 'beds', 'bedding', 'spacious'],
            'facilities_filter': ['facilities', 'amenities', 'wifi', 'pool', 'gym', 'spa', 'parking', 'breakfast'],
            'location_filter': ['location', 'close to', 'near', 'downtown', 'center', 'beach', 'airport'],
            'staff_filter': ['staff', 'service', 'friendly', 'helpful', 'customer service'],
            'value_filter': ['value', 'price', 'budget', 'cheap', 'expensive', 'affordable', 'deal'],
            
            # Visa and travel requirements
            'visa_check': ['visa', 'need visa', 'visa requirement', 'travel document', 'passport', 'requirement'],
            
            # Demographic/personalization intents
            'demographic_search': ['family', 'couple', 'solo', 'business', 'honeymoon', 'group'],
            'age_group_search': ['young', 'elderly', 'kids', 'children', 'teens', 'families with'],
            'gender_search': ['women only', 'male friendly', 'female travel', 'lgbtq'],
            
            # Comparison/review intents
            'comparison': ['compare', 'difference between', 'which is better', 'versus', 'vs'],
            'review_search': ['reviews', 'opinions', 'feedback', 'what do people say', 'ratings'],
            'trending': ['trending', 'popular', 'new', 'latest', 'recent', 'top rated'],
        }
        self.rating_types = {
            'overall_rating': ['rating', 'overall', 'score', 'stars'],
            'cleanliness': ['clean', 'cleanliness', 'hygiene'],
            'comfort': ['comfort', 'comfortable', 'beds'],
            'facilities': ['facilities', 'amenities', 'wifi', 'pool', 'gym'],
            'location': ['location', 'close', 'near', 'downtown'],
            'staff': ['staff', 'service', 'friendly'],
            'value': ['value', 'price', 'budget'],
        }
        self.theme = 'hotel'

    def classify(self, user_input: str) -> Dict[str, str]:
        """
        Classify user intent using keyword matching.

        Args:
            user_input: Raw user query string

        Returns:
            Dict with 'intent', 'theme', 'confidence', 'rating_types' keys.
            Example: {'intent': 'rating_filter', 'theme': 'hotel', 'confidence': 'high', 'rating_types': ['cleanliness', 'staff']}
        """
        text_lower = user_input.lower()
        matched_intent = 'general'
        confidence = 'low'
        rating_types = []

        # Match keywords in priority order
        for intent, keywords in self.intents.items():
            for keyword in keywords:
                if keyword in text_lower:
                    matched_intent = intent
                    confidence = 'high'
                    break
            if confidence == 'high':
                break

        # If a rating-related intent is detected, identify all rating types mentioned
        if 'filter' in matched_intent or 'rating' in matched_intent:
            for rtype, keywords in self.rating_types.items():
                for keyword in keywords:
                    if keyword in text_lower:
                        rating_types.append(rtype)
                        break  # Move to next rating type once found

        return {
            'intent': matched_intent,
            'theme': self.theme,
            'confidence': confidence,
            'rating_types': rating_types,  # List of rating types detected
            'user_input': user_input,
        }


# =====================================================================


# =====================================================================
# SECTION 2: ENTITY EXTRACTION
# =====================================================================
# Named Entity Recognition (NER) using lookup tables and heuristics
# Extracts: hotels, cities, countries, traveller types, ratings, dates

class HotelEntityExtractor:
    """Extract hotel-domain entities from user input using lookup tables and regexes."""

    def __init__(self, csv_dir: str = "csv"):
        """
        Initialize with lookups loaded from CSV files.

        Args:
            csv_dir: Path to directory containing hotels.csv, users.csv, visa.csv
        """
        self.csv_dir = csv_dir
        self.lookups = self._load_lookups()

    def _load_lookups(self) -> Dict[str, Set[str]]:
        """
        Load entity lookups from CSV files in self.csv_dir.

        Populates sets for:
          - hotels: hotel names
          - cities: city names
          - countries: country names
          - traveller_types: business, family, couple, solo, etc.

        Returns:
            Dict mapping entity type to set of values
        """
        lookups = {
            "hotels": set(),
            "cities": set(),
            "countries": set(),
            "traveller_types": set(),
        }

        hotels_path = os.path.join(self.csv_dir, "hotels.csv")
        users_path = os.path.join(self.csv_dir, "users.csv")
        visa_path = os.path.join(self.csv_dir, "visa.csv")

        if os.path.exists(hotels_path):
            try:
                with open(hotels_path, newline='', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if 'hotel_name' in row and row['hotel_name']:
                            lookups['hotels'].add(row['hotel_name'].strip())
                        if 'city' in row and row['city']:
                            lookups['cities'].add(row['city'].strip())
                        if 'country' in row and row['country']:
                            lookups['countries'].add(row['country'].strip())
            except Exception as e:
                print(f"Warning: could not load hotels.csv: {e}")

        if os.path.exists(users_path):
            try:
                with open(users_path, newline='', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if 'traveller_type' in row and row['traveller_type']:
                            lookups['traveller_types'].add(row['traveller_type'].strip().lower())
                        if 'country' in row and row['country']:
                            lookups['countries'].add(row['country'].strip())
            except Exception as e:
                print(f"Warning: could not load users.csv: {e}")

        if os.path.exists(visa_path):
            try:
                with open(visa_path, newline='', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if 'from' in row and row['from']:
                            lookups['countries'].add(row['from'].strip())
                        if 'to' in row and row['to']:
                            lookups['countries'].add(row['to'].strip())
            except Exception as e:
                print(f"Warning: could not load visa.csv: {e}")

        return lookups

    def _fuzzy_match(self, text: str, candidates: Set[str], cutoff: float = 0.75) -> Optional[str]:
        """
        Try exact substring match first, then fuzzy match.

        Args:
            text: Input text to search in
            candidates: Set of candidate entities
            cutoff: Fuzzy match confidence threshold (0-1)

        Returns:
            Best matching entity or None
        """
        if not candidates:
            return None

        text_lower = text.lower()

        # Exact substring match
        for candidate in candidates:
            if candidate.lower() in text_lower:
                return candidate

        # Fuzzy match on tokens
        words = re.findall(r'[A-Za-z0-9\-\']+', text)
        for word in words:
            matches = get_close_matches(word, list(candidates), n=1, cutoff=cutoff)
            if matches:
                return matches[0]

        return None

    def extract(self, user_input: str) -> Dict[str, Optional[str]]:
        """
        Extract entities from user input restricted to the Step 1 requirements.

        Args:
            user_input: Raw user query

        Returns:
            Dict with extracted entities (only):
              - hotel: hotel name (or None)
              - city: city name (or None)
              - country: country name (or None)
              - traveller_type: business, family, couple, solo, etc. (or None)
              - age_group: detected age group (e.g., '25-34') or None
              - gender: detected gender mention (male/female/other) or None
        """
        entities = {
            'hotel': None,
            'city': None,
            'country': None,
            'traveller_type': None,
            'age_group': None,
            'gender': None,
        }

        # Match hotel, city, country using fuzzy lookup
        entities['hotel'] = self._fuzzy_match(user_input, self.lookups['hotels'])
        entities['city'] = self._fuzzy_match(user_input, self.lookups['cities'])
        entities['country'] = self._fuzzy_match(user_input, self.lookups['countries'])

        # Traveller type: lookup + common patterns
        entities['traveller_type'] = self._fuzzy_match(user_input, self.lookups['traveller_types'], cutoff=0.6)
        if not entities['traveller_type']:
            for ttype in ['business', 'family', 'couple', 'solo', 'backpacker']:
                if ttype in user_input.lower():
                    entities['traveller_type'] = ttype
                    break

        # Demographics: attempt to detect age groups and gender mentions
        text_lower = user_input.lower()
        # Age group patterns (simple keywords)
        age_keywords = {
            '18-24': ['18-24', '18 to 24', 'teen', 'teens'],
            '25-34': ['25-34', '25 to 34', 'young'],
            '35-44': ['35-44', '35 to 44', 'mid age', 'middle aged'],
            '45-54': ['45-54', '45 to 54', 'older'],
            '55+': ['55+', '55 and up', 'senior', 'elderly'],
        }
        for ag, kws in age_keywords.items():
            for kw in kws:
                if kw in text_lower:
                    entities['age_group'] = ag
                    break
            if entities['age_group']:
                break

        # Gender mentions
        if any(w in text_lower for w in ['female', 'women', 'woman', 'ladies']):
            entities['gender'] = 'female'
        elif any(w in text_lower for w in ['male', 'men', 'man', 'gentlemen']):
            entities['gender'] = 'male'

        return entities


# =====================================================================
# SECTION 3: INPUT EMBEDDING
# =====================================================================
# Convert user input to vector representations for semantic similarity search

class InputEmbedder:
    """Convert text input to vector embeddings using sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize with a sentence transformer model.

        Args:
            model_name: HuggingFace model ID (default: all-MiniLM-L6-v2)

        Note:
            Requires: pip install sentence-transformers
            First use will download the model (~60 MB).
        """
        self.model_name = model_name
        self.model = None
        self.available = False

        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self.available = True
        except ImportError:
            print(f"Warning: sentence-transformers not installed. Embedding disabled.")
            print("Install with: pip install sentence-transformers")
        except Exception as e:
            print(f"Warning: could not load embedding model {model_name}: {e}")

    def embed(self, text: str) -> Optional[List[float]]:
        """
        Convert text to a vector embedding.

        Args:
            text: Input text to embed

        Returns:
            List of floats (vector) or None if embedding not available

        Example:
            >>> embedder = InputEmbedder()
            >>> vec = embedder.embed("Find hotels in Cairo with rating > 4")
            >>> len(vec)  # 384 dimensions for default model
            384
        """
        if not self.available or not self.model:
            return None

        try:
            embedding = self.model.encode([text], show_progress_bar=False)
            return embedding[0].tolist()
        except Exception as e:
            print(f"Warning: embedding failed: {e}")
            return None


# =====================================================================
# MAIN PREPROCESSING PIPELINE
# =====================================================================

class InputPreprocessor:
    """Orchestrate intent classification, entity extraction, and embedding."""

    def __init__(self, csv_dir: str = "csv", embedding_model: Optional[str] = "all-MiniLM-L6-v2"):
        """
        Initialize preprocessor with all components.

        Args:
            csv_dir: Path to CSV directory
            embedding_model: Model name for embeddings, or None to disable
        """
        self.intent_classifier = HotelIntentClassifier()
        self.entity_extractor = HotelEntityExtractor(csv_dir)
        self.embedder = InputEmbedder(embedding_model) if embedding_model else None

    def process(self, user_input: str) -> Dict[str, any]:
        """
        Run full preprocessing pipeline on user input.

        Args:
            user_input: Raw user query

        Returns:
            Dict with 'intent', 'entities', 'embedding', and 'raw_input'
        """
        intent_result = self.intent_classifier.classify(user_input)
        entities = self.entity_extractor.extract(user_input)
        embedding = self.embedder.embed(user_input) if self.embedder else None

        return {
            'raw_input': user_input,
            'intent': intent_result['intent'],
            'intent_confidence': intent_result['confidence'],
            'rating_types': intent_result.get('rating_types', []),  # List of rating types
            'entities': entities,
            'embedding': embedding,
            'embedding_dim': len(embedding) if embedding else None,
        }


# =====================================================================
# EXAMPLE USAGE & TESTING
# =====================================================================

if __name__ == '__main__':
    # Create preprocessor
    preprocessor = InputPreprocessor()

    # Test queries covering all new intents and flexible rating patterns
    test_queries = [
        # Rating with flexible syntax (not just > <)
        "Find hotels in Cairo with rating > 4",
        "Show me 4+ star hotels in Tokyo",
        "I'm looking for hotels with at least 4.5 stars in London",
        
        # Specific rating type filters
        "Hotels in Paris with excellent cleanliness rating",
        "Find comfortable hotels in Dubai with good bed quality",
        "I need a hotel in Berlin with great staff service",
        "Budget-friendly hotels in Singapore with good value for money",
        
        # MULTIPLE rating filters (NEW!)
        "Hotels with excellent cleanliness AND great staff service",
        "Find hotels in Rome with good comfort and affordable prices",
        "I need a hotel with clean rooms, friendly staff, and good amenities",
        
        # Demographic/traveller type intents
        "Recommend a hotel in New York for a couple",
        "Best hotels in Sydney for families with kids",
        "Business-friendly hotels in Bangkok",
        
        # Visa and travel documents
        "Do I need a visa to travel from Egypt to France?",
        "Visa requirements from India to USA",
        
        # Hotel details and comparisons
        "Show me hotel details for The Azure Tower",
        "Compare hotels in Paris and London",
        
        # Booking and other intents
        "Book a room at The Azure Tower for 2024",
        "What are the trending hotels in Miami this year?",
    ]

    print("=" * 90)
    print("INPUT PREPROCESSING PIPELINE DEMO - ENHANCED INTENT CLASSIFICATION")
    print("=" * 90)

    for query in test_queries:
        print(f"\n{'Query:':<12} {query}")
        result = preprocessor.process(query)

        print(f"{'Intent:':<12} {result['intent']}")
        if result['rating_types']:
            print(f"{'Rating Types:':<12} {', '.join(result['rating_types'])}")
        print(f"{'Confidence:':<12} {result['intent_confidence']}")
        
        print(f"{'Entities:':<12}")
        entities_found = {k: v for k, v in result['entities'].items() if v is not None and k != 'rating_types'}
        if entities_found:
            for key, val in entities_found.items():
                print(f"  - {key}: {val}")
        else:
            print("  (none detected)")
        
        if result['embedding']:
            print(f"{'Embedding:':<12} {result['embedding_dim']} dimensions (first 5: {[round(x, 4) for x in result['embedding'][:5]]})")
        else:
            print(f"{'Embedding:':<12} Not available")
        print("-" * 90)
