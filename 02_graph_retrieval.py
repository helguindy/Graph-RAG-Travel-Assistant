"""
02_GRAPH_RETRIEVAL.PY
=====================================================================
Milestone 3: Graph Retrieval Layer

This module implements the second stage of the Graph-RAG pipeline: retrieving
relevant information from the Neo4j Knowledge Graph using two approaches:

  A. BASELINE RETRIEVAL: Deterministic Cypher queries with exact matches/filters
  B. EMBEDDING-BASED RETRIEVAL: Semantic similarity search using vector embeddings

Contains:
  - 10+ Cypher query templates mapped to intents
  - Baseline retriever (executes Cypher queries)
  - Embedding retriever (placeholder for vector similarity)
  - Query executor that handles Neo4j connections and fallbacks
=====================================================================
"""

from typing import Dict, List, Optional, Any
from neo4j import GraphDatabase
import os
import csv
import json
import math
import time
import importlib.util
from pathlib import Path

try:
    import numpy as np
except Exception:
    np = None


# =====================================================================
# SECTION A: CYPHER QUERY TEMPLATES (10+ Queries)
# =====================================================================
# Deterministic Cypher templates for baseline retrieval
# Each template maps to a user intent and requires specific parameters

class CypherTemplates:
    """Library of 10+ Cypher query templates for hotel domain."""

    @staticmethod
    def get_template(intent: str) -> str:
        """
        Retrieve Cypher template by intent.

        Args:
            intent: User intent (e.g., 'hotel_search', 'recommendation')

        Returns:
            Parameterized Cypher query template
        """
        templates = {
            # Query 1: Basic hotel search by city and minimum rating
            'hotel_search': (
                "MATCH (h:Hotel)-[:LOCATED_IN]->(c:City {name:$city}) "
                "WHERE h.average_reviews_score >= $min_rating "
                "RETURN h.hotel_id, h.name, h.star_rating, h.average_reviews_score "
                "LIMIT 50"
            ),

            # Query 2: Filter hotels by city and numeric attributes (star rating, cleanliness, etc.)
            'hotel_filter': (
                "MATCH (h:Hotel)-[:LOCATED_IN]->(c:City {name:$city}) "
                "WHERE h.star_rating >= $min_star "
                "AND h.cleanliness_base >= $min_clean "
                "AND h.comfort_base >= $min_comfort "
                "AND h.facilities_base >= $min_facilities "
                "RETURN h.hotel_id, h.name, h.star_rating, h.cleanliness_base, h.comfort_base "
                "LIMIT 50"
            ),

            # Query 3: Get hotel details by exact name match
            'hotel_details': (
                "MATCH (h:Hotel {name:$hotel_name}) "
                "RETURN h"
            ),

            # Query 4: Find hotels by country
            'hotels_by_country': (
                "MATCH (h:Hotel)-[:LOCATED_IN]->(c:City)-[:LOCATED_IN]->(co:Country {name:$country}) "
                "RETURN h.hotel_id, h.name, h.star_rating, c.name as city "
                "LIMIT 50"
            ),

            # Query 5: Recommendation for specific traveller type
            'recommend_by_type': (
                "MATCH (t:Traveller {type:$traveller_type})-[:STAYED_AT]->(h:Hotel) "
                "WITH h, COUNT(t) as traveller_count "
                "RETURN h.hotel_id, h.name, h.average_reviews_score, traveller_count "
                "ORDER BY traveller_count DESC, h.average_reviews_score DESC "
                "LIMIT 10"
            ),

            # Query 6: Top-rated hotels in a city
            'top_hotels_by_city': (
                "MATCH (h:Hotel)-[:LOCATED_IN]->(c:City {name:$city}) "
                "RETURN h.hotel_id, h.name, h.average_reviews_score "
                "ORDER BY h.average_reviews_score DESC "
                "LIMIT 10"
            ),

            # Query 7: Visa requirement check
            'visa_check': (
                "MATCH (from:Country {name:$from_country})-[v:NEEDS_VISA]->(to:Country {name:$to_country}) "
                "RETURN v.visa_type, from.name, to.name"
            ),

            # Query 8: Hotels available for a specific date range (if travellers have stayed)
            'hotels_by_date': (
                "MATCH (t:Traveller)-[:WROTE]->(r:Review {date:$target_date})-[:REVIEWED]->(h:Hotel) "
                "RETURN DISTINCT h.hotel_id, h.name, COUNT(r) as reviews_on_date "
                "LIMIT 20"
            ),

            # Query 9: Hotels with high comfort and facilities scores
            'comfortable_hotels': (
                "MATCH (h:Hotel) "
                "WHERE h.comfort_base >= $min_comfort AND h.facilities_base >= $min_facilities "
                "RETURN h.hotel_id, h.name, h.comfort_base, h.facilities_base "
                "ORDER BY h.comfort_base DESC, h.facilities_base DESC "
                "LIMIT 20"
            ),

            # Query 10: Best value-for-money hotels in a city
            'best_value': (
                "MATCH (h:Hotel)-[:LOCATED_IN]->(c:City {name:$city}) "
                "RETURN h.hotel_id, h.name, h.value_for_money_base, h.star_rating "
                "ORDER BY h.value_for_money_base DESC "
                "LIMIT 10"
            ),

            # Query 11: Hotels popular with a specific demographic (gender)
            'hotels_by_demographic': (
                "MATCH (t:Traveller {gender:$gender})-[:STAYED_AT]->(h:Hotel) "
                "WITH h, COUNT(t) as popularity "
                "RETURN h.hotel_id, h.name, h.average_reviews_score, popularity "
                "ORDER BY popularity DESC, h.average_reviews_score DESC "
                "LIMIT 10"
            ),

            # Query 12: Hotels in a country with travellers from specific origin
            'hotels_for_origin': (
                "MATCH (t:Traveller)-[:FROM_COUNTRY]->(origin:Country {name:$origin_country}) "
                "MATCH (t)-[:STAYED_AT]->(h:Hotel)-[:LOCATED_IN]->(c:City)-[:LOCATED_IN]->(dest:Country {name:$dest_country}) "
                "RETURN DISTINCT h.hotel_id, h.name, COUNT(t) as visitor_count "
                "ORDER BY visitor_count DESC "
                "LIMIT 20"
            ),


        }

        return templates.get(intent, "-- No template for intent: {} --".format(intent))


# =====================================================================
# SECTION B: BASELINE RETRIEVER (Cypher Executor)
# =====================================================================
# Executes deterministic Cypher queries against Neo4j

class BaselineRetriever:
    """Execute Cypher queries against Neo4j Knowledge Graph."""

    def __init__(self, uri: str, username: str, password: str):
        """
        Initialize Neo4j driver.

        Args:
            uri: Neo4j connection URI (e.g., 'neo4j+s://...' or 'bolt://localhost:7687')
            username: Neo4j username
            password: Neo4j password
        """
        self.uri = uri
        self.username = username
        self.password = password
        self.driver = None
        self.connected = False

        try:
            self.driver = GraphDatabase.driver(uri, auth=(username, password))
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            self.connected = True
            print(f"✓ Connected to Neo4j: {uri}")
        except Exception as e:
            print(f"✗ Failed to connect to Neo4j: {e}")
            self.connected = False

    def execute(self, template: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a Cypher query template with parameters.

        Args:
            template: Cypher query template (with $param placeholders)
            params: Dict of parameter values to fill template

        Returns:
            Dict with 'status', 'results', 'error' (if failed)
        """
        if not self.connected or not self.driver:
            return {
                'status': 'disconnected',
                'results': [],
                'error': 'Not connected to Neo4j',
                'template': template,
                'params': params,
            }

        try:
            records = []
            with self.driver.session() as session:
                result = session.run(template, **params)
                for record in result:
                    records.append(dict(record))

            return {
                'status': 'success',
                'results': records,
                'count': len(records),
                'template': template,
                'params': params,
            }
        except Exception as e:
            return {
                'status': 'error',
                'results': [],
                'error': str(e),
                'template': template,
                'params': params,
            }

    def close(self):
        """Close Neo4j driver connection."""
        if self.driver:
            self.driver.close()


class LocalBaselineRetriever:
    """Fallback local retriever that reads CSV files and performs simple filters.

    This allows the app to provide baseline-style results when Neo4j is unavailable.
    It supports a subset of the Cypher templates (hotel_search, hotel_filter,
    hotel_details, hotels_by_country, top_hotels_by_city, comfortable_hotels, best_value).
    """

    def __init__(self, csv_dir: str = 'csv'):
        self.csv_dir = Path(csv_dir)
        self.is_local = True
        # Load CSVs into pandas DataFrames lazily
        try:
            import pandas as pd
        except Exception:
            pd = None
        self.pd = pd
        self.hotels = None
        self.reviews = None
        self.users = None
        if pd is not None:
            try:
                self.hotels = pd.read_csv(self.csv_dir / 'hotels.csv')
            except Exception:
                self.hotels = None
            try:
                self.reviews = pd.read_csv(self.csv_dir / 'reviews.csv')
            except Exception:
                self.reviews = None
            try:
                self.users = pd.read_csv(self.csv_dir / 'users.csv')
            except Exception:
                self.users = None

    def _ensure_avg_scores(self):
        if self.hotels is None:
            return
        if self.reviews is None:
            # if no reviews, attempt to use existing column
            if 'average_reviews_score' not in self.hotels.columns:
                self.hotels['average_reviews_score'] = 0.0
            return
        if 'average_reviews_score' not in self.hotels.columns or self.hotels['average_reviews_score'].isnull().all():
            grp = self.reviews.groupby('hotel_id')['score_overall'].mean().reset_index()
            grp.columns = ['hotel_id', 'avg_score']
            self.hotels = self.hotels.merge(grp, how='left', left_on='hotel_id', right_on='hotel_id')
            self.hotels['average_reviews_score'] = self.hotels['avg_score'].fillna(0.0)
            self.hotels.drop(columns=['avg_score'], inplace=True, errors='ignore')

    def execute(self, template: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a pseudo-Cypher template by detecting the intent from the template text
        and applying equivalent pandas filters.
        """
        if self.pd is None or self.hotels is None:
            return {'status': 'no_local_data', 'results': [], 'error': 'pandas or CSVs not available'}

        self._ensure_avg_scores()
        df = self.hotels.copy()

        # Detect intent by template contents
        t = (template or '').lower()

        try:
            if 'average_reviews_score' in t and 'city' in t:
                # hotel_search
                city = params.get('city')
                min_rating = float(params.get('min_rating', 0.0))
                if city:
                    df = df[df['city'].str.lower() == str(city).lower()]
                df = df[df['average_reviews_score'].astype(float) >= min_rating]
                df = df[['hotel_id', 'hotel_name', 'star_rating', 'average_reviews_score']].head(50)
                results = df.to_dict(orient='records')
                return {'status': 'success', 'results': results, 'count': len(results)}

            if 'cleanliness_base' in t or 'comfort_base' in t:
                # hotel_filter
                city = params.get('city')
                min_star = float(params.get('min_star', 0.0))
                min_clean = float(params.get('min_clean', 0.0))
                min_comfort = float(params.get('min_comfort', 0.0))
                min_facilities = float(params.get('min_facilities', 0.0))
                if city:
                    df = df[df['city'].str.lower() == str(city).lower()]
                if 'star_rating' in df.columns:
                    df = df[df['star_rating'].astype(float) >= min_star]
                for col, th in [('cleanliness_base', min_clean), ('comfort_base', min_comfort), ('facilities_base', min_facilities)]:
                    if col in df.columns:
                        df = df[df[col].astype(float) >= th]
                results = df[['hotel_id', 'hotel_name', 'star_rating', 'cleanliness_base', 'comfort_base']].head(50).to_dict(orient='records')
                return {'status': 'success', 'results': results, 'count': len(results)}

            if 'match (h:hotel {name:$hotel_name})' in t or 'match (h:hotel {name:$hotel_name})' in template:
                # hotel_details exact name
                name = params.get('hotel_name')
                if name:
                    df2 = df[df['hotel_name'].str.lower() == str(name).lower()]
                    results = df2.to_dict(orient='records')
                    return {'status': 'success', 'results': results, 'count': len(results)}

            if 'country {name:$country}' in t:
                country = params.get('country')
                if country:
                    df2 = df[df['country'].str.lower() == str(country).lower()]
                    results = df2[['hotel_id', 'hotel_name', 'star_rating', 'city']].to_dict(orient='records')
                    return {'status': 'success', 'results': results, 'count': len(results)}

            if 'order by h.average_reviews_score desc' in t:
                city = params.get('city')
                if city:
                    df = df[df['city'].str.lower() == str(city).lower()]
                df2 = df.sort_values(by='average_reviews_score', ascending=False)[['hotel_id', 'hotel_name', 'average_reviews_score']].head(10)
                return {'status': 'success', 'results': df2.to_dict(orient='records'), 'count': len(df2)}

            # Best effort fallback: return top by average_reviews_score
            df2 = df.sort_values(by='average_reviews_score', ascending=False).head(10)
            return {'status': 'success', 'results': df2.to_dict(orient='records'), 'count': len(df2)}
        except Exception as e:
            return {'status': 'error', 'results': [], 'error': str(e)}


# =====================================================================
# SECTION C: EMBEDDING-BASED RETRIEVER (Placeholder)
# =====================================================================
# Semantic similarity search using vector embeddings

class EmbeddingRetriever:
    """In-memory embedding retriever using hotel descriptions from CSV.

    If a cache of hotel embeddings exists (`.cache/hotel_embeddings.npz`), it will be
    loaded; otherwise embeddings will be computed using the provided embedder and
    cached for future runs.
    """

    def __init__(self, embedder: Optional[Any] = None, csv_dir: str = 'csv', cache_dir: str = '.cache'):
        self.embedder = embedder
        self.csv_dir = csv_dir
        self.cache_dir = Path(cache_dir)
        self.available = embedder is not None and getattr(embedder, 'available', False)
        self.hotel_meta: List[Dict[str, Any]] = []
        self.embeddings: Optional[Any] = None

        # Ensure cache dir exists
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        # Try to load cache; if not available and embedder exists, build cache
        if self._load_cache():
            return

        if self.available:
            try:
                self._build_and_cache_embeddings()
            except Exception as e:
                print(f"Warning: failed to build hotel embeddings: {e}")

    def _hotel_text(self, row: Dict[str, str]) -> str:
        # Create a compact text representation for embedding
        parts = [row.get('hotel_name', ''), row.get('city', ''), row.get('country', '')]
        # include star rating and key numeric attributes if present
        for k in ['star_rating', 'cleanliness_base', 'comfort_base', 'facilities_base', 'location_base', 'staff_base', 'value_for_money_base']:
            if k in row and row[k]:
                parts.append(f"{k.replace('_',' ')} {row[k]}")
        return ' | '.join([p for p in parts if p])

    def _load_cache(self) -> bool:
        # Load .npz cache if present
        embeddings_path = self.cache_dir / 'hotel_embeddings.npz'
        meta_path = self.cache_dir / 'hotel_meta.json'
        if embeddings_path.exists() and meta_path.exists() and np is not None:
            try:
                data = np.load(embeddings_path, allow_pickle=True)
                self.embeddings = data['arr_0']
                with open(meta_path, 'r', encoding='utf-8') as f:
                    self.hotel_meta = json.load(f)
                print(f"✓ Loaded {len(self.hotel_meta)} hotel embeddings from cache")
                return True
            except Exception as e:
                print(f"Warning: failed to load cache: {e}")
        return False

    def _build_and_cache_embeddings(self):
        hotels_path = Path(self.csv_dir) / 'hotels.csv'
        if not hotels_path.exists():
            raise FileNotFoundError(f"{hotels_path} not found")

        rows: List[Dict[str, str]] = []
        with open(hotels_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)

        texts = [self._hotel_text(r) for r in rows]
        # compute embeddings in batches to be memory-friendly
        all_vecs = []
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            vecs = self.embedder.model.encode(batch, show_progress_bar=False)
            all_vecs.append(vecs)
            time.sleep(0.01)

        if np is None:
            raise RuntimeError('numpy is required for embedding retriever')

        emb_array = np.vstack(all_vecs)
        # Save cache
        embeddings_path = self.cache_dir / 'hotel_embeddings.npz'
        meta_path = self.cache_dir / 'hotel_meta.json'
        np.savez_compressed(embeddings_path, emb_array)
        # Build metadata list
        self.hotel_meta = []
        for r in rows:
            self.hotel_meta.append({
                'hotel_id': int(r.get('hotel_id')) if r.get('hotel_id') else None,
                'hotel_name': r.get('hotel_name'),
                'city': r.get('city'),
                'country': r.get('country'),
            })
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(self.hotel_meta, f, ensure_ascii=False, indent=2)
        self.embeddings = emb_array
        print(f"✓ Built and cached {len(self.hotel_meta)} hotel embeddings")

    def _cosine_sim(self, q: Any, matrix: Any) -> List[float]:
        # q: 1D vector, matrix: 2D array (n x d)
        if np is None:
            raise RuntimeError('numpy required for similarity')
        q = np.array(q, dtype=float)
        mat = np.array(matrix, dtype=float)
        q_norm = np.linalg.norm(q)
        mat_norms = np.linalg.norm(mat, axis=1)
        # avoid division by zero
        mat_norms[mat_norms == 0] = 1e-12
        if q_norm == 0:
            q_norm = 1e-12
        sims = np.dot(mat, q) / (mat_norms * q_norm)
        return sims.tolist()

    def retrieve(self, query_embedding: Optional[List[float]], top_k: int = 10) -> Dict[str, Any]:
        if query_embedding is None:
            return {'status': 'no_query_embedding', 'results': [], 'message': 'Query embedding missing'}

        if self.embeddings is None:
            return {'status': 'no_embeddings', 'results': [], 'message': 'No hotel embeddings available'}

        sims = self._cosine_sim(query_embedding, self.embeddings)
        idxs = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:top_k]
        results = []
        for i in idxs:
            meta = self.hotel_meta[i].copy()
            meta['score'] = float(sims[i])
            results.append(meta)

        return {'status': 'success', 'results': results, 'count': len(results)}


# =====================================================================
# SECTION D: HYBRID RETRIEVER (Baseline + Embeddings)
# =====================================================================
# Combine results from both baseline and embedding approaches

class HybridRetriever:
    """Orchestrate baseline and embedding-based retrieval."""

    def __init__(
        self,
        baseline_retriever: Optional[BaselineRetriever] = None,
        embedding_retriever: Optional[EmbeddingRetriever] = None,
    ):
        """
        Initialize hybrid retriever with both components.

        Args:
            baseline_retriever: BaselineRetriever instance
            embedding_retriever: EmbeddingRetriever instance
        """
        self.baseline = baseline_retriever
        self.embedding = embedding_retriever

    def retrieve(
        self,
        intent: str,
        entities: Dict[str, Optional[str]],
        query_embedding: Optional[List[float]] = None,
        method: str = 'baseline',
    ) -> Dict[str, Any]:
        """
        Retrieve results using specified method (baseline, embedding, or hybrid).

        Args:
            intent: User intent (maps to Cypher template)
            entities: Extracted entities to fill query parameters
            query_embedding: Query vector embedding (for embedding method)
            method: 'baseline', 'embeddings', or 'hybrid'

        Returns:
            Dict with retrieval results
        """
        results = {
            'intent': intent,
            'method': method,
            'baseline_results': [],
            'embedding_results': [],
            'merged_results': [],
        }

        # Build parameters from entities (handles multiple rating filters)
        params = build_cypher_params(entities)

        # Choose template: if rating_types present prefer 'hotel_filter' or 'comfortable_hotels'
        chosen_intent = intent
        rating_types = entities.get('rating_types') or []
        if intent in ['hotel_search', 'hotel_filter'] and rating_types:
            # If specific numeric rating types requested, use 'hotel_filter'
            chosen_intent = 'hotel_filter'
        else:
            chosen_intent = intent

        # Run baseline if requested
        if method in ['baseline', 'hybrid']:
            if self.baseline:
                template = CypherTemplates.get_template(chosen_intent)
                baseline_result = self.baseline.execute(template, params)
                results['baseline_results'] = baseline_result.get('results', [])
                results['baseline_status'] = baseline_result.get('status')

        # Run embedding if requested
        if method in ['embeddings', 'hybrid']:
            if self.embedding:
                embedding_result = self.embedding.retrieve(query_embedding, top_k=10)
                results['embedding_results'] = embedding_result.get('results', [])
                results['embedding_status'] = embedding_result.get('status')

        # Merge results if hybrid
        if method == 'hybrid':
            # Simple merge: combine and deduplicate by hotel_id
            seen_hotels = set()
            merged = []
            for item in results['baseline_results'] + results['embedding_results']:
                hotel_id = item.get('hotel_id') or item.get('h', {}).get('hotel_id')
                if hotel_id and hotel_id not in seen_hotels:
                    merged.append(item)
                    seen_hotels.add(hotel_id)
            results['merged_results'] = merged

        return results


# =====================================================================
# CONFIG LOADER
# =====================================================================

def load_neo4j_config(config_path: str = 'config.txt') -> Dict[str, str]:
    """
    Load Neo4j connection credentials from config file.

    Args:
        config_path: Path to config.txt (URI, USERNAME, PASSWORD)

    Returns:
        Dict with 'uri', 'username', 'password'
    """
    config = {'uri': None, 'username': None, 'password': None}

    if not os.path.exists(config_path):
        return config

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            for line in f:
                if '=' in line:
                    key, value = line.strip().split('=', 1)
                    key = key.strip().lower()
                    if key == 'uri':
                        config['uri'] = value.strip()
                    elif key == 'username':
                        config['username'] = value.strip()
                    elif key == 'password':
                        config['password'] = value.strip()
    except Exception as e:
        print(f"Warning: could not load config.txt: {e}")

    return config


# =====================================================================
# HELPERS: Build Cypher parameters from extracted entities
# =====================================================================

def build_cypher_params(entities: Dict[str, Optional[Any]]) -> Dict[str, Any]:
    """Build a parameter dict for Cypher templates from extracted entities.

    Supports mapping multiple rating types (e.g., cleanliness, comfort)
    into the expected template parameter names.

    Args:
        entities: Extracted entities from preprocessing (may include keys:
                  'city', 'country', 'hotel', 'traveller_type', 'min_rating',
                  'rating_types' (list), 'date')

    Returns:
        Dict of parameters to pass to BaselineRetriever.execute()
    """
    params: Dict[str, Any] = {}

    if not entities:
        return params

    # Basic pass-throughs
    if entities.get('city'):
        params['city'] = entities['city']
    if entities.get('country'):
        params['country'] = entities['country']
    if entities.get('hotel'):
        # templates expect $hotel_name
        params['hotel_name'] = entities['hotel']
    if entities.get('traveller_type'):
        params['traveller_type'] = entities['traveller_type']
    if entities.get('date'):
        params['target_date'] = entities['date']

    # Default numeric thresholds
    min_rating = entities.get('min_rating')
    if min_rating is None:
        # sensible defaults
        min_rating = 0.0

    # If user provided explicit rating_types (list), map them to template params
    rating_types = entities.get('rating_types') or []
    # map of rating type -> template parameter name
    type_to_param = {
        'cleanliness': 'min_clean',
        'comfort': 'min_comfort',
        'facilities': 'min_facilities',
        'location': 'min_location',
        'staff': 'min_staff',
        'value': 'min_value',
    }

    # Fill all rating params with either the provided min_rating or 0
    for param in set(type_to_param.values()):
        params[param] = 0.0

    for r in rating_types:
        p = type_to_param.get(r)
        if p:
            params[p] = float(min_rating)

    # For templates that use a generic min_rating (e.g., hotel_search)
    params['min_rating'] = float(min_rating)

    # If star-based query expected
    if entities.get('min_rating'):
        # also populate star threshold
        params['min_star'] = float(entities.get('min_rating'))

    return params


# =====================================================================
# EXAMPLE USAGE & TESTING
# =====================================================================

if __name__ == '__main__':
    print("=" * 80)
    print("GRAPH RETRIEVAL LAYER DEMO")
    print("=" * 80)

    # Load Neo4j config
    config = load_neo4j_config('config.txt')
    print(f"\nConfig loaded:")
    print(f"  URI: {config['uri']}")
    print(f"  Username: {config['username']}")

    # Initialize baseline retriever
    if config['uri'] and config['username'] and config['password']:
        baseline = BaselineRetriever(config['uri'], config['username'], config['password'])
    else:
        print("\nWarning: Neo4j config incomplete. Using mock results only.")
        baseline = None

    # Initialize embedding retriever (without actual embeddings)
    embedding = EmbeddingRetriever(embedder=None)

    # Initialize hybrid retriever
    hybrid = HybridRetriever(baseline, embedding)

    # Test query
    print("\n" + "-" * 80)
    print("Test Query: 'Find hotels in Cairo with rating > 4'")
    print("-" * 80)

    entities = {
        'city': 'Cairo',
        'min_rating': 4.0,
    }

    result = hybrid.retrieve('hotel_search', entities, method='baseline')
    print(f"Intent: hotel_search")
    print(f"Entities: {entities}")
    print(f"Baseline Status: {result.get('baseline_status')}")
    print(f"Results Count: {len(result.get('baseline_results', []))}")
    if result.get('baseline_results'):
        print(f"Sample Results: {result['baseline_results'][:2]}")

    # ------------------------------------------------------------------
    # Multi-rating filters demo
    # ------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("Test Query: 'Hotels with excellent cleanliness AND great staff service'")
    print("-" * 80)

    multi_entities = {
        'city': 'Rome',
        'min_rating': 4.0,
        'rating_types': ['cleanliness', 'staff'],
    }

    # Show how parameters are built for Cypher
    built_params = build_cypher_params(multi_entities)
    print(f"Built Cypher Params: {built_params}")

    result2 = hybrid.retrieve('hotel_search', multi_entities, method='baseline')
    print(f"Chosen Template Results Count: {len(result2.get('baseline_results', []))}")
    print(f"Baseline Status: {result2.get('baseline_status')}")

    # Show available templates
    print("\n" + "-" * 80)
    print("Available Cypher Templates (10+):")
    print("-" * 80)
    templates = {
        'hotel_search': 'Basic hotel search by city and minimum rating',
        'hotel_filter': 'Filter hotels by city and numeric attributes',
        'hotel_details': 'Get hotel details by exact name match',
        'hotels_by_country': 'Find hotels by country',
        'recommend_by_type': 'Recommendation for specific traveller type',
        'top_hotels_by_city': 'Top-rated hotels in a city',
        'visa_check': 'Visa requirement check',
        'hotels_by_date': 'Hotels available for a specific date range',
        'comfortable_hotels': 'Hotels with high comfort and facilities scores',
        'best_value': 'Best value-for-money hotels in a city',
        'hotels_by_demographic': 'Hotels popular with a specific demographic',
        'hotels_for_origin': 'Hotels in a country with travellers from specific origin',
    }
    for i, (key, desc) in enumerate(templates.items(), 1):
        print(f"  {i:2d}. {key:<25} - {desc}")

    if baseline:
        baseline.close()
