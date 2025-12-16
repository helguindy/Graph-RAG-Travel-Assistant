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
# GLOBAL MAPPINGS
# =====================================================================
# City-to-Country mapping for visa queries (derived from hotels.csv)
_city_to_country_map = {
    'new york': 'United States',
    'london': 'United Kingdom',
    'paris': 'France',
    'tokyo': 'Japan',
    'dubai': 'United Arab Emirates',
    'singapore': 'Singapore',
    'sydney': 'Australia',
    'rio de janeiro': 'Brazil',
    'berlin': 'Germany',
    'toronto': 'Canada',
    'shanghai': 'China',
    'mexico city': 'Mexico',
    'mumbai': 'India',
    'rome': 'Italy',
    'cape town': 'South Africa',
    'seoul': 'South Korea',
    'moscow': 'Russia',
    'cairo': 'Egypt',
    'barcelona': 'Spain',
    'bangkok': 'Thailand',
    'istanbul': 'Turkey',
    'amsterdam': 'Netherlands',
    'buenos aires': 'Argentina',
    'lagos': 'Nigeria',
    'wellington': 'New Zealand',
    'chicago': 'United States',
    'san diego': 'United States',
    'denver': 'United States',
    'boston': 'United States',
    'atlanta': 'United States',
    'seattle': 'United States',
    'austin': 'United States',
    'orlando': 'United States',
    'philadelphia': 'United States',
    'honolulu': 'United States',
    'manchester': 'United Kingdom',
    'edinburgh': 'United Kingdom',
    'birmingham': 'United Kingdom',
    'liverpool': 'United Kingdom',
    'cambridge': 'United Kingdom',
    'lyon': 'France',
    'marseille': 'France',
    'nice': 'France',
    'bordeaux': 'France',
    'toulouse': 'France',
    'nagoya': 'Japan',
    'fukuoka': 'Japan',
    'yokohama': 'Japan',
    'sapporo': 'Japan',
    'hiroshima': 'Japan',
    'abu dhabi': 'United Arab Emirates',
    'sharjah': 'United Arab Emirates',
    'ras al khaimah': 'United Arab Emirates',
    'ajman': 'United Arab Emirates',
    'al ain': 'United Arab Emirates',
    'melbourne': 'Australia',
    'brisbane': 'Australia',
    'perth': 'Australia',
    'adelaide': 'Australia',
    'gold coast': 'Australia',
    'sao paulo': 'Brazil',
    'brasilia': 'Brazil',
    'salvador': 'Brazil',
    'curitiba': 'Brazil',
    'fortaleza': 'Brazil',
    'munich': 'Germany',
    'hamburg': 'Germany',
    'frankfurt': 'Germany',
    'cologne': 'Germany',
    'stuttgart': 'Germany',
    'vancouver': 'Canada',
    'montreal': 'Canada',
    'calgary': 'Canada',
    'ottawa': 'Canada',
    'quebec city': 'Canada',
    'beijing': 'China',
    'shenzhen': 'China',
    'guangzhou': 'China',
    'chengdu': 'China',
    'hangzhou': 'China',
    'cancun': 'Mexico',
    'guadalajara': 'Mexico',
    'monterrey': 'Mexico',
    'puebla': 'Mexico',
    'tijuana': 'Mexico',
    'new delhi': 'India',
    'bangalore': 'India',
    'hyderabad': 'India',
    'chennai': 'India',
    'jaipur': 'India',
    'milan': 'Italy',
    'florence': 'Italy',
    'venice': 'Italy',
    'naples': 'Italy',
    'turin': 'Italy',
    'johannesburg': 'South Africa',
    'pretoria': 'South Africa',
    'durban': 'South Africa',
    'port elizabeth': 'South Africa',
    'bloemfontein': 'South Africa',
    'busan': 'South Korea',
    'incheon': 'South Korea',
    'daegu': 'South Korea',
    'daejeon': 'South Korea',
    'gwangju': 'South Korea',
    'saint petersburg': 'Russia',
    'kazan': 'Russia',
    'sochi': 'Russia',
    'novosibirsk': 'Russia',
    'yekaterinburg': 'Russia',
    'alexandria': 'Egypt',
    'giza': 'Egypt',
    'luxor': 'Egypt',
    'hurghada': 'Egypt',
    'sharm el sheikh': 'Egypt',
    'madrid': 'Spain',
    'valencia': 'Spain',
    'seville': 'Spain',
    'malaga': 'Spain',
    'bilbao': 'Spain',
    'phuket': 'Thailand',
    'chiang mai': 'Thailand',
    'pattaya': 'Thailand',
    'krabi': 'Thailand',
    'koh samui': 'Thailand',
    'ankara': 'Turkey',
    'izmir': 'Turkey',
    'antalya': 'Turkey',
    'bursa': 'Turkey',
    'gaziantep': 'Turkey',
    'rotterdam': 'Netherlands',
    'utrecht': 'Netherlands',
    'the hague': 'Netherlands',
    'eindhoven': 'Netherlands',
    'maastricht': 'Netherlands',
    'cordoba': 'Argentina',
    'rosario': 'Argentina',
    'mendoza': 'Argentina',
    'mar del plata': 'Argentina',
    'salta': 'Argentina',
    'abuja': 'Nigeria',
    'port harcourt': 'Nigeria',
    'ibadan': 'Nigeria',
    'kano': 'Nigeria',
    'enugu': 'Nigeria',
    'auckland': 'New Zealand',
    'christchurch': 'New Zealand',
    'queenstown': 'New Zealand',
    'hamilton': 'New Zealand',
    'dunedin': 'New Zealand',
}

# =====================================================================
# SECTION A: CYPHER QUERY TEMPLATES (10+ Queries)
# =====================================================================
# Deterministic Cypher templates for baseline retrieval
# Each template maps to a user intent and requires specific parameters

class CypherTemplates:
    """Library of 10+ Cypher query templates for hotel domain."""
    
    # Mapping of intent to query number (matches template order)
    QUERY_NUMBERS = {
        'hotel_search': 1,              # Query 1: Basic hotel search by city and rating
        'hotel_filter': 2,              # Query 2: Filter hotels by city and numeric attributes
        'hotel_details': 3,             # Query 3: Get hotel details by exact name
        'hotels_by_country': 4,         # Query 4: Find hotels by country
        'visa_check': 5,                # Query 5: Visa requirement check
        'visa_free_destinations': 6,    # Query 6: Find visa-free destinations
        'hotels_by_country_and_type': 7, # Query 7: Hotels by country and traveller type
        'cleanest_hotels_by_country': 8, # Query 8: Cleanest hotels in a country
        'best_location_by_country': 9,  # Query 9: Best located hotels in country
        'comfortable_hotels_by_country': 10, # Query 10: Most comfortable hotels in country
        'hotel_facilities': 11,         # Query 11: Get hotel facilities by name
        'facilities_by_country': 12,    # Query 12: Get facilities in hotels by country
    }

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
            # Query 1: Basic hotel search by city and minimum rating (return quality metrics)
            # Example question: find me hotels in cairo
            'hotel_search': (
                "MATCH (h:Hotel)-[:LOCATED_IN]->(c:City {name:$city}) "
                "MATCH (c)-[:LOCATED_IN]->(co:Country) "
                "WHERE h.average_reviews_score >= $min_rating "
                "RETURN h.hotel_id, h.name, h.star_rating, h.average_reviews_score, h.cleanliness_base, h.comfort_base, h.facilities_base, h.location_base, h.staff_base, h.value_for_money_base, c.name as city, co.name as country "
                "ORDER BY h.cleanliness_base DESC, h.average_reviews_score DESC "
                "LIMIT 50"
            ),

            # Query 2: Filter hotels by city and numeric attributes (star rating, cleanliness, etc.)
            # Example question: Find hotels in Tokyo with cleanliness above 5.0
            'hotel_filter': (
                "MATCH (h:Hotel)-[:LOCATED_IN]->(c:City {name:$city}) "
                "MATCH (c)-[:LOCATED_IN]->(co:Country) "
                "WHERE h.star_rating >= $min_star "
                "AND h.cleanliness_base >= $min_clean "
                "AND h.comfort_base >= $min_comfort "
                "AND h.facilities_base >= $min_facilities "
                "RETURN h.hotel_id, h.name, h.star_rating, h.cleanliness_base, h.comfort_base, h.facilities_base, h.location_base, h.staff_base, h.value_for_money_base, h.average_reviews_score, c.name as city, co.name as country "
                "ORDER BY h.cleanliness_base DESC, h.average_reviews_score DESC "
                "LIMIT 50"
            ),

            # Query 3: Get hotel details by exact name match
            # Example question: "Tell me about The Azure Tower hotel"
            'hotel_details': (
                "MATCH (h:Hotel {name:$hotel_name})-[:LOCATED_IN]->(c:City)-[:LOCATED_IN]->(co:Country) "
                "RETURN h.hotel_id, h.name, h.star_rating, h.cleanliness_base, h.comfort_base, h.facilities_base, h.location_base, h.staff_base, h.value_for_money_base, h.average_reviews_score, h.facilities_list, c.name as city, co.name as country "
                "LIMIT 1"
            ),

            # Query 4: Find hotels by country (with quality signals)
            # Example question: "What are the best hotels in Japan?"
            'hotels_by_country': (
                "MATCH (h:Hotel)-[:LOCATED_IN]->(c:City)-[:LOCATED_IN]->(co:Country {name:$country}) "
                "RETURN h.hotel_id, h.name, h.star_rating, h.cleanliness_base, h.comfort_base, h.facilities_base, h.location_base, h.staff_base, h.value_for_money_base, h.average_reviews_score, c.name as city, co.name as country "
                "ORDER BY h.cleanliness_base DESC, h.average_reviews_score DESC "
                "LIMIT 50"
            ),

            # Query 5: Visa requirement check (between two specific countries)
            # Example question: "Do I need a visa to travel from India to United Arab Emirates?"
            'visa_check': (
                "MATCH (from:Country {name:$from_country}), (to:Country {name:$to_country}) "
                "OPTIONAL MATCH (from)-[v:NEEDS_VISA]->(to) "
                "OPTIONAL MATCH (from)-[f:VISA_FREE]->(to) "
                "RETURN coalesce(v.visa_type, f.visa_type) as visa_type, from.name as from_name, to.name as to_name, "
                "CASE "
                "  WHEN v IS NOT NULL THEN 'Visa required' "
                "  WHEN f IS NOT NULL THEN 'No visa required' "
                "  ELSE 'Unknown' "
                "END as visa_status"
            ),
            
            # Query 6: Find all visa-free destinations from a country
            # Example question: "Which countries can I visit from United Kingdom without a visa?"
            'visa_free_destinations': (
                "MATCH (from:Country {name:$from_country})-[:VISA_FREE]->(to:Country) "
                "RETURN to.name as country_name, 'No visa required' as visa_status "
                "ORDER BY to.name "
                "LIMIT 200"
            ),

            # Query 7: Find hotels in a country that match traveller preferences
            # Scoring based on traveller type + facility preferences:
            # - Couples: comfort (8) + location (8) + staff (7) + concierge/laundry bonus (5)
            # - Family: facilities (9) + cleanliness (8) + location (7) + pool/breakfast bonus (5)
            # - Business: location (9) + staff (8) + comfort (7) + concierge bonus (5)
            # - Solo: gym (8) + laundry (8) + concierge (7) + comfort (6)
            # Example question: "Which hotels in France are good for couples?"
            'hotels_by_country_and_type': (
                "MATCH (h:Hotel)-[:LOCATED_IN]->(c:City)-[:LOCATED_IN]->(co:Country {name:$country}) "
                "WITH h, c.name as city, $traveller_type as ttype, "
                "  CASE $traveller_type "
                "    WHEN 'couple' THEN (h.comfort_base * 0.35 + h.location_base * 0.35 + h.staff_base * 0.20 + h.facilities_base * 0.10) + "
                "                       (CASE WHEN 'concierge' IN h.facilities_list OR 'laundry' IN h.facilities_list THEN 0.5 ELSE 0 END) "
                "    WHEN 'family' THEN (h.facilities_base * 0.40 + h.cleanliness_base * 0.30 + h.location_base * 0.20 + h.comfort_base * 0.10) + "
                "                       (CASE WHEN 'pool' IN h.facilities_list OR 'breakfast' IN h.facilities_list THEN 0.5 ELSE 0 END) "
                "    WHEN 'business' THEN (h.location_base * 0.40 + h.staff_base * 0.35 + h.comfort_base * 0.15 + h.cleanliness_base * 0.10) + "
                "                        (CASE WHEN 'concierge' IN h.facilities_list THEN 0.5 ELSE 0 END) "
                "    WHEN 'solo' THEN (h.comfort_base * 0.30 + h.location_base * 0.25 + h.value_for_money_base * 0.20 + h.facilities_base * 0.15 + h.cleanliness_base * 0.10) + "
                "                     (CASE WHEN 'gym' IN h.facilities_list OR 'laundry' IN h.facilities_list THEN 0.4 ELSE 0 END) + "
                "                     (CASE WHEN 'concierge' IN h.facilities_list THEN 0.3 ELSE 0 END) "
                "    ELSE h.average_reviews_score "
                "  END as type_score "
                "RETURN h.hotel_id, h.name, h.star_rating, h.cleanliness_base, h.comfort_base, h.facilities_base, h.location_base, h.staff_base, h.value_for_money_base, h.average_reviews_score, h.facilities_list, city, type_score "
                "ORDER BY type_score DESC, h.average_reviews_score DESC "
                "LIMIT 50"
            ),

            # Query 8: Cleanest hotels in a country
            # Example question: "What are the cleanest hotels in France?"
            'cleanest_hotels_by_country': (
                "MATCH (h:Hotel)-[:LOCATED_IN]->(c:City)-[:LOCATED_IN]->(co:Country {name:$country}) "
                "RETURN h.hotel_id, h.name, h.cleanliness_base, c.name as city, h.average_reviews_score "
                "ORDER BY h.cleanliness_base DESC "
                "LIMIT 15"
            ),

            # Query 9: Best located hotels in a country
            # Example question: "What hotels in Italy have the best location?"
            'best_location_by_country': (
                "MATCH (h:Hotel)-[:LOCATED_IN]->(c:City)-[:LOCATED_IN]->(co:Country {name:$country}) "
                "RETURN h.hotel_id, h.name, h.location_base, c.name as city, co.name as country, h.average_reviews_score "
                "ORDER BY h.location_base DESC "
                "LIMIT 15"
            ),

            # Query 10: Most comfortable hotels in a country
            # Example question: "Which hotels in Egypt are the most comfortable?"
            'comfortable_hotels_by_country': (
                "MATCH (h:Hotel)-[:LOCATED_IN]->(c:City)-[:LOCATED_IN]->(co:Country {name:$country}) "
                "RETURN h.hotel_id, h.name, h.comfort_base, c.name as city, co.name as country, h.average_reviews_score "
                "ORDER BY h.comfort_base DESC "
                "LIMIT 15"
            ),

            # Query 11: Get hotel facilities by hotel name
            # Example question: "What are the facilities in The Golden Oasis?"
            'hotel_facilities': (
                "MATCH (h:Hotel {name:$hotel_name})-[:LOCATED_IN]->(c:City)-[:LOCATED_IN]->(co:Country) "
                "OPTIONAL MATCH (h)-[:HAS_FACILITY]->(f:Facility) "
                "RETURN h.hotel_id, h.name, h.facilities_list, COLLECT(f.name) as relationship_facilities, c.name as city, co.name as country "
                "LIMIT 1"
            ),

            # Query 12: Get facilities for all hotels in a country
            # Optionally filters by specific facility keyword (gym, pool, spa, wifi, breakfast, laundry, concierge)
            # Example question: "What are the facilities in hotels in Egypt?" or "Hotels with gym in Egypt?"
            'facilities_by_country': (
                "MATCH (h:Hotel)-[:LOCATED_IN]->(c:City)-[:LOCATED_IN]->(co:Country {name:$country}) "
                "WHERE $facility IS NULL OR $facility IN h.facilities_list "
                "RETURN h.hotel_id, h.name, h.facilities_list, c.name as city, co.name as country "
                "ORDER BY h.name "
                "LIMIT 50"
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
            print(f"[OK] Connected to Neo4j: {uri}")
        except Exception as e:
            print(f"[FAIL] Failed to connect to Neo4j: {e}")
            self.connected = False

    def execute(self, template: str, params: Dict[str, Any], query_number: Optional[int] = None) -> Dict[str, Any]:
        """
        Execute a Cypher query template with parameters.

        Args:
            template: Cypher query template (with $param placeholders)
            params: Dict of parameter values to fill template
            query_number: Query number for reference

        Returns:
            Dict with 'status', 'results', 'error' (if failed)
        """
        if not self.connected or not self.driver:
            print(f"[DEBUG] BaselineRetriever not connected!")
            return {
                'status': 'disconnected',
                'results': [],
                'error': 'Not connected to Neo4j',
                'template': template,
                'params': params,
                'query_number': query_number,
            }

        try:
            print(f"[DEBUG] Executing query with params: {params}")
            print(f"[DEBUG] Template: {template[:150]}...")
            records = []
            with self.driver.session() as session:
                result = session.run(template, **params)
                for record in result:
                    records.append(dict(record))

            print(f"[DEBUG] Query returned {len(records)} results")
            return {
                'status': 'success',
                'results': records,
                'count': len(records),
                'template': template,
                'params': params,
                'query_number': query_number,
            }
        except Exception as e:
            print(f"[DEBUG] Query error: {e}")
            return {
                'status': 'error',
                'results': [],
                'error': str(e),
                'template': template,
                'params': params,
                'query_number': query_number,
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
                cols = [c for c in ['hotel_id','hotel_name','star_rating','average_reviews_score','cleanliness_base','comfort_base','facilities_base','location_base','staff_base','value_for_money_base','city','country'] if c in df.columns]
                df = df[cols].head(50)
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
                cols = [c for c in ['hotel_id','hotel_name','star_rating','average_reviews_score','cleanliness_base','comfort_base','facilities_base','location_base','staff_base','value_for_money_base','city','country'] if c in df.columns]
                results = df[cols].head(50).to_dict(orient='records')
                return {'status': 'success', 'results': results, 'count': len(results)}

            if 'hotel {name:$hotel_name}' in t or 'hotel_name' in t:
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
                    cols = [c for c in ['hotel_id','hotel_name','star_rating','average_reviews_score','cleanliness_base','comfort_base','facilities_base','location_base','staff_base','value_for_money_base','city','country'] if c in df2.columns]
                    results = df2[cols].to_dict(orient='records')
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
    """Enhanced multi-model embedding retriever supporting:
    - Multiple embedding models (MiniLM, MPNet, BGE)
    - Text-based embeddings (hotel descriptions)
    - Feature-based embeddings (numerical attributes)
    - Hybrid text+feature fusion

    If a cache exists for the selected model, it will be loaded;
    otherwise embeddings will be computed and cached.
    """

    # Supported embedding models
    MODELS = {
        'minilm': 'sentence-transformers/all-MiniLM-L6-v2',      # 384-dim, fast, default
        'mpnet': 'sentence-transformers/all-mpnet-base-v2',      # 768-dim, high quality
        'bge': 'BAAI/bge-small-en-v1.5',                         # 384-dim, optimized for retrieval
    }

    def __init__(
        self,
        embedder: Optional[Any] = None,
        csv_dir: str = 'csv',
        cache_dir: str = '.cache',
        embedding_model: str = 'minilm',
        use_features: bool = True,
        feature_weight: float = 0.3,
    ):
        """
        Initialize multi-model embedding retriever.

        Args:
            embedder: Legacy embedder object (for backward compatibility)
            csv_dir: Directory containing CSV files
            cache_dir: Directory for caching embeddings
            embedding_model: Model identifier ('minilm', 'mpnet', 'bge')
            use_features: Whether to include feature-based embeddings
            feature_weight: Weight for feature embeddings (0-1), rest is text weight
        """
        self.embedder = embedder
        self.csv_dir = csv_dir
        self.cache_dir = Path(cache_dir)
        self.embedding_model = embedding_model
        self.use_features = use_features
        self.feature_weight = max(0.0, min(1.0, feature_weight))  # Clamp to [0, 1]
        self.text_weight = 1.0 - self.feature_weight
        
        # Check if embedder is available or if we need to load model
        self.available = embedder is not None and getattr(embedder, 'available', False)
        self.hotel_meta: List[Dict[str, Any]] = []
        self.text_embeddings: Optional[Any] = None
        self.feature_embeddings: Optional[Any] = None
        self.model = None  # For direct model access

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
        """Create a compact text representation for embedding."""
        parts = [row.get('hotel_name', ''), row.get('city', ''), row.get('country', '')]
        # include star rating and key numeric attributes if present
        for k in ['star_rating', 'cleanliness_base', 'comfort_base', 'facilities_base', 'location_base', 'staff_base', 'value_for_money_base']:
            if k in row and row[k]:
                parts.append(f"{k.replace('_',' ')} {row[k]}")
        return ' | '.join([p for p in parts if p])

    def _extract_features(self, row: Dict[str, str]) -> List[float]:
        """Extract numerical feature vector from hotel attributes.
        
        Features include:
        - star_rating (1-5)
        - 6 quality dimensions (cleanliness, comfort, facilities, location, staff, value)
        - average_reviews_score (if available)
        
        Returns:
            8-dimensional feature vector
        """
        features = []
        
        # Star rating (1-5)
        features.append(float(row.get('star_rating', 0.0)))
        
        # Quality dimensions (0-10 scale)
        for attr in ['cleanliness_base', 'comfort_base', 'facilities_base', 
                     'location_base', 'staff_base', 'value_for_money_base']:
            val = row.get(attr, 0.0)
            features.append(float(val) if val else 0.0)
        
        # Average review score (if available, otherwise 0)
        avg_score = row.get('average_reviews_score', 0.0)
        features.append(float(avg_score) if avg_score else 0.0)
        
        return features

    def _load_cache(self) -> bool:
        """Load cached embeddings (text and/or features) for the selected model."""
        # Model-specific cache files
        text_emb_path = self.cache_dir / f'hotel_text_{self.embedding_model}.npz'
        feature_emb_path = self.cache_dir / 'hotel_features.npz'
        meta_path = self.cache_dir / 'hotel_meta.json'
        
        if not meta_path.exists() or np is None:
            return False

        try:
            # Load metadata
            with open(meta_path, 'r', encoding='utf-8') as f:
                self.hotel_meta = json.load(f)
            
            # Load text embeddings
            if text_emb_path.exists():
                data = np.load(text_emb_path, allow_pickle=True)
                self.text_embeddings = data['arr_0']
                print(f"[OK] Loaded {len(self.hotel_meta)} text embeddings ({self.embedding_model}) from cache")
            
            # Load feature embeddings if enabled
            if self.use_features and feature_emb_path.exists():
                data = np.load(feature_emb_path, allow_pickle=True)
                self.feature_embeddings = data['arr_0']
                print(f"[OK] Loaded {len(self.hotel_meta)} feature embeddings from cache")
            
            # Success if we have at least text embeddings
            return self.text_embeddings is not None
        except Exception as e:
            print(f"Warning: failed to load cache: {e}")
            return False

    def _build_and_cache_embeddings(self):
        """Build and cache text and feature embeddings using selected model."""
        hotels_path = Path(self.csv_dir) / 'hotels.csv'
        if not hotels_path.exists():
            raise FileNotFoundError(f"{hotels_path} not found")

        rows: List[Dict[str, str]] = []
        with open(hotels_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)

        if np is None:
            raise RuntimeError('numpy is required for embedding retriever')

        # ===== TEXT EMBEDDINGS =====
        print(f"[INFO] Building text embeddings with model: {self.embedding_model}")
        texts = [self._hotel_text(r) for r in rows]
        
        # Load model if not using legacy embedder
        if not self.embedder or not hasattr(self.embedder, 'model'):
            try:
                from sentence_transformers import SentenceTransformer
                model_name = self.MODELS.get(self.embedding_model, self.MODELS['minilm'])
                self.model = SentenceTransformer(model_name)
                print(f"[OK] Loaded embedding model: {model_name}")
            except Exception as e:
                print(f"[ERROR] Failed to load embedding model: {e}")
                raise
        else:
            self.model = self.embedder.model
        
        # Compute text embeddings in batches
        all_text_vecs = []
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            vecs = self.model.encode(batch, show_progress_bar=False)
            all_text_vecs.append(vecs)
            time.sleep(0.01)
        
        text_emb_array = np.vstack(all_text_vecs)
        self.text_embeddings = text_emb_array
        
        # Save text embeddings
        text_emb_path = self.cache_dir / f'hotel_text_{self.embedding_model}.npz'
        np.savez_compressed(text_emb_path, text_emb_array)
        print(f"[OK] Cached text embeddings: {text_emb_array.shape}")

        # ===== FEATURE EMBEDDINGS =====
        if self.use_features:
            print(f"[INFO] Building feature embeddings from numerical attributes")
            feature_vecs = []
            for r in rows:
                feat_vec = self._extract_features(r)
                feature_vecs.append(feat_vec)
            
            feature_emb_array = np.array(feature_vecs, dtype=float)
            # Normalize feature vectors (zero mean, unit variance per dimension)
            mean = np.mean(feature_emb_array, axis=0, keepdims=True)
            std = np.std(feature_emb_array, axis=0, keepdims=True)
            std[std == 0] = 1.0  # Avoid division by zero
            feature_emb_array = (feature_emb_array - mean) / std
            self.feature_embeddings = feature_emb_array
            
            # Save feature embeddings
            feature_emb_path = self.cache_dir / 'hotel_features.npz'
            np.savez_compressed(feature_emb_path, feature_emb_array)
            print(f"[OK] Cached feature embeddings: {feature_emb_array.shape}")

        # ===== METADATA =====
        self.hotel_meta = []
        for r in rows:
            self.hotel_meta.append({
                'hotel_id': int(r.get('hotel_id')) if r.get('hotel_id') else None,
                'hotel_name': r.get('hotel_name'),
                'city': r.get('city'),
                'country': r.get('country'),
            })
        
        meta_path = self.cache_dir / 'hotel_meta.json'
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(self.hotel_meta, f, ensure_ascii=False, indent=2)
        
        print(f"[OK] Built and cached embeddings for {len(self.hotel_meta)} hotels")

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

    def retrieve(
        self,
        query_embedding: Optional[List[float]],
        query_features: Optional[List[float]] = None,
        top_k: int = 10,
    ) -> Dict[str, Any]:
        """Retrieve hotels using multi-modal similarity (text + features).
        
        Args:
            query_embedding: Text embedding from user query
            query_features: Optional feature vector (numerical attributes)
            top_k: Number of results to return
            
        Returns:
            Dict with status, results, and metadata about retrieval method
        """
        if query_embedding is None:
            return {
                'status': 'no_query_embedding',
                'results': [],
                'message': 'Query embedding missing'
            }

        if self.text_embeddings is None:
            return {
                'status': 'no_embeddings',
                'results': [],
                'message': 'No hotel embeddings available'
            }

        # Compute text similarity
        text_sims = self._cosine_sim(query_embedding, self.text_embeddings)
        
        # Compute feature similarity if available
        feature_sims = None
        if self.use_features and self.feature_embeddings is not None and query_features is not None:
            feature_sims = self._cosine_sim(query_features, self.feature_embeddings)
        
        # Combine scores with weights
        if feature_sims is not None:
            # Hybrid: text_weight * text_sim + feature_weight * feature_sim
            final_sims = [
                self.text_weight * t + self.feature_weight * f
                for t, f in zip(text_sims, feature_sims)
            ]
            retrieval_mode = f'hybrid_text_features (text_w={self.text_weight:.2f}, feat_w={self.feature_weight:.2f})'
        else:
            # Text-only
            final_sims = text_sims
            retrieval_mode = 'text_only'
        
        # Get top-k results
        idxs = sorted(range(len(final_sims)), key=lambda i: final_sims[i], reverse=True)[:top_k]
        results = []
        for i in idxs:
            meta = self.hotel_meta[i].copy()
            meta['score'] = float(final_sims[i])
            meta['text_score'] = float(text_sims[i])
            if feature_sims is not None:
                meta['feature_score'] = float(feature_sims[i])
            results.append(meta)

        return {
            'status': 'success',
            'results': results,
            'count': len(results),
            'embedding_model': self.embedding_model,
            'retrieval_mode': retrieval_mode,
        }


# =====================================================================
# SECTION D: HYBRID RETRIEVER (Baseline + Embeddings)
# =====================================================================
# Combine results from both baseline and embedding approaches

class HybridRetriever:
    """Orchestrate baseline and embedding-based retrieval with weighted merging."""

    def __init__(
        self,
        baseline_retriever: Optional[BaselineRetriever] = None,
        embedding_retriever: Optional[EmbeddingRetriever] = None,
        baseline_weight: float = 0.6,
        embedding_weight: float = 0.4,
    ):
        """
        Initialize hybrid retriever with both components and configurable weights.

        Args:
            baseline_retriever: BaselineRetriever instance
            embedding_retriever: EmbeddingRetriever instance
            baseline_weight: Weight for baseline results (0-1)
            embedding_weight: Weight for embedding results (0-1)
            
        Note:
            Weights don't need to sum to 1; they're normalized during merging.
        """
        self.baseline = baseline_retriever
        self.embedding = embedding_retriever
        self.baseline_weight = max(0.0, baseline_weight)
        self.embedding_weight = max(0.0, embedding_weight)
        # Normalize weights
        total = self.baseline_weight + self.embedding_weight
        if total > 0:
            self.baseline_weight /= total
            self.embedding_weight /= total
        else:
            # Default to equal weights if both are zero
            self.baseline_weight = 0.5
            self.embedding_weight = 0.5

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

        # Build parameters from entities (handles multiple rating filters and countries)
        params = build_cypher_params(entities, intent=intent)
        print(f"[DEBUG] HybridRetriever.retrieve() - intent: {intent}, entities: {entities}")
        print(f"[DEBUG] Built params: {params}")
        
        # Choose template: if rating_types present prefer 'hotel_filter' or 'comfortable_hotels'
        chosen_intent = intent
        rating_types = entities.get('rating_types') or []
        traveller_type = params.get('traveller_type')
        
        # Special handling for facilities_filter intent with hotel_name
        if intent == 'facilities_filter' and params.get('hotel_name'):
            chosen_intent = 'hotel_facilities'
        # Special handling for facilities_filter intent with country (no specific hotel)
        elif intent == 'facilities_filter' and params.get('country') and not params.get('hotel_name'):
            chosen_intent = 'facilities_by_country'
        # Smart template selection based on available entities
        # Include quality-specific filter intents and recommendation
        elif intent in ['hotel_search', 'hotel_filter', 'cleanliness_filter', 'comfort_filter', 'facilities_filter', 'staff_filter', 'value_filter', 'location_filter', 'recommendation']:
            # Check if we have city or country
            has_city = bool(params.get('city'))
            has_country = bool(params.get('country'))
            
            # Special: if traveller_type + country, use hotels_by_country_and_type
            if traveller_type and has_country and not has_city:
                chosen_intent = 'hotels_by_country_and_type'
            # Check if this is a quality-specific query for a country
            elif has_country and not has_city and rating_types:
                # Determine which quality dimension is primary
                if 'cleanliness' in rating_types:
                    chosen_intent = 'cleanest_hotels_by_country'
                elif 'comfort' in rating_types:
                    chosen_intent = 'comfortable_hotels_by_country'
                elif 'value' in rating_types:
                    chosen_intent = 'best_value_by_country'
                elif 'location' in rating_types:
                    chosen_intent = 'best_location_by_country'
                else:
                    chosen_intent = 'hotels_by_country'
            elif rating_types and has_city:
                # City + rating-specific filters -> hotel_filter
                chosen_intent = 'hotel_filter'
            elif rating_types and has_country:
                # Country-level with quality question -> show country results (now includes quality fields)
                chosen_intent = 'hotels_by_country'
            elif has_country and not has_city:
                # Only country provided -> country query
                chosen_intent = 'hotels_by_country'
            elif has_city:
                # City provided without explicit rating filters
                chosen_intent = 'hotel_search'
            # If neither city nor country, keep original intent (will fail gracefully)
        else:
            chosen_intent = intent

        # Run baseline if requested
        if method in ['baseline', 'hybrid']:
            print(f"[DEBUG] Running baseline retrieval with chosen_intent: {chosen_intent}")
            # Always get template and query number regardless of whether baseline exists
            template = CypherTemplates.get_template(chosen_intent)
            query_number = CypherTemplates.QUERY_NUMBERS.get(chosen_intent)
            results['cypher_template'] = template
            results['query_number'] = query_number
            results['chosen_intent'] = chosen_intent
            print(f"[DEBUG] Set cypher_template in results: query_number={query_number}, chosen_intent={chosen_intent}")
            
            if self.baseline:
                # For visa_check intent, check if we need visa_free_destinations query
                if chosen_intent == 'visa_check':
                    # If only from_country is provided (no to_country), use visa_free_destinations
                    if params.get('from_country') and not params.get('to_country'):
                        chosen_intent = 'visa_free_destinations'
                        template = CypherTemplates.get_template(chosen_intent)
                        query_number = CypherTemplates.QUERY_NUMBERS.get(chosen_intent)
                        results['cypher_template'] = template
                        results['query_number'] = query_number
                        results['chosen_intent'] = chosen_intent
                    # If both countries provided, use regular visa_check
                    elif not params.get('from_country'):
                        results['baseline_status'] = 'error'
                        results['baseline_results'] = []
                        results['baseline_error'] = 'Source country is required for visa queries'
                        # Don't execute query, skip to embedding
                    else:
                        # Both countries provided, use regular visa_check
                        pass
                
                # Execute query if we have a valid template and params
                if not results.get('baseline_error'):
                    baseline_result = self.baseline.execute(template, params, query_number=query_number)
                    results['baseline_results'] = baseline_result.get('results', [])
                    results['baseline_status'] = baseline_result.get('status')
                    if baseline_result.get('error'):
                        results['baseline_error'] = baseline_result.get('error')
                else:
                    # Error already set, skip execution
                    pass

        # Run embedding if requested (skip for visa_check - embedding retriever is hotel-specific)
        if method in ['embeddings', 'hybrid'] and intent != 'visa_check':
            if self.embedding:
                print(f"[DEBUG] query_embedding is None: {query_embedding is None}")
                embedding_result = self.embedding.retrieve(query_embedding, top_k=10)
                print(f"[DEBUG] Embedding retrieval returned {len(embedding_result.get('results', []))} results, status: {embedding_result.get('status')}")
                results['embedding_results'] = embedding_result.get('results', [])
                results['embedding_status'] = embedding_result.get('status')
        elif intent == 'visa_check':
            # Skip embedding retrieval for visa queries
            results['embedding_results'] = []
            results['embedding_status'] = 'skipped'

        # Filter embedding results by city if city is specified
        city_val = entities.get('city')
        target_city = None
        if city_val:
            if isinstance(city_val, list):
                target_city = city_val[0].lower() if city_val else None
            else:
                target_city = city_val.lower() if city_val else None
        
        if target_city and results['embedding_results']:
            # Filter embedding results to only include hotels in the specified city
            print(f"[DEBUG] Filtering {len(results['embedding_results'])} embedding results by city: {target_city}")
            filtered_embedding = []
            for item in results['embedding_results']:
                item_city = item.get('city', '').lower() if item.get('city') else ''
                if target_city in item_city or item_city in target_city:
                    filtered_embedding.append(item)
            print(f"[DEBUG] After city filtering: {len(filtered_embedding)} results remain")
            results['embedding_results'] = filtered_embedding

        # Merge results if hybrid (skip merging for visa_check - different result types)
        if method == 'hybrid' and intent != 'visa_check':
            # Weighted merge: rank hotels by combining normalized scores from both methods
            merged = self._weighted_merge(
                results['baseline_results'],
                results['embedding_results'],
                top_k=50  # Merge more results, will be filtered by caller if needed
            )
            results['merged_results'] = merged
            results['merge_method'] = f'weighted (baseline={self.baseline_weight:.2f}, embedding={self.embedding_weight:.2f})'
        elif method == 'hybrid' and intent == 'visa_check':
            # For visa queries, merged_results should just be baseline_results (no hotels)
            results['merged_results'] = results['baseline_results']
            results['merge_method'] = 'baseline_only'

        return results

    def _weighted_merge(
        self,
        baseline_results: List[Dict[str, Any]],
        embedding_results: List[Dict[str, Any]],
        top_k: int = 50,
    ) -> List[Dict[str, Any]]:
        """Merge and rank results from baseline and embedding methods using weighted scoring.
        
        Args:
            baseline_results: Results from Cypher queries
            embedding_results: Results from embedding similarity
            top_k: Maximum number of merged results to return
            
        Returns:
            List of merged results, sorted by weighted score
        """
        # Build hotel score dict: hotel_id -> {data, baseline_rank, embedding_rank, scores}
        hotel_scores: Dict[Any, Dict[str, Any]] = {}
        
        # Process baseline results (rank-based scoring: 1.0 for rank 1, decaying)
        for rank, item in enumerate(baseline_results, start=1):
            hotel_id = self._extract_hotel_id(item)
            if hotel_id:
                baseline_score = 1.0 / rank  # Reciprocal rank scoring
                hotel_scores[hotel_id] = {
                    'data': item,
                    'baseline_score': baseline_score,
                    'embedding_score': 0.0,
                    'baseline_rank': rank,
                    'embedding_rank': None,
                }
        
        # Process embedding results (use similarity scores directly)
        for rank, item in enumerate(embedding_results, start=1):
            hotel_id = self._extract_hotel_id(item)
            if hotel_id:
                # Use cosine similarity score if available, else reciprocal rank
                embedding_score = item.get('score', 1.0 / rank)
                
                if hotel_id in hotel_scores:
                    # Hotel appears in both results - update with embedding info
                    hotel_scores[hotel_id]['embedding_score'] = embedding_score
                    hotel_scores[hotel_id]['embedding_rank'] = rank
                else:
                    # Hotel only in embedding results
                    hotel_scores[hotel_id] = {
                        'data': item,
                        'baseline_score': 0.0,
                        'embedding_score': embedding_score,
                        'baseline_rank': None,
                        'embedding_rank': rank,
                    }
        
        # Compute weighted final scores
        for hotel_id, info in hotel_scores.items():
            final_score = (
                self.baseline_weight * info['baseline_score'] +
                self.embedding_weight * info['embedding_score']
            )
            info['final_score'] = final_score
        
        # Sort by final score (descending) and return top-k
        ranked = sorted(
            hotel_scores.values(),
            key=lambda x: x['final_score'],
            reverse=True
        )[:top_k]
        
        # Build output list with score metadata
        merged = []
        for info in ranked:
            result = info['data'].copy()
            result['hybrid_score'] = info['final_score']
            result['baseline_contribution'] = info['baseline_score'] * self.baseline_weight
            result['embedding_contribution'] = info['embedding_score'] * self.embedding_weight
            if info['baseline_rank']:
                result['baseline_rank'] = info['baseline_rank']
            if info['embedding_rank']:
                result['embedding_rank'] = info['embedding_rank']
            merged.append(result)
        
        return merged

    def _extract_hotel_id(self, item: Dict[str, Any]) -> Any:
        """Extract hotel_id from a result item (handles various formats)."""
        # Try multiple ways to get hotel_id (Neo4j returns keys like 'h.hotel_id' with dots)
        hotel_id = (
            item.get('hotel_id') or
            item.get('h.hotel_id') or
            (item.get('h', {}).get('hotel_id') if isinstance(item.get('h'), dict) else None)
        )
        
        # If no hotel_id, try to use hotel name as key
        if not hotel_id:
            hotel_name = (
                item.get('hotel_name') or
                item.get('h.name') or
                (item.get('h', {}).get('name') if isinstance(item.get('h'), dict) else None) or
                item.get('name')
            )
            if hotel_name:
                hotel_id = f"name_{hotel_name}"
        
        return hotel_id


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

def build_cypher_params(entities: Dict[str, Optional[Any]], intent: Optional[str] = None) -> Dict[str, Any]:
    """Build a parameter dict for Cypher templates from extracted entities.

    Supports mapping multiple rating types (e.g., cleanliness, comfort)
    into the expected template parameter names.
    Also handles multiple countries for visa queries.

    Args:
        entities: Extracted entities from preprocessing (may include keys:
                  'city', 'country', 'hotel', 'traveller_type', 'min_rating',
                  'rating_types' (list), 'date')
                  Note: 'city' and 'country' can be lists or single values
        intent: User intent (e.g., 'visa_check') to determine special parameter mapping

    Returns:
        Dict of parameters to pass to BaselineRetriever.execute()
    """
    params: Dict[str, Any] = {}

    if not entities:
        return params

    # Handle city - if list, use first one (or all if template supports it)
    city_val = entities.get('city')
    if city_val:
        if isinstance(city_val, list):
            params['city'] = city_val[0]  # Use first city for now
        else:
            params['city'] = city_val

    # Handle country - special handling for visa_check intent
    country_val = entities.get('country')
    city_val = entities.get('city')
    
    # Special handling for visa_check: if city is provided with country, map city to destination country
    if intent == 'visa_check' and city_val and country_val:
        # We have both city and country for visa query
        # Country is source (from_country), city's country is destination (to_country)
        if isinstance(city_val, list):
            city_val = city_val[0]
        if isinstance(country_val, list):
            country_val = country_val[0]
        
        # Map city to its country (lookup in city_country mapping)
        dest_country = _city_to_country_map.get(city_val.lower()) if city_val else None
        if dest_country:
            params['from_country'] = country_val
            params['to_country'] = dest_country
        else:
            # If city-to-country mapping not found, fall back to original logic
            params['from_country'] = country_val
    elif country_val:
        if intent == 'visa_check':
            # For visa queries, map multiple countries to from_country and to_country
            if isinstance(country_val, list) and len(country_val) >= 2:
                # Use first as from, second as to (order preserved from extraction)
                params['from_country'] = country_val[0]
                params['to_country'] = country_val[1]
            elif isinstance(country_val, list) and len(country_val) == 1:
                # Only one country found - treat as from_country (for visa_free_destinations query)
                params['from_country'] = country_val[0]
                # to_country will be None, which triggers visa_free_destinations query
            else:
                # Single country value - treat as from_country (for visa_free_destinations query)
                params['from_country'] = country_val
                # to_country will be None, which triggers visa_free_destinations query
        else:
            # For other intents, use first country if list
            if isinstance(country_val, list):
                params['country'] = country_val[0]
            else:
                params['country'] = country_val

    if entities.get('hotel'):
        # templates expect $hotel_name
        params['hotel_name'] = entities['hotel']
    if entities.get('traveller_type'):
        params['traveller_type'] = entities['traveller_type']
    if entities.get('facility'):
        # Add facility parameter (gym, pool, spa, wifi, breakfast, laundry, concierge)
        params['facility'] = entities['facility']
    else:
        # If no facility specified, set to None for optional matching
        params['facility'] = None
    if entities.get('date'):
        params['target_date'] = entities['date']

    # Default numeric thresholds
    min_rating = entities.get('min_rating')
    # If user hinted at quality (rating_types) but didn't give a number, default to 4.0
    if min_rating is None:
        if entities.get('rating_types'):
            min_rating = 4.0
        else:
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

    # Always populate star threshold (hotel_filter query uses it)
    params['min_star'] = float(min_rating)

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
