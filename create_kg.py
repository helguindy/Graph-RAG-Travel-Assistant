import pandas as pd
from neo4j import GraphDatabase
import sys
import os
import re
from collections import defaultdict


# Canonical facility keywords grouped by facility type
FACILITY_KEYWORDS = {
    'wifi': ['wifi', 'wi-fi', 'wireless', 'broadband', 'internet', 'lan', 'connection'],
    'gym': ['gym', 'fitness', 'exercise', 'workout', 'treadmill', 'weight', 'training'],
    'pool': ['pool', 'swimming', 'swim', 'hot tub', 'jacuzzi', 'spa pool', 'water'],
    'spa': ['spa', 'sauna', 'massage', 'wellness', 'steam room', 'mud', 'therapy'],
    'parking': ['parking', 'garage', 'valet', 'lot', 'carport', 'vehicle', 'car'],
    'breakfast': ['breakfast', 'buffet', 'restaurant', 'dining', 'bar', 'cafe', 'coffee', 'meal'],
    'laundry': ['laundry', 'dry clean', 'washing', 'ironing', 'linen'],
    'concierge': ['concierge', 'front desk', 'reception', 'service desk', 'help desk'],
    'room_service': ['room service', 'in-room', 'in room', 'delivery', 'minibar'],
    'ac': ['air condition', 'ac', 'climate', 'cooling', 'heating'],
    'tv': ['tv', 'television', 'cable', 'streaming', 'entertainment'],
    'safe': ['safe', 'safety', 'security', 'vault'],
    'balcony': ['balcony', 'terrace', 'patio', 'veranda', 'outdoor'],
    'elevator': ['elevator', 'lift', 'escalator'],
    'pet_friendly': ['pet friendly', 'pet', 'dog', 'cat', 'animal'],
    'business': ['business center', 'conference', 'meeting', 'work space'],
    'garden': ['garden', 'outdoor', 'landscape', 'green', 'scenic'],
}


def extract_facilities_from_text(text):
    if not text or pd.isna(text):
        return set()

    text = str(text).lower()
    found = set()
    for facility, keywords in FACILITY_KEYWORDS.items():
        for keyword in keywords:
            if re.search(r'\b' + re.escape(keyword) + r'\b', text):
                found.add(facility)
                break
    return found


def aggregate_hotel_facilities(reviews_df):
    by_hotel = defaultdict(lambda: defaultdict(int))
    for _, row in reviews_df.iterrows():
        hotel_id = row['hotel_id']
        for facility in extract_facilities_from_text(row['review_text']):
            by_hotel[hotel_id][facility] += 1

    aggregated = {}
    for hotel_id, counts in by_hotel.items():
        sorted_facilities = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        aggregated[hotel_id] = [name for name, _ in sorted_facilities]
    return aggregated


def attach_facilities_to_hotels(hotels_df, hotel_facilities):
    hotels_df = hotels_df.copy()
    hotels_df['facilities'] = hotels_df['hotel_id'].map(lambda x: '|'.join(hotel_facilities.get(x, [])))
    return hotels_df


def read_config(config_file='config.txt'):
    ## Read Neo4j configuration from config.txt
    config = {}
    try:
        with open(config_file, 'r') as f:
            for line in f:
                if '=' in line:
                    key, value = line.strip().split('=', 1)
                    config[key] = value
        return config['URI'], config['USERNAME'], config['PASSWORD']
    except Exception as e:
        print(f"Error reading config file: {e}")
        sys.exit(1)


class KnowledgeGraphBuilder:
    def __init__(self, uri, username, password):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))

    def close(self):
        self.driver.close()

    def clear_database(self):
        ## Clear all nodes and relationships
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            print("Database cleared")

    def create_constraints(self):
        ## Create unique constraints for node IDs
        with self.driver.session() as session:
            constraints = [
                "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Traveller) REQUIRE t.user_id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (h:Hotel) REQUIRE h.hotel_id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (r:Review) REQUIRE r.review_id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (c:City) REQUIRE c.name IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (co:Country) REQUIRE co.name IS UNIQUE"
            ]
            for constraint in constraints:
                session.run(constraint)
            print("Constraints created")


    def load_countries(self, users_df, hotels_df):
        # Create Country nodes from users and hotels data
        countries = set(users_df['country'].unique()) | set(hotels_df['country'].unique())

        with self.driver.session() as session:
            for country in countries:
                session.run("""
                    MERGE (c:Country {name: $name})
                """, name=country)
        print(f"Loaded {len(countries)} countries")


    def load_cities(self, hotels_df):
        # Create City nodes and relationships to Countries
        cities = hotels_df[['city', 'country']].drop_duplicates()

        with self.driver.session() as session:
            for _, row in cities.iterrows():
                session.run("""
                    MERGE (city:City {name: $city_name})
                    MERGE (country:Country {name: $country_name})
                    MERGE (city)-[:LOCATED_IN]->(country)
                """, city_name=row['city'], country_name=row['country'])
        print(f"Loaded {len(cities)} cities")


    def load_hotels(self, hotels_df):
        # Create Hotel nodes and relationships to Cities
        with self.driver.session() as session:
            for _, hotel in hotels_df.iterrows():
                session.run("""
                    MERGE (h:Hotel {
                        hotel_id: $hotel_id,
                        name: $name,
                        star_rating: $star_rating,
                        cleanliness_base: $cleanliness_base,
                        comfort_base: $comfort_base,
                        facilities_base: $facilities_base,
                        location_base: $location_base,
                        staff_base: $staff_base,
                        value_for_money_base: $value_for_money_base,
                        average_reviews_score: 0.0
                    })
                    MERGE (c:City {name: $city})
                    MERGE (h)-[:LOCATED_IN]->(c)
                """,
                    hotel_id=int(hotel['hotel_id']),
                    name=hotel['hotel_name'],
                    star_rating=float(hotel['star_rating']),
                    cleanliness_base=float(hotel['cleanliness_base']),
                    comfort_base=float(hotel['comfort_base']),
                    facilities_base=float(hotel['facilities_base']),
                    location_base=float(hotel.get('location_base', 0.0)),
                    staff_base=float(hotel.get('staff_base', 0.0)),
                    value_for_money_base=float(hotel.get('value_for_money_base', 0.0)),
                    city=hotel['city']
                )
        print(f"Loaded {len(hotels_df)} hotels")


    def load_travellers(self, users_df):
        # Create Traveller nodes and relationships to Countries 
        with self.driver.session() as session:
            for _, user in users_df.iterrows():
                age_group = user['age_group']
                if '-' in str(age_group):
                    age = int(str(age_group).split('-')[0])
                elif str(age_group).strip().endswith('+'):
                    age = int(str(age_group).strip().replace('+', ''))
                else:
                    try:
                        age = int(age_group)
                    except Exception:
                        age = None

                session.run("""
                    MERGE (t:Traveller {
                        user_id: $user_id,
                        age: $age,
                        age_group: $age_group,
                        type: $type,
                        gender: $gender
                    })
                    MERGE (c:Country {name: $country})
                    MERGE (t)-[:FROM_COUNTRY]->(c)
                """,
                    user_id=int(user['user_id']),
                    age=age,
                    age_group=str(user['age_group']),
                    type=user['traveller_type'],
                    gender=user['user_gender'],
                    country=user['country']
                )
        print(f"Loaded {len(users_df)} travellers")

    def load_reviews(self, reviews_df):
        # Create Review nodes and relationships
        with self.driver.session() as session:
            for _, review in reviews_df.iterrows():
                session.run("""
                    MERGE (r:Review {
                        review_id: $review_id,
                        text: $text,
                        date: $date,
                        score_overall: $score_overall,
                        score_cleanliness: $score_cleanliness,
                        score_comfort: $score_comfort,
                        score_facilities: $score_facilities,
                        score_location: $score_location,
                        score_staff: $score_staff,
                        score_value_for_money: $score_value_for_money
                    })
                    MERGE (t:Traveller {user_id: $user_id})
                    MERGE (h:Hotel {hotel_id: $hotel_id})
                    MERGE (t)-[:WROTE]->(r)
                    MERGE (r)-[:REVIEWED]->(h)
                    MERGE (t)-[:STAYED_AT]->(h)
                """,
                    review_id=int(review['review_id']),
                    text=str(review.get('review_text', '')),
                    date=str(review['review_date']),
                    score_overall=float(review['score_overall']),
                    score_cleanliness=float(review['score_cleanliness']),
                    score_comfort=float(review['score_comfort']),
                    score_facilities=float(review['score_facilities']),
                    score_location=float(review['score_location']),
                    score_staff=float(review['score_staff']),
                    score_value_for_money=float(review['score_value_for_money']),
                    user_id=int(review['user_id']),
                    hotel_id=int(review['hotel_id'])
                )
        print(f"Loaded {len(reviews_df)} reviews")

    def load_facilities(self, hotel_facilities):
        # Create Facility nodes and HAS_FACILITY relationships
        facilities = sorted({f for items in hotel_facilities.values() for f in items})
        with self.driver.session() as session:
            for facility in facilities:
                session.run("MERGE (f:Facility {name: $name})", name=facility)

            for hotel_id, facilities_list in hotel_facilities.items():
                session.run(
                    "MATCH (h:Hotel {hotel_id: $hotel_id}) SET h.facilities_list = $facilities_list",
                    hotel_id=int(hotel_id),
                    facilities_list=facilities_list,
                )
                for facility in facilities_list:
                    session.run(
                        "MATCH (h:Hotel {hotel_id: $hotel_id}) MATCH (f:Facility {name: $facility}) MERGE (h)-[:HAS_FACILITY]->(f)",
                        hotel_id=int(hotel_id),
                        facility=facility,
                    )
        print(f"Loaded {len(facilities)} facilities across {len(hotel_facilities)} hotels")

    def update_hotel_average_scores(self):
        # Update average_reviews_score for all hotels
        with self.driver.session() as session:
            session.run("""
                MATCH (h:Hotel)<-[:REVIEWED]-(r:Review)
                WITH h, AVG(r.score_overall) as avg_score
                SET h.average_reviews_score = avg_score
            """)
        print("Updated hotel average scores")

    def load_visa_requirements(self, visa_df):
        # Create NEEDS_VISA relationships between countries
        with self.driver.session() as session:
            for _, visa in visa_df.iterrows():
                requires = str(visa['requires_visa']).strip().lower()
                if requires in ('yes', 'y', 'true'):
                    session.run("""
                        MATCH (from:Country {name: $from_country})
                        MATCH (to:Country {name: $to_country})
                        MERGE (from)-[v:NEEDS_VISA]->(to)
                        SET v.visa_type = $visa_type
                    """,
                        from_country=visa['from'],
                        to_country=visa['to'],
                        visa_type=str(visa.get('visa_type', ''))
                    )
                elif requires in ('no', 'n', 'false'):
                    session.run("""
                        MATCH (from:Country {name: $from_country})
                        MATCH (to:Country {name: $to_country})
                        MERGE (from)-[v:VISA_FREE]->(to)
                        SET v.visa_type = coalesce($visa_type, 'No visa required')
                    """,
                        from_country=visa['from'],
                        to_country=visa['to'],
                        visa_type=str(visa.get('visa_type', 'No visa required'))
                    )
        print(f"Loaded visa requirements")


def build_knowledge_graph(uri, username, password):
    ## Load CSV files and build knowledge graph
    try:
        hotels_df = pd.read_csv('csv/hotels.csv')
        reviews_df = pd.read_csv('csv/reviews.csv', engine='python')
        users_df = pd.read_csv('csv/users.csv')
        visa_df = pd.read_csv('csv/visa.csv')
    except Exception as e:
        print(f"Error loading CSV files: {e}")
        sys.exit(1)

    hotel_facilities = aggregate_hotel_facilities(reviews_df)
    hotels_df = attach_facilities_to_hotels(hotels_df, hotel_facilities)

    kg_builder = KnowledgeGraphBuilder(uri, username, password)
    try:
        kg_builder.clear_database()
        kg_builder.create_constraints()
        kg_builder.load_countries(users_df, hotels_df)
        kg_builder.load_cities(hotels_df)
        kg_builder.load_hotels(hotels_df)
        kg_builder.load_travellers(users_df)
        kg_builder.load_reviews(reviews_df)
        kg_builder.load_facilities(hotel_facilities)
        kg_builder.update_hotel_average_scores()
        kg_builder.load_visa_requirements(visa_df)
        print("\nKnowledge Graph created successfully!")
    except Exception as e:
        print(f"Error building knowledge graph: {e}")
        import traceback
        traceback.print_exc()
    finally:
        kg_builder.close()


def main():
    uri, username, password = read_config('config.txt')
    build_knowledge_graph(uri, username, password)


if __name__ == '__main__':
    main()

