import uuid
import pandas as pd
import numpy as np
import math
from datasets import load_dataset
from qdrant_client import models, QdrantClient
from sentence_transformers import SentenceTransformer

class QdrantPoint:
    def __init__(self, id, vector, payload):
        self.id = id
        self.vector = vector
        self.payload = payload


def load_and_encode_data(dataset_id, encoder):
    """Loads the dataset and encodes hotel descriptions."""
    dataset = load_dataset(dataset_id)
    data = [entry for entry in dataset['train'] if entry['hotel_description'] is not None]
    descriptions = [entry['hotel_description'] for entry in data]
    vectors = encoder.encode(descriptions, show_progress_bar=True).tolist()
    return data, vectors

def create_qdrant_records(data, vectors):
    """Creates Qdrant records from data and encoded vectors."""
    records = []
    for entry, vector in zip(data, vectors):
        record_id = str(uuid.uuid4())
        payload = {key: (None if isinstance(value, float) and math.isnan(value) else value) for key, value in entry.items()}
        record = models.Record(id=record_id, vector=vector, payload=payload)
        records.append(record)
    return records

def upload_records_to_qdrant(qdrant_client, collection_name, records):
    points = [record for record in records]
    qdrant_client.upload_points(collection_name=collection_name, points=points)


def search_in_qdrant(qdrant_client, collection_name, search_query, encoder, limit=3):
    query_vector = encoder.encode([search_query]).tolist()[0]

    search_results = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=limit * 10, 
    )

    print("Initial number of hits:", len(search_results))

    unique_hotel_names = set()
    unique_hits = []

    for hit in search_results:
        payload = hit.payload if hasattr(hit, 'payload') else {}
        
        hotel_name = payload.get('hotel_name') if 'hotel_name' in payload else None

        if hotel_name and hotel_name not in unique_hotel_names:
            unique_hotel_names.add(hotel_name)
            unique_hits.append(hit)
            if len(unique_hits) == limit:
                break

    print("Unique Search Results:")
    for hit in unique_hits:
        print(hit.payload.get('hotel_name') if hasattr(hit, 'payload') and 'hotel_name' in hit.payload else "No Name")
        print()

    return unique_hits

def list_data_dictionaries(unique_hits):
    """Converts unique hits into a list of data dictionaries with more detailed hotel information."""
    data_dictionaries = []

    for hit in unique_hits:
        payload = hit.payload if hasattr(hit, 'payload') else {}
        
        data_dict = {
            'hotel_name': payload.get('hotel_name', 'No Name'),
            'hotel_description': payload.get('hotel_description', 'No Description'),
            'hotel_image': payload.get('hotel_image', 'No Image Available'),
            'price_range': payload.get('price_range', 'No Price Range Available'),
            'rating_value': payload.get('rating_value', 'No Rating Available'),
            'review_count': payload.get('review_count', 'No Review Count Available'),
            'street_address': payload.get('street_address', 'No Street Address Available'),
            'locality': payload.get('locality', 'No Locality Available'),
            'country': payload.get('country', 'No Country Available'),
            'hotel_url': payload.get('hotel_url','No URL Available'),
        }
        
        data_dictionaries.append(data_dict)
    
    return data_dictionaries


def main(text):
    dataset_id = "traversaal-ai-hackathon/hotel_datasets"
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    qdrant_client = QdrantClient(host="127.0.0.1", port=6333)
    collection_name = "hotel_collection"

    qdrant_client.recreate_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=encoder.get_sentence_embedding_dimension(), distance=models.Distance.COSINE),
    )

    data, vectors = load_and_encode_data(dataset_id, encoder)
    records = create_qdrant_records(data, vectors)
    upload_records_to_qdrant(qdrant_client, collection_name, records)

    search_query = text
    unique_hits = search_in_qdrant(qdrant_client, collection_name, search_query, encoder, limit=3)

    data_dictionaries = list_data_dictionaries(unique_hits)
    return data_dictionaries

    