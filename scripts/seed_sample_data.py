import sys
import os

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.embedder import embedder
from app.services.vector_service import vector_service

sample_data = [
    {
        "id": "product_1",
        "text": "Example product: A three-room property with a swimming pool, breakfast, and high-speed Wi-Fi. Price starts at $120 per night."
    },
    {
        "id": "service_1",
        "text": "Example service: A full-day guided experience starting at 5:00 AM with professional guides and private transportation."
    },
    {
        "id": "faq_1",
        "text": "Example FAQ: Customers can review the menu, opening hours, accessibility information, and available payment methods before booking."
    }
]

print("Vectorizing sample documents...")
vector_service.ensure_collection()
texts = [item["text"] for item in sample_data]
ids = [item["id"] for item in sample_data]
embeddings = embedder.embed_documents(texts)

print("Saving vectors into Qdrant Cloud...")
vector_service.upsert_documents(ids=ids, documents=texts, embeddings=embeddings)
print("Seeding complete! Qdrant now contains initial sample data.")
