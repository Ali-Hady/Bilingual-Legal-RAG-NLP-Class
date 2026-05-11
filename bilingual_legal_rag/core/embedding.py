from pymongo import MongoClient
from bilingual_legal_rag.app.config import settings
from bilingual_legal_rag.core.rag import embed_collection

from bilingual_legal_rag.core.client import LegalGenerator 

def run_embedding():
    print("Connecting to databases...")
    
    mongo_client = MongoClient(settings.MONGO_URI)
    db = mongo_client[settings.DB_NAME]
    english_collection = db[settings.ENGLISH_COLLECTION]
    arabic_collection = db[settings.ARABIC_COLLECTION]
    
    lm_client = LegalGenerator(host=settings.OLLAMA_HOST, model=settings.OLLAMA_MODEL)

    try: 
        print("Starting English chunking and embedding...")
        embed_collection(english_collection, "en", lm_client)
        print("English Laws chunked and stored in vectorDB")
    except Exception as e:
        print(f"Failed to chunk and embed English collection: {e}")

    try:
        print("Starting Arabic chunking and embedding...")
        embed_collection(arabic_collection, "ar", lm_client)
        print("Arabic Laws chunked and stored in vectorDB")
    except Exception as e:
        print(f"Failed to chunk and embed Arabic collection: {e}")
        
    finally:
        mongo_client.close()

if __name__ == "__main__":
    run_embedding()