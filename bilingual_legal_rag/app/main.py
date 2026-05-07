from contextlib import asynccontextmanager
from fastapi import FastAPI
from pymongo import MongoClient
from bilingual_legal_rag.app.config import settings # config.py
import json
from bilingual_legal_rag.app.routers import laws, search, rag
from bilingual_legal_rag.core.rag import embed_collection

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("RAG App booting...")

    client = MongoClient(settings.MONGO_URI)
    db = client[settings.DB_NAME]
    english_collection = db[settings.ENGLISH_COLLECTION]
    arabic_collection = db[settings.ARABIC_COLLECTION]

    english_doc_cnt = english_collection.count_documents({})
    arabic_doc_cnt = arabic_collection.count_documents({})

    if english_doc_cnt == 0:
        print("No English Laws Stored, checking seed...")
        if settings.ENGLISH_SEED.exists():
            with open(settings.ENGLISH_SEED, "r", encoding="utf-8") as f:
                laws_data = json.load(f)
                
                if laws_data:
                    english_collection.insert_many(laws_data)
                    print("Created Collection of English Laws in MongoDB")
                    try:
                        embed_collection(english_collection, "en")
                        print("English Laws chunked and stored in vectorDB")
                    except:
                        print("Failed to chunk and embed English collection")
                else:
                    print("No laws found in english json seed")
        else:
            print("No english JSON seed file found")

    else:
        print("English Laws found in MongoDB")    
    

    if arabic_doc_cnt == 0:
        print("No Arabic Laws Stored, checking seed...")
        if settings.ARABIC_SEED.exists():
            with open(settings.ARABIC_SEED, "r", encoding="utf-8") as f:
                laws_data = json.load(f)
                
                if laws_data:
                    arabic_collection.insert_many(laws_data)
                    print("Created Collection of Arabic Laws in MongoDB")
                    try:
                        embed_collection(arabic_collection, "ar")
                        print("Arabic Laws chunked and stored in vectorDB")
                    except:
                        print("Failed to chunk and embed Arabic collection")
                else:
                    print("No laws found in Arabic json seed")
        else:
            print("No Arabic JSON seed file found")

    else:
        print("Arabic Laws found in MongoDB")

    
    app.state.mongo_client = client
    app.state.db = db

    yield

    print("Shutting Down API, cleaning DB Connection")
    try:
        app.state.mongo_client.close()
    except:
        print("Error Closing Connection")


app = FastAPI(lifespan=lifespan, title="Bilingual Legal RAG API", version="0.1.0",
    description="API for English and Arabic legal document retrieval and RAG.")


app.include_router(laws.router)
app.include_router(rag.router)

@app.get('/')
async def root():
    return {
        "Message" : "Hello, We are the Raggers"
    }