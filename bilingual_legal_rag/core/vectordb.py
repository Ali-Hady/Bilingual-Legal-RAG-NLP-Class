import lancedb
from lancedb.pydantic import LanceModel, Vector
from pathlib import Path
import numpy as np
from bilingual_legal_rag.core.chunker import ChunkingManager
from sentence_transformers import SentenceTransformer
import json 


# embedding model
CONTEXT_LIMIT = 512
NDIMS = 384
embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
# bge-small


class EnglishChunk(LanceModel):
    doc_id: int
    act_id: str
    chunk_index: int 
    text: str #chunk string
    vector: Vector(NDIMS)  # type: ignore 384


class ArabicChunk(LanceModel):
    doc_id: int
    page_num: int
    chunk_index: int 
    text: str #chunk string
    vector: Vector(NDIMS)  # type: ignore 384


class Embedder:
    def embed(self, texts: list[str]) -> list[np.ndarray]:
        embeddings = []
        response = embedding_model.encode(texts)
        for vector in response:
            embeddings.append(np.array(vector))
        return embeddings


class LanceManager:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            db_path = Path(__file__).parent.parent.parent
            #db_path.parent.mkdir(parents=True, exist_ok=True)
            cls._instance.db = lancedb.connect(db_path / "lancedb_data")
            cls._instance.embedder = Embedder()
        return cls._instance


    def _get_or_create_table(self):
            table_map = {}

            if "english_library" in self.db.table_names():
                table = self.db.open_table("english_library")
            else:
                table = self.db.create_table(
                    "english_library",
                    schema=EnglishChunk
                )
            table_map["english_library"] = table

            if "arabic_library" in self.db.table_names():
                table = self.db.open_table("arabic_library")
            else:
                table = self.db.create_table(
                    "arabic_library",
                    schema=ArabicChunk
                )
            table_map["arabic_library"] = table

            return table_map
        

    def get_top_chunks(self, query: str,lang:str) -> str:

        tables = self._get_or_create_table()

        if lang == "en":
            table = tables["english_library"]
        else:
            table = tables["arabic_library"]


        query_vector = self.embedder.embed([query])[0]
        top_chunks = (
            table.search(query_vector)
            .limit(8)
            .to_list()
        )

        if not top_chunks:
            return f"No relevant content found for query: {query}"

        res = [f"--- Chunk {c['chunk_index']} ---\n{c['text']}" for c in top_chunks]
        return f"Relevant context from the file:\n\n{'\n\n'.join(res)}"


    def index_dataset(self, docs:list[str], lang: str):
        tables = self._get_or_create_table()

        if lang == "en":
            table = tables["english_library"]
        else:
            table = tables["arabic_library"]

        chunker = ChunkingManager(
            model=embedding_model,
            context_len=CONTEXT_LIMIT
        )

        chunks_info = []

        for doc_id, doc in enumerate(docs):
            chunks = chunker.chunk_doc(doc=doc, doc_type=lang)
            
            if chunks:
                vectors = self.embedder.embed(chunks)
            else:
                vectors = []

            for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
                if lang == "en":
                    chunks_info.append({
                        "doc_id": doc_id,
                        "act_id": doc["act_id"],
                        "chunk_index": i,
                        "text": chunk,
                        "vector": vector.tolist()
                    })
                elif lang == "ar":
                    chunks_info.append({
                        "doc_id": doc_id,
                        "page_num": doc["page_number"],
                        "chunk_index": i,
                        "text": chunk,
                        "vector": vector.tolist()
                    })

        table.add(chunks_info)