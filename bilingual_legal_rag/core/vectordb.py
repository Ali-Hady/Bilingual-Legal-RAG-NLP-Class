import lancedb
from lancedb.pydantic import LanceModel, Vector
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from bilingual_legal_rag.core.chunker import ChunkingManager
from sentence_transformers import SentenceTransformer
import lancedb 
import json 


# embedding model
CONTEXT_LIMIT = 512
NDIMS = 384
embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
# bge-small


class TextChunk(LanceModel):
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
            # BASE_DIR = Path(__file__).resolve().parent.parent.parent
            # db_path = BASE_DIR / ".memory" / "lancedb"
            db_path = Path("data/lancedb")
            db_path.parent.mkdir(parents=True, exist_ok=True)
            cls._instance.db = lancedb.connect(str(db_path))
            cls._instance.embedder = Embedder()
        return cls._instance

    def _get_or_create_table(self):
            table_map = {}

            for table_name in ["english_library", "arabic_library"]:

                if table_name in self.db.table_names():
                    table = self.db.open_table(table_name)
                else:
                    table = self.db.create_table(
                        table_name,
                        schema=TextChunk
                    )
                table_map[table_name] = table

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


    def index_dataset(self, path:str, lang: str):
        
        p = Path(path)
        if not p.exists :
            raise FileNotFoundError("file not fount")
        
        with open (p,"r",encoding="utf-8") as f:
                    docs = json.load(f)

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

            for i, chunk in enumerate(chunks):
                vector = self.embedder.embed([chunk])[0]

                chunks_info.append({
                    "doc_id": doc_id,
                    "chunk_index": i,
                    "text": chunk,
                    "vector": vector.tolist()
                })

        table.add(chunks_info)