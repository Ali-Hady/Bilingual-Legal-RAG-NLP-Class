import lancedb
from lancedb.pydantic import LanceModel, Vector
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from bilingual_legal_rag.core.chunker import ChunkingManager


# embedding model
#BASE_DIR = Path(__file__).resolve().parent.parent.parent
config_pth = find_config()
config, metadata = parse_embedding_config(config_pth=config_pth)
CONTEXT_LIMIT = metadata["context_limit"]
NDIMS = metadata["ndims"] 

embed_model = Llama(**config)
# bge-small


class TextChunk(LanceModel):
    chunk_index: int 
    last_accessed: datetime
    text: str
    vector: Vector(NDIMS)  # type: ignore 784


class Embedder:
    def embed(self, texts: list[str]) -> list[np.ndarray]:
        embeddings = []
        for text in texts:
            response = embed_model.create_embedding(text)
            vector = response["data"][0]["embedding"]
            embeddings.append(np.array(vector))
        return embeddings


class LanceManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            BASE_DIR = Path(__file__).resolve().parent.parent.parent
            db_path = str(BASE_DIR / ".memory" / "lancedb")
            cls._instance.db = lancedb.connect(db_path)
            cls._instance.embedder = Embedder()
        return cls._instance

    def _get_or_create_table(self):
        table_name = "english_library"
        if table_name not in self.db.table_names():
            return self.db.create_table(table_name, schema=TextChunk)
        return self.db.open_table(table_name)

    def ask_text_file(self, filepath: str, query: str) -> str:
        pth = Path(filepath).expanduser().resolve()
        filepath = str(pth)

        if not pth.exists() or not pth.is_file():
            return f"Error: Cannot find file at {filepath}"

        if pth.stat().st_size > 5 * 1024 * 1024:
            return f"Error: File is too large ({pth.stat().st_size} bytes). Max allowed is 5MB."

        table = self._get_or_create_table()
        now = datetime.now(timezone.utc)

        existing_by_hash = table.search().where(f"file_hash = '{file_hash}'").limit(1).to_list()
        existing_by_path = table.search().where(f"filepath = '{filepath}'").limit(1).to_list()

        if existing_by_hash:
            table.update(
                where=f"file_hash = '{file_hash}'",
                values={"filepath": filepath, "last_accessed": now}
            )
        elif existing_by_path:
            table.delete(where=f"filepath = '{filepath}'")

        if not existing_by_hash:
            self._index_file(table, pth, filepath, file_hash, now)

        query_vector = self.embedder.embed([query])[0]
        top_chunks = (
            table.search(query_vector)
            .where(f"filepath = '{filepath}'")
            .limit(8)
            .to_list()
        )

        if not top_chunks:
            return f"No relevant content found in {filepath} for query: {query}"

        res = [f"--- Chunk {c['chunk_index']} ---\n{c['text']}" for c in top_chunks]
        return f"Relevant context from the file:\n\n{'\n\n'.join(res)}"

    def _index_file(self, table, pth: Path, filepath: str, file_hash: str, now: datetime):
        chunker = ChunkingManager(model=embed_model, context_len=CONTEXT_LIMIT)

        with open(pth, "r", encoding="utf-8", errors="replace") as f:
            doc = f.read()

        suffix_map = {
            
        }
        doc_type = suffix_map.get(pth.suffix, "prose")
        chunks = chunker.chunk_doc(doc=doc, doc_type="english")

        chunks_info = []
        for i, chunk in enumerate(chunks):
            vector = self.embedder.embed([chunk])[0]
            chunks_info.append({
                "last_accessed": now,
                "chunk_index": i,
                "text": chunk,
                "vector": vector.tolist()
            })

        table.add(chunks_info)