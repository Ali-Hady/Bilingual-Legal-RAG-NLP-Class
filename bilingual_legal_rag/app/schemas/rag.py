from pydantic import BaseModel, Field
from typing import Optional

# What the user must send
# What your API will return
# What fields are required
# What types are allowed
# What validation rules should be applied

class RagQueryRequest(BaseModel):
    query : str = Field(..., min_length = 3)
    top_k : int = Field(default = 5, ge = 1, le = 10)

class Source(BaseModel):
    title : str
    category : str
    act_id : str
    url_source : str
    chunk_index : str

class RetrievedChunk(BaseModel):
    text : str
    score : Optional[float]
    metaData : Source

class Response(BaseModel):
    answer : str
    sources : list[Source]
    retrieved_chunks : list[RetrievedChunk]


class QueryRequest(BaseModel):
    query: str
    lang: str # "en" or "ar"