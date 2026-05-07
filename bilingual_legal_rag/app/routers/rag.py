from fastapi import APIRouter, Request, HTTPException, status
from bilingual_legal_rag.app.schemas.rag import RagQueryRequest, Response, RetrievedChunk
from bilingual_legal_rag.core.rag import generate

router = APIRouter(prefix = '/rag', tags = ['RAG'])


@router.post('/query', response_model = Response, status_code = status.HTTP_201_CREATED)
async def query_rag(payload : RagQueryRequest, req : Request):
    try:
        database = req.app.state.db
        answer = generate(db = database, query = payload.query, top_k = payload.top_k)
        return answer
    except Exception as e:
        raise HTTPException(status_code = status.HTTP_500_INTERNAL_SERVER_ERROR, detail = f"query failed : {str(e)}")