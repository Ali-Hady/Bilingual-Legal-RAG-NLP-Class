from fastapi import APIRouter, Request, HTTPException, status
from bilingual_legal_rag.app.schemas.rag import RagQueryRequest, Response, QueryRequest
from bilingual_legal_rag.core.rag import generate

router = APIRouter(prefix = '/api/v1/rag', tags = ['RAG'])


@router.post('/query', response_model = Response, status_code = status.HTTP_201_CREATED)
async def query_rag(payload : RagQueryRequest, req : Request):
    try:
        database = req.app.state.db
        answer = generate(db = database, query = payload.query, top_k = payload.top_k)
        return answer
    except Exception as e:
        raise HTTPException(status_code = status.HTTP_500_INTERNAL_SERVER_ERROR, detail = f"query failed : {str(e)}")
    

@router.post('/chat')
async def ask_legal_question(qreq: QueryRequest, request: Request):
    if qreq.lang not in ["en", "ar"]:
        raise HTTPException(status_code=400, detail="Language must be 'en' or 'ar'")
    
    db_manager = request.app.state.LM
    generator = request.app.state.generator

    optimized_query = await generator.generate_search_query(
        query=qreq.query, 
        lang=qreq.lang
    )

    retrieved_context = db_manager.get_top_chunks(
        query=optimized_query, 
        lang=qreq.lang
    )
    
    answer = await generator.generate_answer(
        query=qreq.query, 
        context=retrieved_context, 
        lang=qreq.lang
    )

    return {
        "original_query": qreq.query,
        "optimized_search_query": optimized_query,
        "answer": answer,
        "retrieved_context": retrieved_context
    }