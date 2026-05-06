from fastapi import APIRouter, Request, HTTPException, Query, status
import uuid
# from pydantic import Field
from bilingual_legal_rag.app.helpers import serialize_data
router = APIRouter(prefix = '/laws', tags = ["Laws"])



@router.get('/english')
async def getEnglishLaws(req : Request, limit : int = Query(default = 10, ge = 1, le = 100)
                        ,skip : int = Query(default = 0, ge = 0)):
    allLaws = req.app.state.db["english_laws"]
    data = list(allLaws.find(
        {},
        {
            "raw_xml_snippet": 0,
            "cleaned_text": 0,
        }
    ).skip(skip).limit(limit))

    return {
        "Count" : len(data),
        "Results" : [serialize_data(d) for d in data]
    }