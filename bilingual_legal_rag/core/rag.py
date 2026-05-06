


def generate(db, query : str, top_k : int = 5):
    ans = ""
    sources = ""
    retrieved_chunks = ""
    return {"answer" : ans, "sources" : sources, "retrieved_chunks" : retrieved_chunks}