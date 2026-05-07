from vectordb  import LanceManager

LM = LanceManager()

def embed_collection(collection, lang):
    cursor = collection.find({})
    docs = list(cursor)
    LM.index_dataset(docs, lang)

#LM.index_dataset("seed_data/arabic_laws_seed.json","ar")
#LM.index_dataset("seed_data/english_laws.json","eng")

def generate(db, query : str,lang:str, top_k : int = 5):

    retrieved_chunks =LM.get_top_chunks(query=query,lang=lang,lim=top_k)
    ans = ""
    sources = ""
    return {"answer" : ans, "sources" : sources, "retrieved_chunks" : retrieved_chunks}