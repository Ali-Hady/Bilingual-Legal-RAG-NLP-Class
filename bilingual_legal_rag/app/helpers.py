def serialize_data(doc : dict):
    doc["_id"] = str(doc["_id"])
    return doc