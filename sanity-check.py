import pickle, re
with open("data/all_chunks.pkl","rb") as f:
    chunks = pickle.load(f)

hits = []
for c in chunks:
    if "chunk" in c:
        text = c["chunk"]["text"]
    else:
        text = c.get("text", "")
    if "Lomax-Smith" in text:
        hits.append(c)

