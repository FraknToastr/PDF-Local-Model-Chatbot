import pickle, re
with open("data/all_chunks.pkl","rb") as f:
    chunks = pickle.load(f)

hits = [c for c in chunks if "Lomax-Smith" in c["chunk"]["text"]]
print("Hits:", len(hits))
print("Examples:", [ (c["metadata"].get("source_file"), c["metadata"].get("page_number")) for c in hits[:5] ])
