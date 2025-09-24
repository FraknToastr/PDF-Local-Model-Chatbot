import pickle

with open("data/all_chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

print(f"Chunks loaded: {len(chunks)}")
for i, c in enumerate(chunks[:5]):  # preview first 5
    print(i, c.keys(), c)
