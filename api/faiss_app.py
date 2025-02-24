import faiss

index = faiss.read_index("faiss_index/index.faiss")
print("Number of vectors:", index.ntotal)
print("Vector dimension:", index.d)