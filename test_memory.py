# test_memory.py
import numpy as np
import json
from memory_system import MemorySystem, enhanced_cosine_similarity

def main():
    mem = MemorySystem()

    # clear old docs for clean test - using clear_document instead of delete_all
    mem.clear_document("dallas-tx-1")

    # make two fake embeddings (dim=8 for simplicity)
    emb1 = [0.1, 0.2, 0.3, 0.4, 0, 0, 0, 0]
    emb2 = [0.2, 0.1, 0.4, 0.3, 0, 0, 0, 0]

    mem.save_document(
        doc_id="dallas-tx-1",
        section_id="test-sec-1",
        title="Noise Control",
        text="It shall be unlawful to create unreasonable noise...",
        refs=[{"source": "dallas-code"}],
        embedding=emb1,
    )

    mem.save_document(
        doc_id="dallas-tx-1",
        section_id="test-sec-2",
        title="Parking Restrictions",
        text="No vehicle shall park within 15 feet of a fire hydrant...",
        refs=[{"source": "dallas-code"}],
        embedding=emb2,
    )

    # simulate a query embedding similar to emb1
    query = [0.09, 0.19, 0.31, 0.41, 0, 0, 0, 0]

    # Simple similarity search implementation
    docs = mem.get_all_documents()
    results = []
    for doc in docs:
        if doc['embedding']:
            similarity = enhanced_cosine_similarity(query, doc['embedding'])
            results.append({
                'section_id': doc['section_id'],
                'title': doc['title'],
                'similarity': similarity,
                'text': doc['text']
            })
    
    # Sort by similarity descending and take top k
    results.sort(key=lambda x: x['similarity'], reverse=True)
    results = results[:2]
    
    for r in results:
        print(f"Section {r['section_id']} ({r['title']}) similarity={r['similarity']:.4f}")

    print("Test completed successfully!")

if __name__ == "__main__":
    main()
