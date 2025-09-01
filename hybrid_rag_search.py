import os, asyncio, logging
from dotenv import load_dotenv
from elasticsearch import AsyncElasticsearch
from mistralai import Mistral

load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
ES_URL = os.getenv("ES_URL", "http://localhost:9200")
INDEX_NAME = "building_codes"

logging.basicConfig(level=logging.INFO)

# --- Clients ---
es = AsyncElasticsearch([ES_URL])
mistral = Mistral(api_key=MISTRAL_API_KEY)

# --- Embeddings ---
async def embed(text: str) -> list[float]:
    r = await mistral.embeddings.create_async(model="mistral-embed", inputs=[text])
    return r.data[0].embedding

# --- Retrieval ---
async def hybrid_search(query: str, k: int = 5):
    try:
        vec = await embed(query)

        # Hybrid search combining BM25 and vector similarity
        body = {
            "query": {
                "bool": {
                    "should": [
                        # BM25 text search
                        {
                            "multi_match": {
                                "query": query,
                                "fields": ["title^2", "text", "code"],
                                "type": "best_fields"
                            }
                        },
                        # Vector similarity search
                        {
                            "script_score": {
                                "query": {"match_all": {}},
                                "script": {
                                    "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                                    "params": {"query_vector": vec}
                                }
                            }
                        }
                    ]
                }
            },
            "size": k
        }

        resp = await es.search(index=INDEX_NAME, body=body)
        return [
            (
                hit["_score"], 
                hit["_source"].get("code", hit["_source"].get("section_id", "UNKNOWN")), 
                hit["_source"].get("title", ""),
                hit["_source"]["text"][:500] + "..." if len(hit["_source"]["text"]) > 500 else hit["_source"]["text"]
            )
            for hit in resp["hits"]["hits"]
        ]
    
    except Exception as e:
        logging.error(f"Hybrid search failed: {e}")
        # Fallback to simple text search
        try:
            resp = await es.search(
                index=INDEX_NAME,
                body={
                    "query": {
                        "multi_match": {
                            "query": query,
                            "fields": ["title^2", "text", "code"]
                        }
                    },
                    "size": k
                }
            )
            return [
                (
                    hit["_score"], 
                    hit["_source"].get("code", "UNKNOWN"), 
                    hit["_source"].get("title", ""),
                    hit["_source"]["text"][:500] + "..." if len(hit["_source"]["text"]) > 500 else hit["_source"]["text"]
                )
                for hit in resp["hits"]["hits"]
            ]
        except Exception as fallback_e:
            logging.error(f"Fallback search also failed: {fallback_e}")
            return []

# --- Generation ---
async def generate_answer(query: str, results) -> str:
    if not results:
        return "I couldn't find relevant information in the building codes to answer your question. Please try rephrasing your query or contact the Dallas Building Department directly."
    
    # Format context with section codes, titles, and text
    context_parts = []
    for score, code, title, text in results:
        section = f"[{code}]"
        if title:
            section += f" {title}"
        context_parts.append(f"{section}\n{text}")
    
    context = "\n\n---\n\n".join(context_parts)

    messages = [
        {
            "role": "system", 
            "content": """Rules for Answering
                -Use only the retrieved code sections that are directly relevant to the user’s question.
                -Discard irrelevant sections (e.g., parking rules, unrelated zoning chapters). Do not mention them in your answer.
                -Always cite specific section numbers (e.g., [48C-46]) for any ordinance referenced.
                -If the retrieved sections do not provide enough detail, state that clearly and suggest the likely next step (e.g., contacting the Building Inspection office or checking permit requirements in Chapter 52).
                -Focus on practical compliance guidance in plain English.

            Organize your answer clearly:
                -Direct requirements from cited sections
                -What’s missing or unclear
                -Next step for the user"""
        },
        {
            "role": "user", 
            "content": f"Question: {query}\n\nRelevant Dallas Building Code sections:\n\n{context}\n\nAnswer the question using only the provided sections. Cite section numbers for all information used."
        }
    ]

    try:
        r = await mistral.chat.complete_async(model="mistral-small", messages=messages)
        return r.choices[0].message.content
    except Exception as e:
        logging.error(f"Answer generation failed: {e}")
        return f"I found relevant sections but encountered an error generating the response. Please try your question again. Found sections: {', '.join([code for _, code, _, _ in results])}"

# --- Main ---
async def run_query(query: str):
    print(f"\n🔍 COGS Query: {query}")
    print("=" * 50)
    
    try:
        results = await hybrid_search(query)
        
        if results:
            print(f"\n📋 Top {len(results)} relevant sections:")
            for i, (score, code, title, text) in enumerate(results, 1):
                print(f"{i}. [{code}] {title[:60]}{'...' if len(title) > 60 else ''} (score: {score:.3f})")
            
            print(f"\n🤖 COGS Response:")
            print("-" * 30)
            answer = await generate_answer(query, results)
            print(answer)
        else:
            print("❌ No relevant building code sections found for your query.")
            print("💡 Try rephrasing your question or using different keywords.")
        
        print("\n" + "=" * 50)
        
    finally:
        # Clean up connections
        await es.close()

if __name__ == "__main__":
    import sys
    # The or statement allows you to run it without a command-line argument.
    # python script_name.py "your query here"
    # or python script_name.py
    asyncio.run(run_query(" ".join(sys.argv[1:]) or "building permit requirements"))
