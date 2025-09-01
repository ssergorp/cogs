# COGS Architecture

This document describes the technical architecture and design decisions behind the COGS (City Ordinance Guidance System).

## System Overview

COGS is a hybrid RAG (Retrieval-Augmented Generation) system that combines traditional search with AI to provide accurate answers about Dallas building codes.

```
┌─────────────────────────────────────────────────────────────┐
│                        User Interface                       │
│                   python hybrid_rag_search.py               │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                   Query Processing                          │
│              • Input sanitization                           │
│              • Query analysis                               │
│              • Strategy selection                           │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                 Hybrid Search Engine                        │
│  ┌─────────────────┐           ┌─────────────────────────┐  │
│  │  BM25 Search    │           │   Vector Search         │  │
│  │ (Elasticsearch) │           │ (Mistral Embeddings)    │  │
│  │                 │           │                         │  │
│  │ • Keyword match │           │ • Semantic similarity   │  │
│  │ • Exact terms   │           │ • Context understanding │  │
│  │ • Field weights │           │ • Concept matching      │  │
│  └─────────────────┘           └─────────────────────────┘  │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│             Result Ranking & Selection                      │
│              • Score combination                            │
│              • Relevance filtering                          │
│              • Top-K selection                              │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                  Answer Generation                          │
│                   (Mistral AI)                              │
│              • Context assembly                             │
│              • Prompt engineering                           │
│              • Citation generation                          │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                  Query Logging                              │
│               (Analytics & Monitoring)                      │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Document Storage Layer

**Elasticsearch Index**: `building_codes`
```json
{
  "mappings": {
    "properties": {
      "code": {"type": "keyword"},
      "title": {"type": "text"},
      "text": {"type": "text"},
      "category": {"type": "keyword"},
      "embedding": {
        "type": "dense_vector",
        "dims": 1024,
        "similarity": "cosine"
      }
    }
  }
}
```

**Document Structure**:
- **3,961 documents** total
- Each represents a section of Dallas building code
- Full-text + metadata + 1024-dimensional embeddings

### 2. Hybrid Search Engine

**BM25 Component** (`elasticsearch_client.py`):
- Traditional keyword-based search
- Field boosting (title^2 > text > code)
- Fast exact term matching

**Vector Component**:
- Mistral embeddings (1024 dimensions)
- Cosine similarity search
- Semantic understanding beyond keywords

**Combination Strategy**:
```python
# Elasticsearch query structure
{
  "query": {
    "bool": {
      "should": [
        {
          "multi_match": {
            "query": query,
            "fields": ["title^2", "text", "code"]
          }
        },
        {
          "script_score": {
            "query": {"match_all": {}},
            "script": {
              "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0"
            }
          }
        }
      ]
    }
  }
}
```

### 3. AI Integration

**Mistral AI Components**:
- **Embeddings**: `mistral-embed` model for vectors
- **Chat**: `mistral-small` model for answer generation  
- **Async clients**: Non-blocking API calls

**Prompt Engineering**:
```python
system_prompt = """
Rules for Answering:
- Use only the retrieved code sections that are directly relevant
- Always cite specific section numbers (e.g., [48C-46])
- Focus on practical compliance guidance in plain English
- If information is incomplete, state clearly what's missing
"""
```

### 4. Query Processing Pipeline

**Query Analysis** (`cogs_controller.py`):
1. Input sanitization
2. Query type detection (zoning, permits, safety, etc.)
3. Strategy selection (exact match, semantic, hybrid)
4. Confidence scoring

**Retrieval Strategies**:
- **Exact Code Match**: Direct section lookup
- **Semantic Search**: Vector similarity for concepts  
- **Hybrid**: Combined BM25 + vector scores
- **Fallback**: Simple text search if embeddings fail

### 5. Monitoring and Analytics

**Query Logging** (`cogs_query_logger.py`):
```python
@dataclass
class QueryLogEntry:
    query_id: str
    timestamp: str
    query_text: str
    strategy_used: str
    confidence_score: float
    response_time_ms: int
    results_count: int
    success: bool
```

### Latency Breakdown

- **Embedding Generation**: ~200ms (query embedding)
- **Elasticsearch Search**: <10ms (hybrid query)
- **Answer Generation**: ~2000ms (Mistral chat API)
- **Total Response Time**: ~2.2 seconds


