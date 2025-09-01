# COGS Architecture

This document describes the technical architecture and design decisions behind the COGS (City Ordinance Guidance System).

## System Overview

COGS is a hybrid RAG (Retrieval-Augmented Generation) system that combines traditional search with AI to provide accurate answers about Dallas building codes.

```
┌─────────────────────────────────────────────────────────────┐
│                        User Interface                       │
│                   python hybrid_rag_search.py              │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                   Query Processing                         │
│              • Input sanitization                          │
│              • Query analysis                              │
│              • Strategy selection                          │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                 Hybrid Search Engine                       │
│  ┌─────────────────┐           ┌─────────────────────────┐  │
│  │  BM25 Search    │           │   Vector Search         │  │
│  │ (Elasticsearch) │           │ (Mistral Embeddings)   │  │
│  │                 │           │                         │  │
│  │ • Keyword match │    +      │ • Semantic similarity  │  │
│  │ • Exact terms   │           │ • Context understanding │  │
│  │ • Field weights │           │ • Concept matching     │  │
│  └─────────────────┘           └─────────────────────────┘  │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                Result Ranking & Selection                  │
│              • Score combination                           │
│              • Relevance filtering                         │
│              • Top-K selection                             │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                  Answer Generation                         │
│                   (Mistral AI)                             │
│              • Context assembly                            │
│              • Prompt engineering                          │
│              • Citation generation                         │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                  Query Logging                             │
│               (Analytics & Monitoring)                     │
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

**Kibana Dashboards**:
- Query volume over time
- Popular search terms
- Response time performance
- Success/failure rates

## Data Flow

### Query Processing Flow

1. **Input**: User submits natural language query
2. **Analysis**: Query type detection and strategy selection
3. **Embedding**: Convert query to 1024-dim vector (if using semantic search)
4. **Search**: Execute hybrid Elasticsearch query
5. **Ranking**: Combine BM25 and vector scores
6. **Selection**: Select top 5 most relevant sections
7. **Generation**: Send context to Mistral for answer generation
8. **Response**: Return formatted answer with citations
9. **Logging**: Record query metrics for analytics

### Document Processing Flow (Setup)

1. **Ingestion**: HTML/PDF building code documents
2. **Parsing**: Extract structure (sections, titles, text)
3. **Processing**: Clean and normalize content
4. **Indexing**: Store in Elasticsearch with mappings
5. **Embedding**: Generate Mistral embeddings for each document
6. **Storage**: Update documents with embedding vectors

## Design Decisions

### Why Hybrid Search?

**BM25 Strengths**:
- Fast exact term matching
- Good for code references ("Section 47-4")
- Proven relevance for legal documents

**Vector Search Strengths**:  
- Semantic understanding ("building permit" → "construction authorization")
- Handles synonyms and concepts
- Better for natural language queries

**Combined**: Captures both exact matches and semantic similarity

### Why Mistral AI?

- **Performance**: Good balance of speed and accuracy
- **Cost**: Competitive pricing for embeddings + chat
- **API**: Stable async API with good rate limits
- **Models**: Specialized models for embeddings vs. generation

### Why Elasticsearch?

- **Scale**: Handles 3,961 documents easily, can scale to millions
- **Performance**: Sub-10ms search times
- **Features**: Built-in BM25, vector search, analytics
- **Ecosystem**: Integrates with Kibana for monitoring

## Performance Characteristics

### Latency Breakdown

- **Embedding Generation**: ~200ms (query embedding)
- **Elasticsearch Search**: <10ms (hybrid query)
- **Answer Generation**: ~2000ms (Mistral chat API)
- **Total Response Time**: ~2.2 seconds

### Scalability

**Current Capacity**:
- 3,961 documents indexed
- ~50 queries/minute sustainable  
- Single-node Elasticsearch setup

**Scaling Options**:
- Elasticsearch cluster for higher throughput
- Embedding caching for repeated queries
- Load balancing for multiple API keys
- Database optimization for query logging

## Security Architecture

### Data Protection

- **API Keys**: Environment variables, never committed
- **Input Validation**: Query sanitization and length limits  
- **Access Control**: No authentication currently (single-user)
- **Data Privacy**: Building codes are public information

### Potential Vulnerabilities

- **Prompt Injection**: Mitigated by structured prompts
- **API Abuse**: Rate limiting should be implemented
- **Data Exposure**: Elasticsearch access should be restricted

## Future Enhancements

### Planned Features

1. **Web Interface**: Flask/FastAPI web app
2. **User Authentication**: Multi-user support
3. **Advanced Analytics**: User behavior analysis
4. **Mobile App**: React Native/Flutter client
5. **API Endpoints**: RESTful API for third-party integration

### Technical Improvements

1. **Caching**: Redis for embedding/response caching
2. **Load Balancing**: Multiple API keys and endpoints
3. **A/B Testing**: Different prompt strategies
4. **Fine-tuning**: Custom model training on building codes
5. **Multi-modal**: Support for images/diagrams in codes

This architecture provides a solid foundation for accurate, scalable building code assistance while maintaining good performance and user experience.