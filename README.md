# COGS - City Ordinance Guidance System

An intelligent RAG (Retrieval-Augmented Generation) system for querying Dallas building codes and city ordinances using hybrid search and AI-powered responses.

## Overview ##

COGS combines traditional keyword search (BM25) with semantic vector search to help users navigate complex building codes. The system provides accurate, citation-rich responses to questions about zoning, permits, building requirements, and more.

## Features ##

- **Hybrid Search**: Combines Elasticsearch BM25 with vector similarity search
- **3,961 Building Code Documents**: Complete Dallas building code database with embeddings
- **AI-Powered Responses**: Uses Mistral AI for professional, citation-rich answers
- **Real-time Monitoring**: Kibana dashboard for query analytics
- **Production Ready**: Error handling, fallbacks, and robust architecture

## 🚀 Quick Start 🚀 ##

### Prerequisites ##

- Python 3.8+
- Elasticsearch 8.x
- Kibana 8.x (optional, for monitoring)

### Installation ###

1. **Clone the repository**
   ```bash
   git clone <repo-url>
   cd cogs-project
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp .env.template .env
   # Edit .env with your Mistral API key
   ```

4. **Start Elasticsearch**
   ```bash
   # Using Docker
   docker run -d --name elasticsearch \
     -p 9200:9200 -p 9300:9300 \
     -e "discovery.type=single-node" \
     elasticsearch:8.11.0
   ```

### Usage ###

**Note**: Before using COGS, you'll need to load building code data into Elasticsearch and generate embeddings. See [docs/setup.md](docs/setup.md) for detailed instructions.

**Ask a question about Dallas building codes:**
```bash
python hybrid_rag_search.py "Can I build a starbucks in the CS district?"
```

**Sample output:**
```
🔍 COGS Query: Can I build a starbucks in the CS district?
==================================================

📋 Top 5 relevant sections:
1. [51A-4.202] SEC. 51A-4.202. COMMERCIAL AND BUSINESS SERVICE USES. (score: 16.638)
2. [51A-4.210] SEC. 51A-4.210. RETAIL AND PERSONAL SERVICE USES. (score: 15.417)

🤖 COGS Response:
------------------------------
Based on section [51A-4.210], a Starbucks would likely fall under retail and 
personal service uses. However, you'll need to check the specific CS district 
regulations to confirm this use is permitted...
```

## 📁 Project Structure

```
cogs-project/
├── hybrid_rag_search.py          # Main Q&A interface
├── elasticsearch_client.py       # Elasticsearch integration
├── embedding_generator.py        # Vector embedding generation
├── cogs_controller.py            # Query routing and orchestration
├── cogs_query_logger.py          # Query logging for analytics
├── document_processor.py         # Document processing utilities
├── memory_system.py              # Vector database integration
├── html_parser.py               # HTML parsing utilities
├── retrieval_system.py          # Multi-tier retrieval system
├── requirements.txt             # Python dependencies
├── .env.template               # Environment variables template
└── docs/                       # Documentation
    ├── setup.md                # Detailed setup instructions
    ├── api.md                  # API documentation
    └── architecture.md         # System architecture
```

## Configuration ##

### Environment Variables ###

```bash
# Required
MISTRAL_API_KEY=your_mistral_api_key_here

# Optional
ELASTICSEARCH_URL=http://localhost:9200
LOG_LEVEL=INFO
```

### Elasticsearch Setup ###

The system requires an Elasticsearch index with building code documents and vector embeddings:

```bash
# Check if data is loaded
curl "localhost:9200/building_codes/_count"
# Should return: {"count": 3961}

# Verify embeddings are present
curl "localhost:9200/building_codes/_search" -d '{"query":{"exists":{"field":"embedding"}},"size":0}'
```

## Example Queries ##

- **Zoning**: "Can I build a restaurant in the CS district?"
- **Permits**: "What building permits do I need for a renovation?"
- **Setbacks**: "What are the setback requirements for residential buildings?"
- **Fire Safety**: "What are the fire safety requirements for commercial buildings?"

## Monitoring ##

Access the Kibana dashboard at `http://localhost:5601` to monitor:
- Query volume and patterns
- Response times
- Popular search terms
- System performance metrics

### Running Tests ###

```bash
python debug.py
python rag_debug.py
python test_memory.py
```

### Adding New Documents ###

```bash
# Process new documents
python document_processor.py --source new_documents/

# Generate embeddings
python embedding_generator.py
```

## Performance ##

- **Search Latency**: <10ms for hybrid search
- **Total Response Time**: ~2-3 seconds end-to-end
- **Database Size**: 3,961 documents with full-text and vector search
- **Accuracy**: High relevance for building code queries

## License ##

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Built for the City of Dallas Building Code Navigation**
*Empowering residents, developers, and city staff with AI-powered building code assistance*
