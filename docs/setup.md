# COGS Setup Guide

This guide walks through the complete setup process for the COGS (City Ordinance Guidance System) from scratch.

## Prerequisites

- **Python 3.8+** (3.10+ recommended)
- **Elasticsearch 8.x**
- **8GB+ RAM** (for Elasticsearch and embeddings)
- **Mistral AI API Key** (from [console.mistral.ai](https://console.mistral.ai/))

## Quick Start

### 1. Clone and Install

```bash
git clone <repository-url>
cd cogs-project
pip install -r requirements.txt
```

### 2. Environment Configuration

```bash
cp .env.template .env
# Edit .env and add your Mistral API key
```

### 3. Start Elasticsearch

**Option A: Docker (Recommended)**
```bash
docker run -d --name elasticsearch \
  -p 9200:9200 -p 9300:9300 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  elasticsearch:8.11.0
```

**Option B: Local Installation**
```bash
# Download and extract Elasticsearch 8.11.0
# Start with: bin/elasticsearch
```

### 4. Verify Elasticsearch

```bash
curl http://localhost:9200
# Should return cluster information
```

### 5. Test the System

```bash
python hybrid_rag_search.py "building permit requirements"
```

## Detailed Setup

### Data Loading

If starting fresh, you'll need to load the Dallas building code data:

1. **Process Documents**: Convert HTML/PDF building codes to structured data
2. **Index to Elasticsearch**: Load documents with proper mapping
3. **Generate Embeddings**: Create vector embeddings for all documents

```bash
# Example commands (adjust based on your data source)
python document_processor.py --source data/dallas_codes/
python elasticsearch_indexer.py
python embedding_generator.py
```

### System Architecture

The system consists of several components:

```
📊 Data Layer
├── Elasticsearch (document storage + BM25 search)
├── Vector embeddings (semantic search)
└── SQLite database (query history)

🔍 Search Layer
├── Hybrid search (BM25 + vector similarity)
├── Query routing and analysis
└── Result ranking and filtering

🤖 AI Layer
├── Mistral embeddings (query + document vectors)
├── Mistral chat (response generation)
└── Prompt engineering for legal accuracy

📈 Monitoring Layer
├── Query logging
├── Performance metrics
└── Kibana dashboards
```

### Performance Tuning

**Elasticsearch Configuration**
```bash
# In elasticsearch.yml
indices.query.bool.max_clause_count: 10000
cluster.max_shards_per_node: 3000
```

**System Resources**
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 2GB for building code documents + embeddings
- **CPU**: Multi-core recommended for concurrent queries

### Troubleshooting

**Common Issues:**

1. **Elasticsearch connection failed**
   ```bash
   # Check if Elasticsearch is running
   curl http://localhost:9200/_cluster/health
   ```

2. **Mistral API errors**
   ```bash
   # Test API key
   python -c "import os; from mistralai import Mistral; print(Mistral(api_key=os.getenv('MISTRAL_API_KEY')))"
   ```

3. **No search results**
   ```bash
   # Check if documents are indexed
   curl http://localhost:9200/building_codes/_count
   ```

4. **Memory issues with embeddings**
   - Reduce BATCH_SIZE in .env
   - Process embeddings in smaller chunks

**Log Files:**
- `embedding_generation.log` - Embedding process logs
- System logs via Python logging to stdout/stderr

## Security Considerations

- **API Keys**: Never commit .env files to Git
- **Network Access**: Restrict Elasticsearch access in production
- **Input Validation**: The system sanitizes queries but validate user input
- **Rate Limiting**: Consider implementing rate limits for public APIs

## Next Steps

1. **Set up Kibana** for monitoring: [Kibana Setup Guide](kibana_setup.md)
2. **Explore the API**: [API Documentation](api.md)  
3. **Understand the architecture**: [Architecture Guide](architecture.md)
4. **Deploy to production**: [Deployment Guide](deployment.md)

## Support

- **GitHub Issues**: Report bugs and feature requests
- **Documentation**: Check `/docs` folder for detailed guides
- **Dallas Building Codes**: Contact Dallas Building Inspection Department for code interpretation