#!/usr/bin/env python3
"""
COGS (City Ordinance Guidance System) Controller
Main orchestration layer that routes queries through different retrieval strategies
with clear decision rules and confidence scoring.
"""

import os
import re
import json
import uuid
import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

# Local imports
from elasticsearch_client import COGSElasticsearchClient, ElasticsearchResult
from memory_system import MemorySystem, enhanced_search_chunks
from retrieval_system import RetrievalSystem

# Configure logging
logger = logging.getLogger(__name__)

class QueryType(Enum):
    """Query type classifications"""
    EXACT_CODE = "exact_code"          # Section codes like "48C-46"
    EXACT_TITLE = "exact_title"        # Specific title searches
    NATURAL_LANGUAGE = "natural_language"  # Complex questions
    KEYWORD_HEAVY = "keyword_heavy"    # Lots of specific terms
    FUZZY_MATCH = "fuzzy_match"        # Potential typos

class RetrievalStrategy(Enum):
    """Available retrieval strategies"""
    ELASTICSEARCH_EXACT = "elasticsearch_exact"
    ELASTICSEARCH_SEMANTIC = "elasticsearch_semantic"
    ELASTICSEARCH_HYBRID = "elasticsearch_hybrid"  # New hybrid BM25 + vector strategy
    VECTOR_SIMILARITY = "vector_similarity"
    TFIDF_KEYWORD = "tfidf_keyword"
    HYBRID_MULTI = "hybrid_multi"
    WEB_FALLBACK = "web_fallback"

@dataclass
class QueryAnalysis:
    """Analysis results for a query"""
    query_type: QueryType
    confidence: float
    detected_codes: List[str]
    keyword_ratio: float
    word_count: int
    has_typos: bool
    categories: List[str]

@dataclass
class RetrievalResult:
    """Unified result from any retrieval strategy"""
    id: str
    title: str
    content: str
    source: str  # 'elasticsearch', 'vector', 'tfidf', 'web'
    score: float
    confidence: float
    highlights: Dict[str, List[str]]
    metadata: Dict[str, Any]
    
@dataclass
class COGSResponse:
    """Complete COGS response"""
    query: str
    query_id: str
    results: List[RetrievalResult]
    strategy_used: RetrievalStrategy
    confidence_score: float
    total_results: int
    response_time_ms: int
    fallback_used: bool
    sources: List[Dict[str, str]]  # Citation information

# Confidence thresholds for decision making
CONFIDENCE_THRESHOLDS = {
    "exact_match": 0.95,      # Section codes, exact titles
    "high_semantic": 0.85,    # Strong vector similarity
    "medium_keyword": 0.70,   # Good TF-IDF + keyword match
    "low_acceptable": 0.60,   # Minimum acceptable confidence
    "fallback_trigger": 0.50, # Trigger web search fallback
}

# Regex patterns for query analysis
PATTERNS = {
    "section_code": re.compile(r'\b\d+[A-Z]?[-\.]\d+(\.\d+)?\b'),  # 48C-46, 12.5, etc.
    "section_reference": re.compile(r'\bsec(?:tion)?\s+\d+', re.IGNORECASE),
    "chapter_reference": re.compile(r'\bchap(?:ter)?\s+\d+', re.IGNORECASE),
    "ordinal": re.compile(r'\b\d+(st|nd|rd|th)\b'),
}

# Keywords that indicate specific query types
BUILDING_KEYWORDS = {
    "permits": ["permit", "permits", "permitting", "application"],
    "zoning": ["zoning", "zone", "district", "residential", "commercial"],
    "safety": ["fire", "safety", "emergency", "sprinkler", "alarm"],
    "construction": ["building", "construction", "structural", "foundation"],
    "noise": ["noise", "sound", "decibel", "quiet", "disturbance"],
    "parking": ["parking", "vehicle", "automobile", "car", "space"],
}

class QueryAnalyzer:
    """Analyzes queries to determine optimal retrieval strategy"""
    
    def __init__(self):
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 
            'for', 'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through'
        }
    
    def analyze(self, query: str) -> QueryAnalysis:
        """Comprehensive query analysis"""
        query_lower = query.lower().strip()
        words = query_lower.split()
        
        # Detect section codes
        detected_codes = []
        for pattern_name, pattern in PATTERNS.items():
            matches = pattern.findall(query)
            detected_codes.extend(matches)
        
        # Calculate keyword ratio (non-stop words)
        content_words = [w for w in words if w not in self.stop_words]
        keyword_ratio = len(content_words) / len(words) if words else 0
        
        # Detect potential categories
        categories = []
        for category, keywords in BUILDING_KEYWORDS.items():
            if any(keyword in query_lower for keyword in keywords):
                categories.append(category)
        
        # Determine query type and confidence
        if detected_codes:
            query_type = QueryType.EXACT_CODE
            confidence = 0.95
        elif len(words) <= 5 and keyword_ratio > 0.8:
            query_type = QueryType.KEYWORD_HEAVY
            confidence = 0.85
        elif len(words) > 10:
            query_type = QueryType.NATURAL_LANGUAGE
            confidence = 0.75
        else:
            query_type = QueryType.FUZZY_MATCH
            confidence = 0.65
        
        # Basic typo detection (very simple)
        has_typos = self._detect_typos(query_lower)
        
        return QueryAnalysis(
            query_type=query_type,
            confidence=confidence,
            detected_codes=detected_codes,
            keyword_ratio=keyword_ratio,
            word_count=len(words),
            has_typos=has_typos,
            categories=categories
        )
    
    def _detect_typos(self, query: str) -> bool:
        """Simple typo detection"""
        # Look for common patterns that suggest typos
        typo_indicators = [
            r'\w{15,}',  # Very long words
            r'\d[a-z]\d',  # Mixed numbers/letters (could be codes though)
            r'(.)\1{3,}',  # Repeated characters
        ]
        
        for pattern in typo_indicators:
            if re.search(pattern, query):
                return True
        return False

class COGSController:
    """
    Main COGS controller that orchestrates all retrieval strategies
    with clear decision rules and comprehensive logging
    """
    
    def __init__(self, 
                 es_hosts: Optional[List[str]] = None,
                 db_path: str = "data/cogs_memory.db"):
        """Initialize COGS with all retrieval systems"""
        
        # Initialize retrieval clients
        self.es_client = COGSElasticsearchClient(hosts=es_hosts)
        self.vector_memory = MemorySystem(db_path=db_path)
        self.tfidf_retrieval = RetrievalSystem()
        
        # Initialize query analyzer
        self.analyzer = QueryAnalyzer()
        
        # Strategy performance tracking
        self.strategy_stats = {strategy.value: {"queries": 0, "avg_confidence": 0.0} 
                              for strategy in RetrievalStrategy}
        
        logger.info("COGS Controller initialized successfully")
    
    async def query(self, query: str, max_results: int = 10) -> COGSResponse:
        """Main query entry point with full orchestration"""
        start_time = datetime.now()
        query_id = str(uuid.uuid4())
        
        logger.info(f"[{query_id}] Processing query: '{query}'")
        
        try:
            # Step 1: Analyze query
            analysis = self.analyzer.analyze(query)
            logger.debug(f"[{query_id}] Query analysis: {analysis}")
            
            # Step 2: Determine strategy
            strategy = self._determine_strategy(analysis)
            logger.info(f"[{query_id}] Selected strategy: {strategy.value}")
            
            # Step 3: Execute retrieval
            results = await self._execute_strategy(query, analysis, strategy, max_results)
            
            # Step 4: Calculate overall confidence
            confidence = self._calculate_confidence(results, strategy, analysis)
            
            # Step 5: Fallback if confidence too low
            fallback_used = False
            if confidence < CONFIDENCE_THRESHOLDS["fallback_trigger"] and len(results) < 3:
                logger.warning(f"[{query_id}] Low confidence ({confidence:.3f}), attempting fallback")
                fallback_results = await self._fallback_search(query, max_results)
                results.extend(fallback_results)
                fallback_used = True
                # Recalculate confidence with fallback results
                confidence = self._calculate_confidence(results, strategy, analysis)
            
            # Step 6: Extract citations
            sources = self._extract_citations(results)
            
            # Step 7: Build response
            response_time = int((datetime.now() - start_time).total_seconds() * 1000)
            
            response = COGSResponse(
                query=query,
                query_id=query_id,
                results=results[:max_results],  # Limit final results
                strategy_used=strategy,
                confidence_score=confidence,
                total_results=len(results),
                response_time_ms=response_time,
                fallback_used=fallback_used,
                sources=sources
            )
            
            # Update performance stats
            self._update_stats(strategy, confidence)
            
            logger.info(f"[{query_id}] Query completed: {len(results)} results, "
                       f"{confidence:.3f} confidence, {response_time}ms")
            
            return response
            
        except Exception as e:
            logger.error(f"[{query_id}] Query failed: {e}")
            raise
    
    def _determine_strategy(self, analysis: QueryAnalysis) -> RetrievalStrategy:
        """Determine optimal retrieval strategy based on query analysis"""
        
        # Rule 1: Exact section codes -> Elasticsearch exact search
        if analysis.detected_codes:
            return RetrievalStrategy.ELASTICSEARCH_EXACT
        
        # Rule 2: High keyword ratio, short query -> Elasticsearch semantic
        if analysis.keyword_ratio > 0.75 and analysis.word_count <= 6:
            return RetrievalStrategy.ELASTICSEARCH_SEMANTIC
        
        # Rule 3: Natural language questions -> Hybrid BM25 + Vector search
        if analysis.query_type == QueryType.NATURAL_LANGUAGE:
            return RetrievalStrategy.ELASTICSEARCH_HYBRID
        
        # Rule 4: Category-specific queries with natural language -> Hybrid
        if len(analysis.categories) > 0 and analysis.word_count > 5:
            return RetrievalStrategy.ELASTICSEARCH_HYBRID
        
        # Rule 5: Category-specific short queries -> Multi-strategy
        if len(analysis.categories) > 0:
            return RetrievalStrategy.HYBRID_MULTI
        
        # Rule 6: Default to hybrid for better semantic understanding
        return RetrievalStrategy.ELASTICSEARCH_HYBRID
    
    async def _execute_strategy(self, 
                                query: str, 
                                analysis: QueryAnalysis, 
                                strategy: RetrievalStrategy,
                                max_results: int) -> List[RetrievalResult]:
        """Execute the determined retrieval strategy"""
        
        try:
            if strategy == RetrievalStrategy.ELASTICSEARCH_EXACT:
                return await self._elasticsearch_exact_search(query, analysis, max_results)
            
            elif strategy == RetrievalStrategy.ELASTICSEARCH_SEMANTIC:
                return await self._elasticsearch_semantic_search(query, analysis, max_results)
            
            elif strategy == RetrievalStrategy.ELASTICSEARCH_HYBRID:
                return await self._elasticsearch_hybrid_search(query, analysis, max_results)
            
            elif strategy == RetrievalStrategy.VECTOR_SIMILARITY:
                return await self._vector_search(query, max_results)
            
            elif strategy == RetrievalStrategy.TFIDF_KEYWORD:
                return await self._tfidf_search(query, max_results)
            
            elif strategy == RetrievalStrategy.HYBRID_MULTI:
                return await self._hybrid_search(query, analysis, max_results)
            
            else:
                # Default to hybrid search
                return await self._elasticsearch_hybrid_search(query, analysis, max_results)
                
        except Exception as e:
            logger.error(f"Strategy execution failed for {strategy.value}: {e}")
            return []
    
    async def _elasticsearch_exact_search(self, 
                                          query: str, 
                                          analysis: QueryAnalysis,
                                          max_results: int) -> List[RetrievalResult]:
        """Execute Elasticsearch exact search for codes/titles"""
        results = []
        
        # Search for detected codes
        for code in analysis.detected_codes:
            es_results = self.es_client.exact_code_search(code, max_results)
            results.extend(self._convert_es_results(es_results, "elasticsearch"))
        
        # If no code results, try exact title search
        if not results:
            es_results = self.es_client.exact_title_search(query, max_results)
            results.extend(self._convert_es_results(es_results, "elasticsearch"))
        
        return results[:max_results]
    
    async def _elasticsearch_semantic_search(self, 
                                             query: str, 
                                             analysis: QueryAnalysis,
                                             max_results: int) -> List[RetrievalResult]:
        """Execute Elasticsearch semantic search"""
        # Use category filtering if categories detected
        if analysis.categories:
            results = []
            for category in analysis.categories[:2]:  # Limit to 2 categories
                es_results = self.es_client.category_filtered_search(
                    query, category, max_results // len(analysis.categories[:2])
                )
                results.extend(self._convert_es_results(es_results, "elasticsearch"))
        else:
            es_results = self.es_client.semantic_search(query, max_results)
            results = self._convert_es_results(es_results, "elasticsearch")
        
        return results[:max_results]
    
    async def _elasticsearch_hybrid_search(self, 
                                           query: str, 
                                           analysis: QueryAnalysis,
                                           max_results: int) -> List[RetrievalResult]:
        """Execute Elasticsearch hybrid BM25 + vector search"""
        try:
            es_results = await self.es_client.hybrid_vector_search(query, max_results)
            return self._convert_es_results(es_results, "elasticsearch_hybrid")
        except Exception as e:
            logger.error(f"Elasticsearch hybrid search failed: {e}")
            # Fallback to semantic search
            es_results = self.es_client.semantic_search(query, max_results)
            return self._convert_es_results(es_results, "elasticsearch_fallback")
    
    async def _vector_search(self, query: str, max_results: int) -> List[RetrievalResult]:
        """Execute vector similarity search"""
        try:
            # Use the enhanced search from memory_system
            search_results = await enhanced_search_chunks(
                self.vector_memory, query, k=max_results
            )
            
            results = []
            for result in search_results:
                results.append(RetrievalResult(
                    id=result.section_id,
                    title=result.title,
                    content=result.text,
                    source="vector",
                    score=result.score,
                    confidence=result.score * 0.8,  # Vector confidence multiplier
                    highlights={},
                    metadata=result.metadata
                ))
            
            return results
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
    
    async def _tfidf_search(self, query: str, max_results: int) -> List[RetrievalResult]:
        """Execute TF-IDF keyword search"""
        try:
            tfidf_results = self.tfidf_retrieval.retrieve(query, k=max_results)
            
            results = []
            for result in tfidf_results:
                results.append(RetrievalResult(
                    id=result['id'],
                    title=result['title'],
                    content=result['content'],
                    source="tfidf",
                    score=result['similarity'],
                    confidence=result['similarity'] * 0.7,  # TF-IDF confidence multiplier
                    highlights={},
                    metadata={"doc_type": result.get('doc_type', '')}
                ))
            
            return results
            
        except Exception as e:
            logger.error(f"TF-IDF search failed: {e}")
            return []
    
    async def _hybrid_search(self, 
                             query: str, 
                             analysis: QueryAnalysis,
                             max_results: int) -> List[RetrievalResult]:
        """Execute hybrid multi-strategy search"""
        all_results = []
        
        # Get results from multiple strategies
        es_results = await self._elasticsearch_semantic_search(query, analysis, max_results // 2)
        vector_results = await self._vector_search(query, max_results // 2)
        
        all_results.extend(es_results)
        all_results.extend(vector_results)
        
        # Deduplicate and sort by confidence
        seen_ids = set()
        unique_results = []
        
        for result in sorted(all_results, key=lambda x: x.confidence, reverse=True):
            if result.id not in seen_ids:
                unique_results.append(result)
                seen_ids.add(result.id)
        
        return unique_results[:max_results]
    
    async def _fallback_search(self, query: str, max_results: int) -> List[RetrievalResult]:
        """Fallback search when primary strategies fail"""
        # Try fuzzy search
        es_results = self.es_client.fuzzy_search(query, fuzziness="AUTO", size=max_results)
        results = self._convert_es_results(es_results, "fallback")
        
        # TODO: Add web search fallback here
        # web_results = await self._web_search_fallback(query, max_results)
        # results.extend(web_results)
        
        return results
    
    def _convert_es_results(self, 
                           es_results: List[ElasticsearchResult], 
                           source: str) -> List[RetrievalResult]:
        """Convert Elasticsearch results to unified format"""
        results = []
        
        for es_result in es_results:
            results.append(RetrievalResult(
                id=es_result.id,
                title=es_result.title,
                content=es_result.text,
                source=source,
                score=es_result.score,
                confidence=es_result.score,
                highlights=es_result.highlight or {},
                metadata={
                    "code": es_result.code,
                    "category": es_result.category,
                    "text_length": es_result.text_length,
                    "word_count": es_result.word_count
                }
            ))
        
        return results
    
    def _calculate_confidence(self, 
                             results: List[RetrievalResult], 
                             strategy: RetrievalStrategy,
                             analysis: QueryAnalysis) -> float:
        """Calculate overall confidence score for the response"""
        if not results:
            return 0.0
        
        # Base confidence from top result
        base_confidence = results[0].confidence
        
        # Strategy-specific multipliers
        strategy_multipliers = {
            RetrievalStrategy.ELASTICSEARCH_EXACT: 1.0,
            RetrievalStrategy.ELASTICSEARCH_SEMANTIC: 0.9,
            RetrievalStrategy.ELASTICSEARCH_HYBRID: 0.95,  # High confidence for hybrid approach
            RetrievalStrategy.VECTOR_SIMILARITY: 0.85,
            RetrievalStrategy.TFIDF_KEYWORD: 0.8,
            RetrievalStrategy.HYBRID_MULTI: 0.95,
            RetrievalStrategy.WEB_FALLBACK: 0.6
        }
        
        # Number of results bonus
        result_count_bonus = min(len(results) * 0.05, 0.15)
        
        # Query type confidence adjustment
        query_confidence = analysis.confidence
        
        # Calculate final confidence
        final_confidence = (
            base_confidence * 
            strategy_multipliers.get(strategy, 0.8) * 
            query_confidence + 
            result_count_bonus
        )
        
        return min(final_confidence, 1.0)  # Cap at 1.0
    
    def _extract_citations(self, results: List[RetrievalResult]) -> List[Dict[str, str]]:
        """Extract citation information from results"""
        citations = []
        
        for result in results[:5]:  # Top 5 sources
            citation = {
                "title": result.title,
                "code": result.metadata.get("code", ""),
                "source": result.source,
                "confidence": f"{result.confidence:.2f}"
            }
            citations.append(citation)
        
        return citations
    
    def _update_stats(self, strategy: RetrievalStrategy, confidence: float):
        """Update strategy performance statistics"""
        stats = self.strategy_stats[strategy.value]
        stats["queries"] += 1
        
        # Update rolling average confidence
        current_avg = stats["avg_confidence"]
        query_count = stats["queries"]
        new_avg = ((current_avg * (query_count - 1)) + confidence) / query_count
        stats["avg_confidence"] = new_avg
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            "strategies": self.strategy_stats,
            "total_queries": sum(stats["queries"] for stats in self.strategy_stats.values()),
            "elasticsearch_health": self.es_client.health_check()
        }
    
    async def close(self):
        """Close all connections"""
        await self.es_client.close()

# Test the controller
if __name__ == "__main__":
    async def test_controller():
        controller = COGSController()
        
        test_queries = [
            "48C-46",  # Exact code
            "building permit requirements",  # Semantic search
            "What are the noise regulations for construction?",  # Natural language
            "fire safety equipment needed",  # Category-specific
        ]
        
        for query in test_queries:
            print(f"\n=== Testing: '{query}' ===")
            try:
                response = await controller.query(query, max_results=3)
                print(f"Strategy: {response.strategy_used.value}")
                print(f"Confidence: {response.confidence_score:.3f}")
                print(f"Results: {response.total_results}")
                print(f"Time: {response.response_time_ms}ms")
                
                for i, result in enumerate(response.results, 1):
                    print(f"{i}. [{result.confidence:.3f}] {result.title[:60]}...")
                    
            except Exception as e:
                print(f"Error: {e}")
        
        # Print stats
        stats = controller.get_stats()
        print(f"\n=== Controller Stats ===")
        print(f"Total queries: {stats['total_queries']}")
        print(f"Elasticsearch healthy: {stats['elasticsearch_health']}")
        
        await controller.close()
    
    asyncio.run(test_controller())