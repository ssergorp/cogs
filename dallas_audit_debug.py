#!/usr/bin/env python3
"""
Dallas Ordinance Files Audit & Embedding Generation Debug Script
================================================================

This script performs comprehensive analysis of:
1. Source Dallas ordinance files (content audit)
2. Database embedding quality and uniqueness
3. Missing critical zoning/restaurant regulations
4. Embedding generation pipeline debugging

Usage:
    python dallas_audit_debug.py --db data/cogs_memory.db --source-dir data/
"""

import sqlite3
import json
import os
import re
import hashlib
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple, Optional
import logging
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class EmbeddingAnalysis:
    """Container for embedding analysis results"""
    doc_id: str
    section_id: str
    embedding_hash: str
    embedding_stats: Dict[str, float]
    content_preview: str
    is_duplicate: bool = False

@dataclass
class ContentGap:
    """Container for identified content gaps"""
    category: str
    missing_terms: List[str]
    expected_sections: List[str]
    found_count: int
    expected_count: int

class DallasOrdinanceAuditor:
    """Comprehensive auditor for Dallas ordinance files and database"""
    
    def __init__(self, db_path: str, source_dir: str = None):
        self.db_path = db_path
        self.source_dir = source_dir
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        
        # Critical Dallas ordinance patterns
        self.critical_patterns = {
            'zoning_cs': [
                r'commercial.*service.*district',
                r'cs.*zoning.*district',
                r'cs.*district.*regulations',
                r'sec\.?\s*51a-4\.20[0-9]',  # CS district sections
                r'chapter.*51a.*zoning'
            ],
            'restaurant_patio': [
                r'restaurant.*patio.*permit',
                r'outdoor.*dining.*permit',
                r'sidewalk.*cafe.*permit',
                r'patio.*construction.*requirements',
                r'restaurant.*outdoor.*seating'
            ],
            'zoning_general': [
                r'chapter.*51a?.*zoning',
                r'zoning.*ordinance',
                r'land.*use.*regulations',
                r'permitted.*uses.*table'
            ],
            'restaurant_licensing': [
                r'restaurant.*license',
                r'food.*service.*permit',
                r'commercial.*kitchen.*requirements',
                r'restaurant.*health.*code'
            ]
        }
        
        # Expected Dallas zoning sections
        self.expected_zoning_sections = [
            'SEC. 51A-4.200',  # CS district purpose
            'SEC. 51A-4.201',  # CS principal uses
            'SEC. 51A-4.202',  # CS accessory uses
            'SEC. 51A-4.203',  # CS conditional uses
            'SEC. 51A-4.204',  # CS development standards
            'SEC. 51A-4.205',  # CS parking requirements
            'SEC. 51A-13.500', # Restaurant regulations
            'SEC. 28-121',     # Restaurant licensing
        ]

    def audit_source_files(self) -> Dict[str, any]:
        """Audit source Dallas ordinance files for completeness"""
        logger.info("🔍 Auditing source Dallas ordinance files...")
        
        if not self.source_dir or not os.path.exists(self.source_dir):
            return {"error": "Source directory not found or not specified"}
        
        audit_results = {
            "files_found": [],
            "total_size": 0,
            "content_analysis": {},
            "missing_sections": [],
            "file_types": Counter(),
            "zoning_chapters": [],
        }
        
        # Scan source directory
        for root, dirs, files in os.walk(self.source_dir):
            for file in files:
                if file.lower().endswith(('.txt', '.json', '.html', '.pdf')):
                    file_path = os.path.join(root, file)
                    file_size = os.path.getsize(file_path)
                    
                    audit_results["files_found"].append({
                        "path": file_path,
                        "size": file_size,
                        "type": file.split('.')[-1].lower()
                    })
                    audit_results["total_size"] += file_size
                    audit_results["file_types"][file.split('.')[-1].lower()] += 1
        
        # Analyze file contents
        for file_info in audit_results["files_found"]:
            try:
                content_analysis = self._analyze_file_content(file_info["path"])
                audit_results["content_analysis"][file_info["path"]] = content_analysis
                
                # Check for zoning chapters
                if content_analysis.get("has_chapter_51a"):
                    audit_results["zoning_chapters"].append(file_info["path"])
                    
            except Exception as e:
                logger.error(f"Error analyzing {file_info['path']}: {e}")
        
        # Check for missing critical sections
        audit_results["missing_sections"] = self._find_missing_sections(audit_results["content_analysis"])
        
        return audit_results

    def _analyze_file_content(self, file_path: str) -> Dict[str, any]:
        """Analyze individual file content for Dallas ordinance patterns"""
        analysis = {
            "size": os.path.getsize(file_path),
            "has_chapter_51a": False,
            "has_cs_district": False,
            "has_restaurant_regs": False,
            "pattern_matches": defaultdict(int),
            "section_count": 0,
            "content_preview": ""
        }
        
        try:
            # Try to read file content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            analysis["content_preview"] = content[:500] + "..." if len(content) > 500 else content
            
            # Count sections
            analysis["section_count"] = len(re.findall(r'sec\.?\s*\d+[a-z]?-\d+', content, re.IGNORECASE))
            
            # Check for critical patterns
            for category, patterns in self.critical_patterns.items():
                for pattern in patterns:
                    matches = len(re.findall(pattern, content, re.IGNORECASE))
                    if matches > 0:
                        analysis["pattern_matches"][category] += matches
            
            # Specific checks
            analysis["has_chapter_51a"] = bool(re.search(r'chapter.*51a', content, re.IGNORECASE))
            analysis["has_cs_district"] = bool(re.search(r'cs.*district', content, re.IGNORECASE))
            analysis["has_restaurant_regs"] = bool(re.search(r'restaurant.*regulation', content, re.IGNORECASE))
            
        except Exception as e:
            analysis["error"] = str(e)
            
        return analysis

    def _find_missing_sections(self, content_analysis: Dict) -> List[str]:
        """Find missing expected Dallas zoning sections"""
        found_sections = set()
        
        for file_path, analysis in content_analysis.items():
            if "error" not in analysis:
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    for expected_section in self.expected_zoning_sections:
                        if expected_section.replace(' ', '').lower() in content.replace(' ', '').lower():
                            found_sections.add(expected_section)
                except:
                    continue
        
        missing = [section for section in self.expected_zoning_sections if section not in found_sections]
        return missing

    def debug_embedding_generation(self) -> Dict[str, any]:
        """Debug embedding generation issues and identify duplicates"""
        logger.info("🧮 Debugging embedding generation...")
        
        debug_results = {
            "total_documents": 0,
            "embedding_analysis": [],
            "duplicate_groups": [],
            "embedding_stats": {},
            "quality_issues": [],
            "recommendations": []
        }
        
        # Get all documents with embeddings
        cursor = self.conn.execute("""
            SELECT doc_id, section_id, title, text, embedding
            FROM documents 
            WHERE embedding IS NOT NULL AND embedding != ''
            ORDER BY doc_id, section_id
        """)
        
        embedding_hashes = defaultdict(list)
        all_analyses = []
        
        for row in cursor.fetchall():
            try:
                # Parse embedding
                embedding_data = json.loads(row['embedding'])
                if isinstance(embedding_data, list):
                    embedding_array = np.array(embedding_data, dtype=np.float32)
                else:
                    debug_results["quality_issues"].append(f"Invalid embedding format: {row['doc_id']}-{row['section_id']}")
                    continue
                
                # Calculate embedding hash for duplicate detection
                embedding_hash = hashlib.md5(json.dumps(embedding_data, sort_keys=True).encode()).hexdigest()
                
                # Calculate embedding statistics
                stats = {
                    'min': float(np.min(embedding_array)),
                    'max': float(np.max(embedding_array)),
                    'mean': float(np.mean(embedding_array)),
                    'std': float(np.std(embedding_array)),
                    'zeros': int(np.sum(embedding_array == 0)),
                    'dimension': len(embedding_array)
                }
                
                analysis = EmbeddingAnalysis(
                    doc_id=row['doc_id'],
                    section_id=row['section_id'],
                    embedding_hash=embedding_hash,
                    embedding_stats=stats,
                    content_preview=row['text'][:200] if row['text'] else "No content",
                )
                
                embedding_hashes[embedding_hash].append(analysis)
                all_analyses.append(analysis)
                
            except Exception as e:
                debug_results["quality_issues"].append(f"Error processing {row['doc_id']}-{row['section_id']}: {e}")
        
        debug_results["total_documents"] = len(all_analyses)
        debug_results["embedding_analysis"] = [
            {
                "doc_id": a.doc_id,
                "section_id": a.section_id,
                "stats": a.embedding_stats,
                "content_preview": a.content_preview,
                "is_duplicate": len(embedding_hashes[a.embedding_hash]) > 1
            }
            for a in all_analyses[:20]  # Show first 20 for brevity
        ]
        
        # Find duplicate groups
        for embedding_hash, analyses in embedding_hashes.items():
            if len(analyses) > 1:
                debug_results["duplicate_groups"].append({
                    "hash": embedding_hash,
                    "count": len(analyses),
                    "documents": [
                        {
                            "doc_id": a.doc_id,
                            "section_id": a.section_id,
                            "content_preview": a.content_preview
                        }
                        for a in analyses
                    ]
                })
        
        # Overall embedding statistics
        if all_analyses:
            all_stats = [a.embedding_stats for a in all_analyses]
            debug_results["embedding_stats"] = {
                "unique_hashes": len(embedding_hashes),
                "duplicate_percentage": (len(all_analyses) - len(embedding_hashes)) / len(all_analyses) * 100,
                "avg_dimension": np.mean([s['dimension'] for s in all_stats]),
                "avg_std": np.mean([s['std'] for s in all_stats]),
                "zero_embeddings": sum(1 for s in all_stats if s['std'] == 0),
            }
        
        # Generate recommendations
        debug_results["recommendations"] = self._generate_embedding_recommendations(debug_results)
        
        return debug_results

    def _generate_embedding_recommendations(self, debug_results: Dict) -> List[str]:
        """Generate recommendations based on embedding analysis"""
        recommendations = []
        
        if debug_results["embedding_stats"].get("duplicate_percentage", 0) > 10:
            recommendations.append("🚨 HIGH: >10% duplicate embeddings detected - check embedding generation pipeline")
        
        if debug_results["embedding_stats"].get("zero_embeddings", 0) > 0:
            recommendations.append("🚨 CRITICAL: Zero-variance embeddings found - embedding generation failed")
        
        if debug_results["embedding_stats"].get("avg_std", 0) < 0.01:
            recommendations.append("⚠️  MEDIUM: Very low embedding variance - may indicate poor quality embeddings")
        
        if len(debug_results["duplicate_groups"]) > 5:
            recommendations.append("🔧 ACTION: Implement embedding deduplication in your pipeline")
        
        recommendations.append("✅ VERIFY: Test embedding generation with known different content")
        recommendations.append("✅ VERIFY: Check if Mistral API is returning cached/default responses")
        
        return recommendations

    def analyze_content_gaps(self) -> Dict[str, ContentGap]:
        """Analyze gaps in critical Dallas ordinance content"""
        logger.info("📋 Analyzing content gaps...")
        
        gaps = {}
        
        cursor = self.conn.execute("""
            SELECT doc_id, section_id, title, text
            FROM documents
        """)
        
        all_content = []
        for row in cursor.fetchall():
            all_content.append({
                'doc_id': row['doc_id'],
                'section_id': row['section_id'], 
                'title': row['title'] or '',
                'text': row['text'] or ''
            })
        
        # Analyze each critical category
        for category, patterns in self.critical_patterns.items():
            found_count = 0
            found_sections = []
            
            for doc in all_content:
                full_text = f"{doc['title']} {doc['text']}".lower()
                
                for pattern in patterns:
                    if re.search(pattern, full_text, re.IGNORECASE):
                        found_count += 1
                        found_sections.append(f"{doc['doc_id']}-{doc['section_id']}")
                        break
            
            # Expected counts (rough estimates based on typical municipal codes)
            expected_counts = {
                'zoning_cs': 8,  # Should have ~8 CS district sections
                'restaurant_patio': 5,  # Should have ~5 patio/outdoor dining sections
                'zoning_general': 20,  # Should have ~20 general zoning sections
                'restaurant_licensing': 10  # Should have ~10 restaurant licensing sections
            }
            
            gaps[category] = ContentGap(
                category=category,
                missing_terms=[p for p in patterns if not any(re.search(p, f"{doc['title']} {doc['text']}", re.IGNORECASE) for doc in all_content)],
                expected_sections=[],  # Would need domain expert input
                found_count=found_count,
                expected_count=expected_counts.get(category, 5)
            )
        
        return gaps

    def generate_comprehensive_report(self) -> str:
        """Generate comprehensive audit and debug report"""
        logger.info("📊 Generating comprehensive report...")
        
        # Run all analyses
        source_audit = self.audit_source_files()
        embedding_debug = self.debug_embedding_generation()
        content_gaps = self.analyze_content_gaps()
        
        report = [
            "=" * 80,
            "🏛️  DALLAS ORDINANCE AUDIT & EMBEDDING DEBUG REPORT",
            "=" * 80,
            "",
            "📁 SOURCE FILES AUDIT",
            "-" * 40,
        ]
        
        if "error" in source_audit:
            report.extend([
                f"❌ Error: {source_audit['error']}",
                "💡 Recommendation: Specify --source-dir with Dallas ordinance files",
                ""
            ])
        else:
            report.extend([
                f"📊 Files found: {len(source_audit['files_found'])}",
                f"📊 Total size: {source_audit['total_size'] / 1024 / 1024:.2f} MB",
                f"📊 File types: {dict(source_audit['file_types'])}",
                f"📊 Files with Chapter 51A: {len(source_audit['zoning_chapters'])}",
                f"📊 Missing critical sections: {len(source_audit['missing_sections'])}",
                ""
            ])
            
            if source_audit['missing_sections']:
                report.extend([
                    "🚨 MISSING CRITICAL SECTIONS:",
                    *[f"   - {section}" for section in source_audit['missing_sections'][:10]],
                    ""
                ])
        
        report.extend([
            "🧮 EMBEDDING ANALYSIS",
            "-" * 40,
            f"📊 Total documents analyzed: {embedding_debug['total_documents']}",
            f"📊 Unique embeddings: {embedding_debug['embedding_stats'].get('unique_hashes', 0)}",
            f"📊 Duplicate percentage: {embedding_debug['embedding_stats'].get('duplicate_percentage', 0):.2f}%",
            f"📊 Zero-variance embeddings: {embedding_debug['embedding_stats'].get('zero_embeddings', 0)}",
            f"📊 Average std deviation: {embedding_debug['embedding_stats'].get('avg_std', 0):.6f}",
            ""
        ])
        
        if embedding_debug['duplicate_groups']:
            report.extend([
                "🚨 DUPLICATE EMBEDDING GROUPS (Top 5):",
                *[f"   - {group['count']} documents with identical embeddings" 
                  for group in embedding_debug['duplicate_groups'][:5]],
                ""
            ])
        
        report.extend([
            "📋 CONTENT GAP ANALYSIS",
            "-" * 40,
        ])
        
        for category, gap in content_gaps.items():
            coverage = (gap.found_count / gap.expected_count) * 100 if gap.expected_count > 0 else 0
            status = "✅" if coverage > 80 else "⚠️ " if coverage > 50 else "❌"
            report.append(f"{status} {category.replace('_', ' ').title()}: {gap.found_count}/{gap.expected_count} ({coverage:.1f}%)")
        
        report.extend([
            "",
            "🎯 RECOMMENDATIONS",
            "-" * 40,
            *embedding_debug['recommendations'],
            "",
            "🔧 NEXT STEPS:",
            "1. Fix source data: Ensure complete Dallas Chapter 51A is included",
            "2. Fix embedding pipeline: Resolve duplicate embedding generation",
            "3. Add missing content: Focus on CS district and restaurant regulations",
            "4. Improve chunking: Avoid splitting related regulations",
            "5. Test with known queries: Verify improvements with specific test cases",
            ""
        ])
        
        return "\n".join(report)

    def __del__(self):
        """Clean up database connection"""
        if hasattr(self, 'conn'):
            self.conn.close()


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Audit Dallas ordinance files and debug embeddings")
    parser.add_argument("--db", required=True, help="Path to SQLite database")
    parser.add_argument("--source-dir", help="Path to source Dallas ordinance files directory")
    parser.add_argument("--output", help="Output file for report (default: print to console)")
    parser.add_argument("--debug-embeddings", action="store_true", help="Focus on embedding debugging only")
    parser.add_argument("--audit-sources", action="store_true", help="Focus on source file audit only")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.db):
        print(f"❌ Database not found: {args.db}")
        return 1
    
    auditor = DallasOrdinanceAuditor(args.db, args.source_dir)
    
    try:
        if args.debug_embeddings:
            # Embedding debugging only
            results = auditor.debug_embedding_generation()
            print(json.dumps(results, indent=2))
        elif args.audit_sources:
            # Source audit only
            results = auditor.audit_source_files()
            print(json.dumps(results, indent=2))
        else:
            # Comprehensive report
            report = auditor.generate_comprehensive_report()
            
            if args.output:
                with open(args.output, 'w') as f:
                    f.write(report)
                print(f"📄 Report saved to: {args.output}")
            else:
                print(report)
    
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
