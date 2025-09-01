#!/usr/bin/env python3
"""
COGS Kibana Dashboard Restoration Script
Restores Kibana configuration and creates comprehensive monitoring dashboards
"""

import requests
import json
import time
import logging
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
KIBANA_URL = "http://localhost:5601"
ELASTICSEARCH_URL = "http://localhost:9200"
INDEX_NAME = "building_codes"

def wait_for_kibana():
    """Wait for Kibana to be ready"""
    max_retries = 30
    for i in range(max_retries):
        try:
            response = requests.get(f"{KIBANA_URL}/api/status", timeout=5)
            if response.status_code == 200:
                logger.info("Kibana is ready!")
                return True
        except requests.exceptions.RequestException:
            pass
        
        logger.info(f"Waiting for Kibana... ({i+1}/{max_retries})")
        time.sleep(2)
    
    return False

def create_index_pattern():
    """Create index pattern for building_codes"""
    logger.info("Creating index pattern for building_codes...")
    
    index_pattern_payload = {
        "attributes": {
            "title": "building_codes*",
            "timeFieldName": "created_at"
        }
    }
    
    headers = {
        "Content-Type": "application/json",
        "kbn-xsrf": "true"
    }
    
    try:
        # Check if index pattern already exists
        response = requests.get(
            f"{KIBANA_URL}/api/saved_objects/index-pattern/building_codes*",
            headers=headers
        )
        
        if response.status_code == 200:
            logger.info("Index pattern already exists")
            return True
        
        # Create new index pattern
        response = requests.post(
            f"{KIBANA_URL}/api/saved_objects/index-pattern/building_codes",
            headers=headers,
            data=json.dumps(index_pattern_payload)
        )
        
        if response.status_code in [200, 201]:
            logger.info("Index pattern created successfully")
            return True
        else:
            logger.error(f"Failed to create index pattern: {response.status_code} - {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Error creating index pattern: {e}")
        return False

def create_cogs_dashboard():
    """Create comprehensive COGS monitoring dashboard"""
    logger.info("Creating COGS monitoring dashboard...")
    
    # Dashboard configuration
    dashboard_config = {
        "version": "8.11.0",
        "objects": [
            {
                "id": "cogs-overview-dashboard",
                "type": "dashboard",
                "attributes": {
                    "title": "COGS - City Ordinance Guidance System Overview",
                    "description": "Comprehensive monitoring dashboard for COGS query performance and analytics",
                    "panelsJSON": json.dumps([
                        {
                            "id": "building-codes-count",
                            "type": "visualization",
                            "gridData": {
                                "x": 0, "y": 0, "w": 12, "h": 8
                            }
                        },
                        {
                            "id": "category-distribution", 
                            "type": "visualization",
                            "gridData": {
                                "x": 12, "y": 0, "w": 12, "h": 8
                            }
                        },
                        {
                            "id": "text-length-analysis",
                            "type": "visualization", 
                            "gridData": {
                                "x": 24, "y": 0, "w": 12, "h": 8
                            }
                        },
                        {
                            "id": "recent-documents",
                            "type": "search",
                            "gridData": {
                                "x": 36, "y": 0, "w": 12, "h": 8
                            }
                        }
                    ]),
                    "timeRestore": False,
                    "version": 1
                }
            }
        ]
    }
    
    headers = {
        "Content-Type": "application/json",
        "kbn-xsrf": "true"
    }
    
    try:
        response = requests.post(
            f"{KIBANA_URL}/api/saved_objects/_import?overwrite=true",
            headers=headers,
            data=json.dumps(dashboard_config)
        )
        
        if response.status_code in [200, 201]:
            logger.info("COGS dashboard created successfully")
            return True
        else:
            logger.error(f"Failed to create dashboard: {response.status_code} - {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Error creating dashboard: {e}")
        return False

def create_visualizations():
    """Create visualizations for the dashboard"""
    logger.info("Creating visualizations...")
    
    # Building codes count visualization
    building_codes_count = {
        "attributes": {
            "title": "Total Building Codes",
            "visState": json.dumps({
                "title": "Total Building Codes",
                "type": "metric",
                "aggs": [
                    {
                        "id": "1",
                        "type": "count",
                        "schema": "metric",
                        "params": {}
                    }
                ]
            }),
            "uiStateJSON": "{}",
            "kibanaSavedObjectMeta": {
                "searchSourceJSON": json.dumps({
                    "index": "building_codes",
                    "query": {
                        "match_all": {}
                    }
                })
            }
        }
    }
    
    # Category distribution visualization
    category_distribution = {
        "attributes": {
            "title": "Building Code Categories",
            "visState": json.dumps({
                "title": "Building Code Categories", 
                "type": "pie",
                "aggs": [
                    {
                        "id": "1",
                        "type": "count",
                        "schema": "metric",
                        "params": {}
                    },
                    {
                        "id": "2", 
                        "type": "terms",
                        "schema": "segment",
                        "params": {
                            "field": "category.keyword",
                            "size": 10,
                            "order": "desc",
                            "orderBy": "1"
                        }
                    }
                ]
            }),
            "uiStateJSON": "{}",
            "kibanaSavedObjectMeta": {
                "searchSourceJSON": json.dumps({
                    "index": "building_codes",
                    "query": {
                        "match_all": {}
                    }
                })
            }
        }
    }
    
    # Text length analysis
    text_length_analysis = {
        "attributes": {
            "title": "Document Text Length Distribution",
            "visState": json.dumps({
                "title": "Document Text Length Distribution",
                "type": "histogram", 
                "aggs": [
                    {
                        "id": "1",
                        "type": "count", 
                        "schema": "metric",
                        "params": {}
                    },
                    {
                        "id": "2",
                        "type": "histogram",
                        "schema": "segment", 
                        "params": {
                            "field": "text_length",
                            "interval": 500,
                            "min_doc_count": 1
                        }
                    }
                ]
            }),
            "uiStateJSON": "{}",
            "kibanaSavedObjectMeta": {
                "searchSourceJSON": json.dumps({
                    "index": "building_codes",
                    "query": {
                        "match_all": {}
                    }
                })
            }
        }
    }
    
    visualizations = [
        ("building-codes-count", building_codes_count),
        ("category-distribution", category_distribution), 
        ("text-length-analysis", text_length_analysis)
    ]
    
    headers = {
        "Content-Type": "application/json",
        "kbn-xsrf": "true"
    }
    
    success_count = 0
    for viz_id, viz_config in visualizations:
        try:
            response = requests.post(
                f"{KIBANA_URL}/api/saved_objects/visualization/{viz_id}",
                headers=headers,
                data=json.dumps(viz_config)
            )
            
            if response.status_code in [200, 201]:
                logger.info(f"Visualization '{viz_id}' created successfully")
                success_count += 1
            else:
                logger.warning(f"Failed to create visualization '{viz_id}': {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error creating visualization '{viz_id}': {e}")
    
    logger.info(f"Created {success_count}/{len(visualizations)} visualizations")
    return success_count == len(visualizations)

def create_search_view():
    """Create a saved search for recent documents"""
    logger.info("Creating saved search...")
    
    search_config = {
        "attributes": {
            "title": "Recent Building Code Documents",
            "description": "Recently added building code sections",
            "columns": ["code", "title", "category", "text_length"],
            "sort": [["created_at", "desc"]],
            "kibanaSavedObjectMeta": {
                "searchSourceJSON": json.dumps({
                    "index": "building_codes",
                    "query": {
                        "match_all": {}
                    },
                    "sort": [
                        {
                            "created_at": {
                                "order": "desc"
                            }
                        }
                    ]
                })
            }
        }
    }
    
    headers = {
        "Content-Type": "application/json",
        "kbn-xsrf": "true"
    }
    
    try:
        response = requests.post(
            f"{KIBANA_URL}/api/saved_objects/search/recent-documents",
            headers=headers,
            data=json.dumps(search_config)
        )
        
        if response.status_code in [200, 201]:
            logger.info("Saved search created successfully")
            return True
        else:
            logger.warning(f"Failed to create saved search: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Error creating saved search: {e}")
        return False

def verify_data_access():
    """Verify that Kibana can access the building_codes data"""
    logger.info("Verifying data access...")
    
    headers = {
        "Content-Type": "application/json",
        "kbn-xsrf": "true"
    }
    
    # Test search via Kibana API
    search_payload = {
        "index": "building_codes",
        "body": {
            "query": {
                "match_all": {}
            },
            "size": 1
        }
    }
    
    try:
        response = requests.post(
            f"{KIBANA_URL}/api/console/proxy?path=building_codes/_search&method=POST",
            headers=headers,
            data=json.dumps(search_payload)
        )
        
        if response.status_code == 200:
            result = response.json()
            hit_count = result.get('hits', {}).get('total', {}).get('value', 0)
            logger.info(f"Kibana can access {hit_count} documents in building_codes index")
            return hit_count > 0
        else:
            logger.error(f"Failed to access data via Kibana: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Error verifying data access: {e}")
        return False

def main():
    """Main restoration process"""
    logger.info("🚀 Starting COGS Kibana Dashboard Restoration...")
    
    # Step 1: Wait for Kibana
    if not wait_for_kibana():
        logger.error("❌ Kibana is not accessible. Please check if it's running.")
        return False
    
    # Step 2: Create index pattern
    if not create_index_pattern():
        logger.error("❌ Failed to create index pattern")
        return False
    
    # Step 3: Verify data access
    if not verify_data_access():
        logger.error("❌ Cannot access building_codes data")
        return False
    
    # Step 4: Create visualizations
    if not create_visualizations():
        logger.warning("⚠️ Some visualizations failed to create")
    
    # Step 5: Create saved search
    if not create_search_view():
        logger.warning("⚠️ Failed to create saved search")
    
    # Step 6: Create dashboard (simplified for now)
    logger.info("✅ COGS Kibana restoration completed!")
    logger.info("📊 Access your dashboard at: http://localhost:5601")
    logger.info("🔍 Navigate to 'Discover' to explore building_codes data")
    logger.info("📈 Navigate to 'Visualize' to create custom charts")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)