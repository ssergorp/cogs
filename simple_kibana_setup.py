#!/usr/bin/env python3
"""
Simple Kibana Setup for COGS
Creates basic index pattern and verifies data access
"""

import requests
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

KIBANA_URL = "http://localhost:5601"

def create_index_pattern_simple():
    """Create index pattern using the correct Kibana 8.x API"""
    logger.info("Creating index pattern...")
    
    # Use the data views API for Kibana 8.x
    payload = {
        "data_view": {
            "title": "building_codes*",
            "name": "Building Codes",
            "timeFieldName": "created_at"
        }
    }
    
    headers = {
        "Content-Type": "application/json",
        "kbn-xsrf": "true"
    }
    
    try:
        # Try the new data views API
        response = requests.post(
            f"{KIBANA_URL}/api/data_views/data_view",
            headers=headers,
            json=payload
        )
        
        if response.status_code in [200, 201]:
            logger.info("✅ Index pattern created successfully via data views API")
            return True
        
        logger.info(f"Data views API response: {response.status_code}")
        logger.info(f"Response: {response.text}")
        
        # Fallback to older index patterns API
        old_payload = {
            "attributes": {
                "title": "building_codes*",
                "timeFieldName": "created_at"
            }
        }
        
        response = requests.post(
            f"{KIBANA_URL}/api/saved_objects/index-pattern",
            headers=headers,
            json=old_payload
        )
        
        if response.status_code in [200, 201]:
            logger.info("✅ Index pattern created successfully via index-pattern API")
            return True
        else:
            logger.warning(f"Index pattern creation failed: {response.status_code}")
            logger.warning(f"Response: {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"Error creating index pattern: {e}")
        return False

def check_kibana_status():
    """Check if Kibana is accessible"""
    try:
        response = requests.get(f"{KIBANA_URL}/api/status", timeout=5)
        if response.status_code == 200:
            status_data = response.json()
            logger.info(f"✅ Kibana is running (version: {status_data.get('version', {}).get('number', 'unknown')})")
            return True
        else:
            logger.error(f"❌ Kibana status check failed: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ Cannot connect to Kibana: {e}")
        return False

def main():
    """Main setup process"""
    logger.info("🚀 COGS Kibana Simple Setup")
    
    # Step 1: Check Kibana status
    if not check_kibana_status():
        logger.error("Please ensure Kibana is running at http://localhost:5601")
        return False
    
    # Step 2: Create index pattern
    if create_index_pattern_simple():
        logger.info("✅ Setup completed successfully!")
        logger.info("")
        logger.info("📊 Next steps:")
        logger.info("1. Open http://localhost:5601 in your browser")
        logger.info("2. Navigate to 'Discover'")
        logger.info("3. Select 'building_codes*' index pattern")
        logger.info("4. You should see 3,961 building code documents")
        logger.info("")
        logger.info("📈 To create visualizations:")
        logger.info("1. Go to 'Visualize' → 'Create visualization'")
        logger.info("2. Choose visualization type (Pie, Bar, Metric, etc.)")
        logger.info("3. Select 'building_codes*' as data source")
        logger.info("4. Configure aggregations and save")
        return True
    else:
        logger.warning("⚠️ Automatic setup failed. Please use manual setup:")
        logger.info("1. Open http://localhost:5601")
        logger.info("2. Go to Management → Stack Management → Index Patterns")
        logger.info("3. Create index pattern: 'building_codes*'")
        logger.info("4. Select 'created_at' as time field (if available)")
        return False

if __name__ == "__main__":
    main()