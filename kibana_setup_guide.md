# COGS Kibana Dashboard Setup Guide

Your Kibana instance is running but needs to be reconfigured to access the building_codes data. Here's how to restore it:

## Quick Verification
First, let's verify your data is accessible:
```bash
# Check if building_codes index exists
curl "localhost:9200/_cat/indices?v"

# Should show:
# building_codes    3961 documents    7.3mb
```

## Manual Kibana Setup (Recommended)

### Step 1: Access Kibana
1. Open http://localhost:5601 in your browser
2. You should see the Kibana welcome screen

### Step 2: Create Index Pattern
1. Navigate to **Management** → **Stack Management** → **Index Patterns**
2. Click **"Create index pattern"**
3. Enter index pattern: `building_codes*`
4. Click **"Next step"**
5. For time field, select `created_at` (if available) or skip
6. Click **"Create index pattern"**

### Step 3: Verify Data Access
1. Navigate to **Discover**
2. Select `building_codes*` index pattern
3. You should see 3,961 documents with fields like:
   - `code`: Section codes (e.g., "48C-46")
   - `title`: Section titles
   - `text`: Full section content
   - `category`: Auto-categorized tags
   - `text_length`: Character count
   - `word_count`: Word count

### Step 4: Create COGS Dashboard Visualizations

#### A. Document Count Metric
1. Navigate to **Visualize** → **Create visualization**
2. Select **Metric**
3. Choose `building_codes*` index
4. Default aggregation (Count) shows total documents
5. Save as: "Total Building Codes"

#### B. Category Distribution Pie Chart
1. Create new **Pie** visualization
2. Choose `building_codes*` index
3. Add bucket: **Split slices**
4. Aggregation: **Terms**
5. Field: `category.keyword`
6. Size: 10
7. Save as: "Building Code Categories"

#### C. Text Length Histogram
1. Create new **Histogram** visualization
2. Choose `building_codes*` index
3. X-axis aggregation: **Histogram**
4. Field: `text_length`
5. Interval: 500
6. Save as: "Document Length Distribution"

#### D. Code Search Table
1. Create new **Data table** visualization
2. Add columns for: `code.keyword`, `title.keyword`, `category.keyword`
3. Add search bar for filtering
4. Save as: "Building Code Search"

### Step 5: Create Dashboard
1. Navigate to **Dashboard** → **Create new dashboard**
2. Click **"Add"** and select your saved visualizations:
   - Total Building Codes (top-left)
   - Building Code Categories (top-right)  
   - Document Length Distribution (bottom-left)
   - Building Code Search (bottom-right)
3. Arrange panels and resize as needed
4. Save as: **"COGS - Building Code Analytics"**

## Advanced COGS Query Analytics

### Step 6: Add Query Logging Index (Future)
Once we implement query logging, you can:

1. Create `cogs_queries*` index pattern for query analytics
2. Track metrics like:
   - Query frequency
   - Response times
   - Confidence scores
   - Strategy usage (exact/semantic/hybrid)
   - User search patterns

3. Create advanced visualizations:
   - Query volume over time
   - Confidence score distribution
   - Strategy performance comparison
   - Popular search terms
   - Low-confidence query analysis

### Step 7: Real-time Monitoring Setup
```bash
# Future: Stream COGS query logs to Elasticsearch
# This will enable real-time monitoring of:
# - System performance
# - User behavior
# - Content gaps
# - Error rates
```

## Troubleshooting

### If you can't see data:
1. **Check index pattern**: Make sure it's `building_codes*` (with asterisk)
2. **Verify time range**: Set to "Last 7 days" or "No time filter"
3. **Refresh index**: Management → Index Patterns → building_codes* → Refresh

### If visualizations are empty:
1. Check the selected time range (top-right in Kibana)
2. Verify field mappings in Management → Index Patterns
3. Try a simple query in Discover first

### If Kibana is slow:
1. Reduce time range
2. Add filters to limit data
3. Check Elasticsearch cluster health: `curl localhost:9200/_cluster/health`

## Sample Useful Queries

### Search for specific topics:
```
# Fire safety codes
text: "fire safety" OR category: "safety"

# Building permits
text: "building permit" OR code: "*permit*"

# Noise regulations  
text: "noise" AND (title: "regulation" OR category: "noise")
```

### Filter by document characteristics:
```
# Long documents (over 1000 characters)
text_length: >1000

# Recently added
created_at: [now-7d TO now]

# Specific categories
category: ("safety" OR "permits" OR "zoning")
```

Your building_codes data is intact with 3,961 documents ready for analysis!