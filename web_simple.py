from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from cogs_main import COGSSystem

app = Flask(__name__)
CORS(app)

# Initialize COGS
cogs = COGSSystem("data/city_ordinances", model_type="mock")
cogs.load_state()

HTML_TEMPLATE = """
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>COGS - City Ordinance Guidance System</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }

        .header {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-bottom: 1px solid rgba(255, 255, 255, 0.2);
            padding: 1rem 2rem;
            position: sticky;
            top: 0;
            z-index: 100;
        }

        .header h1 {
            color: white;
            font-size: 1.8rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .header .subtitle {
            color: rgba(255, 255, 255, 0.8);
            font-size: 0.9rem;
            font-weight: normal;
            margin-top: 0.25rem;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
            display: grid;
            grid-template-columns: 1fr 300px;
            gap: 2rem;
            min-height: calc(100vh - 120px);
        }

        .main-content {
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }

        .query-section {
            padding: 2rem;
            border-bottom: 1px solid #eee;
        }

        .query-input {
            position: relative;
            margin-bottom: 1rem;
        }

        .query-input input {
            width: 100%;
            padding: 1rem 1.5rem;
            border: 2px solid #e1e5e9;
            border-radius: 50px;
            font-size: 1rem;
            outline: none;
            transition: all 0.3s ease;
            background: #f8f9fa;
        }

        .query-input input:focus {
            border-color: #667eea;
            background: white;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }

        .query-buttons {
            display: flex;
            gap: 1rem;
            align-items: center;
        }

        .btn {
            padding: 0.75rem 1.5rem;
            border: none;
            border-radius: 25px;
            font-size: 0.9rem;
            cursor: pointer;
            transition: all 0.3s ease;
            font-weight: 600;
        }

        .btn-primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }

        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
        }

        .btn-secondary {
            background: #f8f9fa;
            color: #666;
            border: 1px solid #e1e5e9;
        }

        .btn-secondary:hover {
            background: #e9ecef;
        }

        .loading {
            display: none;
            text-align: center;
            padding: 2rem;
            color: #666;
        }

        .spinner {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 2px solid #f3f3f3;
            border-top: 2px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-right: 0.5rem;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .results {
            flex: 1;
            padding: 2rem;
            overflow-y: auto;
        }

        .answer-card {
            background: #f8f9ff;
            border-left: 4px solid #667eea;
            padding: 1.5rem;
            border-radius: 0 10px 10px 0;
            margin-bottom: 1.5rem;
        }

        .answer-text {
            line-height: 1.6;
            color: #2c3e50;
            white-space: pre-line;
        }

        .sources {
            margin-top: 1.5rem;
        }

        .sources h3 {
            color: #495057;
            margin-bottom: 1rem;
            font-size: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .source-item {
            background: white;
            border: 1px solid #e9ecef;
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 0.5rem;
            transition: all 0.2s ease;
        }

        .source-item:hover {
            border-color: #667eea;
            box-shadow: 0 2px 8px rgba(102, 126, 234, 0.1);
        }

        .source-title {
            font-weight: 600;
            color: #2c3e50;
            margin-bottom: 0.25rem;
        }

        .source-meta {
            font-size: 0.85rem;
            color: #6c757d;
        }

        .confidence-badge {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
            margin-top: 1rem;
        }

        .confidence-high {
            background: #d4edda;
            color: #155724;
        }

        .confidence-medium {
            background: #fff3cd;
            color: #856404;
        }

        .confidence-low {
            background: #f8d7da;
            color: #721c24;
        }

        .sidebar {
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }

        .sidebar-card {
            background: white;
            border-radius: 15px;
            padding: 1.5rem;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        }

        .sidebar-card h3 {
            color: #495057;
            margin-bottom: 1rem;
            font-size: 1rem;
        }

        .example-queries {
            list-style: none;
        }

        .example-queries li {
            padding: 0.5rem 0;
            border-bottom: 1px solid #f1f3f4;
            cursor: pointer;
            color: #666;
            font-size: 0.9rem;
            transition: color 0.2s ease;
        }

        .example-queries li:hover {
            color: #667eea;
        }

        .example-queries li:last-child {
            border-bottom: none;
        }

        .stats {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
        }

        .stat-item {
            text-align: center;
            padding: 1rem;
            background: #f8f9fa;
            border-radius: 8px;
        }

        .stat-number {
            font-size: 1.5rem;
            font-weight: 700;
            color: #667eea;
        }

        .stat-label {
            font-size: 0.8rem;
            color: #6c757d;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .feedback-section {
            margin-top: 1rem;
            padding-top: 1rem;
            border-top: 1px solid #eee;
            text-align: center;
        }

        .feedback-buttons {
            display: flex;
            gap: 1rem;
            justify-content: center;
            margin-top: 0.5rem;
        }

        .feedback-btn {
            padding: 0.5rem 1rem;
            border: 1px solid #ddd;
            background: white;
            border-radius: 20px;
            cursor: pointer;
            font-size: 0.85rem;
            transition: all 0.2s ease;
        }

        .feedback-btn.positive:hover {
            background: #d4edda;
            border-color: #c3e6cb;
        }

        .feedback-btn.negative:hover {
            background: #f8d7da;
            border-color: #f5c6cb;
        }

        .feedback-btn.selected {
            background: #667eea;
            color: white;
            border-color: #667eea;
        }

        @media (max-width: 768px) {
            .container {
                grid-template-columns: 1fr;
                gap: 1rem;
                padding: 1rem;
            }

            .header {
                padding: 1rem;
            }

            .query-buttons {
                flex-direction: column;
            }
        }
    </style>
</head>
<body>
    <header class="header">
        <h1>
            🏛️ COGS
            <div class="subtitle">City Ordinance Guidance System</div>
        </h1>
    </header>

    <div class="container">
        <main class="main-content">
            <div class="query-section">
                <div class="query-input">
                    <input type="text" id="queryInput" placeholder="Ask about city ordinances, building codes, permits, or zoning regulations..." />
                </div>
                <div class="query-buttons">
                    <button class="btn btn-primary" id="askButton">Ask COGS</button>
                    <button class="btn btn-secondary" id="clearButton">Clear</button>
                </div>
            </div>

            <div class="loading" id="loadingIndicator">
                <div class="spinner"></div>
                Searching city ordinances...
            </div>

            <div class="results" id="resultsContainer">
                <div style="text-align: center; color: #666; margin-top: 3rem;">
                    <h2>Welcome to COGS</h2>
                    <p>Your intelligent assistant for city ordinances and building codes. Ask any question about local regulations, permits, or compliance requirements.</p>
                    <div style="margin-top: 2rem; padding: 1.5rem; background: #f8f9ff; border-radius: 10px; text-align: left;">
                        <h3 style="color: #495057; margin-bottom: 1rem;">💡 How it works:</h3>
                        <ul style="color: #666; line-height: 1.6; padding-left: 1.5rem;">
                            <li><strong>Privacy First:</strong> All data stays local - your queries and information remain private</li>
                            <li><strong>Comprehensive Search:</strong> Searches through all city ordinances, building codes, and zoning regulations</li>
                            <li><strong>Transparent Sources:</strong> Every answer includes citations to specific ordinances and sections</li>
                            <li><strong>Learning System:</strong> Improves over time based on your feedback</li>
                        </ul>
                    </div>
                </div>
            </div>
        </main>

        <aside class="sidebar">
            <div class="sidebar-card">
                <h3>📝 Example Questions</h3>
                <ul class="example-queries" id="exampleQueries">
                    <li>What permits do I need for a deck addition?</li>
                    <li>What are the noise ordinance restrictions for construction?</li>
                    <li>How do I appeal a zoning decision?</li>
                    <li>What are the setback requirements in residential zones?</li>
                    <li>Can I run a business from my home?</li>
                    <li>What are the parking requirements for new construction?</li>
                    <li>How tall can I build a fence in my yard?</li>
                    <li>What are the regulations for short-term rentals?</li>
                </ul>
            </div>

            <div class="sidebar-card">
                <h3>📊 System Stats</h3>
                <div class="stats">
                    <div class="stat-item">
                        <div class="stat-number" id="docCount">--</div>
                        <div class="stat-label">Documents</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-number" id="queryCount">--</div>
                        <div class="stat-label">Queries Today</div>
                    </div>
                </div>
            </div>

            <div class="sidebar-card">
                <h3>🎯 Target Audience</h3>
                <ul style="list-style: none; color: #666; font-size: 0.9rem; line-height: 1.5;">
                    <li>🏛️ City officials and planners</li>
                    <li>🔨 Builders and contractors</li>
                    <li>⚖️ Legal professionals</li>
                    <li>👥 General public</li>
                </ul>
            </div>
        </aside>
    </div>

    <script>
        class COGSInterface {
            constructor() {
                this.currentQueryId = null;
                this.apiBase = '/api';  // Will be configured based on backend
                this.initializeEventListeners();
                this.loadStats();
            }

            initializeEventListeners() {
                const queryInput = document.getElementById('queryInput');
                const askButton = document.getElementById('askButton');
                const clearButton = document.getElementById('clearButton');
                const exampleQueries = document.getElementById('exampleQueries');

                // Query input and buttons
                queryInput.addEventListener('keypress', (e) => {
                    if (e.key === 'Enter') {
                        this.submitQuery();
                    }
                });

            askButton.addEventListener('click', () => this.submitQuery());
                clearButton.addEventListener('click', () => this.clearResults());

                // Example queries
                exampleQueries.addEventListener('click', (e) => {
                    if (e.target.tagName === 'LI') {
                        queryInput.value = e.target.textContent;
                        this.submitQuery();
                    }
                });
            }

async submitQuery() {
    const queryInput = document.getElementById('queryInput');
    const query = queryInput.value.trim();

    if (!query) return;

    this.showLoading();

    try {
        const response = await fetch('/api/query', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query })
        });

        const result = await response.json();

        if (response.ok) {
            // Adapt result to the displayResult format
            this.displayResult(query, {
                query_id: Date.now(),
                answer: result.answer,
                sources: result.sources.map(s => ({
                    title: s.title,
                    doc_type: s.doc_type,
                    relevance: s.similarity || 0.8,
                    section: s.section || 'N/A',
                    effective_date: s.effective_date || 'N/A'
                })),
                confidence: result.confidence || 0.8,
                similar_past_queries: []
            });
        } else {
            this.displayError(result.error || 'Unknown error from server');
        }
    } catch (error) {
        this.displayError('Error communicating with server.');
        console.error(error);
    }
}


            async simulateAPICall(query) {
                // Simulate API delay
                await new Promise(resolve => setTimeout(resolve, 1500));

                // Simulate different types of responses based on query content
                let answer = '';
                let sources = [];
                let confidence = 0.8;

                if (query.toLowerCase().includes('permit')) {
                    answer = `For most deck additions, you will need a building permit from the City Planning Department. The requirements include:

• Structural plans showing deck dimensions, materials, and foundation details
• Site plan showing setback distances from property lines (minimum 5 feet required)

• Electrical permits if adding lighting or outlets
• Applications must be submitted at least 10 business days before construction

The permit fee is typically $125 for decks under 200 square feet. All construction must comply with the International Residential Code (IRC) and local amendments.

Contact the Building Permit Office at (555) 123-4567 or visit City Hall, Room 201 for application forms.`;

                    sources = [
                        {
                            title: 'Building Permit Requirements - Chapter 15',
                            doc_type: 'building',
                            relevance: 0.95,
                            section: 'Section 15.3.2',
                            effective_date: '2024-01-01'
                        },
                        {
                            title: 'Residential Construction Standards',
                            doc_type: 'building',
                            relevance: 0.87,
                            section: 'Section 12.4',
                            effective_date: '2023-06-15'
                        }
                    ];
                } else if (query.toLowerCase().includes('noise')) {
                    answer = `Construction noise is regulated under City Ordinance 8.24. Key restrictions include:

• Construction hours: Monday-Friday 7:00 AM to 6:00 PM, Saturday 8:00 AM to 5:00 PM
• No construction noise on Sundays or city holidays
• Sound levels cannot exceed 65 dB at property line during permitted hours
• Heavy machinery (jackhammers, concrete saws) limited to 9:00 AM to 4:00 PM weekdays only

Violations can result in fines of $500-$2,000 per incident. Emergency repairs are exempt but require notification to the Noise Control Officer.

For noise complaints, call (555) 123-NOISE or file online at city.gov/noise-complaint.`;

                    sources = [
                        {
                            title: 'Noise Control Ordinance 8.24',
                            doc_type: 'noise',
                            relevance: 0.92,
                            section: 'Section 8.24.040',
                            effective_date: '2023-03-01'
                        }
                    ];
                } else if (query.toLowerCase().includes('zoning')) {
                    answer = `To appeal a zoning decision, you must file within 30 days of the decision notice. The process involves:

1. **File Appeal**: Submit Form ZA-100 with $450 filing fee to the City Clerk
2. **Required Documents**: Copy of original decision, grounds for appeal, supporting evidence
3. **Hearing**: Appeals heard by the Zoning Board of Appeals on the 2nd Tuesday of each month
4. **Timeline**: Board must decide within 60 days of complete application

Appeals must be based on:
• Procedural errors in the original decision
• New evidence not previously considered
• Misapplication of zoning code provisions

Contact the City Clerk at (555) 123-4500 for appeal forms and scheduling.`;

                    sources = [
                        {
                            title: 'Zoning Appeals Process - Chapter 17',
                            doc_type: 'zoning',
                            relevance: 0.89,
                            section: 'Section 17.8',
                            effective_date: '2023-09-15'
                        }
                    ];
                } else {
                    answer = `I found information related to your query in the city ordinances. However, for the most accurate and up-to-date information specific to your situation, I recommend:

• Contacting the relevant city department directly
• Reviewing the complete ordinance text for detailed requirements
• Consulting with a local professional if needed

Please try rephrasing your question or use one of the example queries to get more specific guidance.`;

                    sources = [
                        {
                            title: 'General Municipal Code',
                            doc_type: 'general',
                            relevance: 0.65,
                            section: 'Various',
                            effective_date: '2024-01-01'
                        }
                    ];
                    confidence = 0.6;
                }

                return {
                    query_id: Date.now(),
                    answer,
                    sources,
                    confidence,
                    similar_past_queries: []
                };
            }

            showLoading() {
                document.getElementById('loadingIndicator').style.display = 'block';
                document.getElementById('resultsContainer').style.display = 'none';
            }

            hideLoading() {
                document.getElementById('loadingIndicator').style.display = 'none';
                document.getElementById('resultsContainer').style.display = 'block';
            }

            displayResult(query, result) {
                this.hideLoading();
                this.currentQueryId = result.query_id;

                const resultsContainer = document.getElementById('resultsContainer');

                // Confidence badge class
                let confidenceClass = 'confidence-low';
                let confidenceText = 'Low Confidence';
                if (result.confidence > 0.8) {
                    confidenceClass = 'confidence-high';
                    confidenceText = 'High Confidence';
                } else if (result.confidence > 0.6) {
                    confidenceClass = 'confidence-medium';
                    confidenceText = 'Medium Confidence';
                }

                resultsContainer.innerHTML = `
                    <div class="answer-card">
                        <div class="answer-text">${result.answer}</div>
                        <div class="confidence-badge ${confidenceClass}">
                            ${confidenceText} (${Math.round(result.confidence * 100)}%)
                        </div>

                        <div class="feedback-section">
                            <p style="font-size: 0.9rem; color: #666; margin-bottom: 0.5rem;">Was this answer helpful?</p>
                            <div class="feedback-buttons">
                                <button class="feedback-btn positive" onclick="cogsInterface.provideFeedback(${result.query_id}, true)">
                                    👍 Yes, helpful
                                </button>
                                <button class="feedback-btn negative" onclick="cogsInterface.provideFeedback(${result.query_id}, false)">
                                    👎 Not helpful
                                </button>
                            </div>
                        </div>
                    </div>

                    ${result.sources.length > 0 ? `
                        <div class="sources">
                            <h3>📚 Sources (${result.sources.length} documents)</h3>
                            ${result.sources.map(source => `
                                <div class="source-item">
                                    <div class="source-title">${source.title}</div>
                                    <div class="source-meta">
                                        Type: ${source.doc_type.charAt(0).toUpperCase() + source.doc_type.slice(1)} |
                                        Relevance: ${Math.round(source.relevance * 100)}% |
                                        ${source.section} |
                                        Effective: ${source.effective_date}
                                    </div>
                                </div>
                            `).join('')}
                        </div>
                    ` : ''}
                `;

                // Scroll to results
                resultsContainer.scrollIntoView({ behavior: 'smooth' });

                // Update query count
                this.incrementQueryCount();
            }

            displayError(message) {
                this.hideLoading();
                const resultsContainer = document.getElementById('resultsContainer');

                resultsContainer.innerHTML = `
                    <div style="text-align: center; color: #666; margin-top: 2rem;">
                        <h3 style="color: #dc3545;">⚠️ Error</h3>
                        <p>${message}</p>
                    </div>
                `;
            }

            clearResults() {
                document.getElementById('queryInput').value = '';
                document.getElementById('resultsContainer').innerHTML = `
                    <div style="text-align: center; color: #666; margin-top: 3rem;">
                        <h2>Welcome to COGS</h2>
                        <p>Your intelligent assistant for city ordinances and building codes. Ask any question about local regulations, permits, or compliance requirements.</p>
                        <div style="margin-top: 2rem; padding: 1.5rem; background: #f8f9ff; border-radius: 10px; text-align: left;">
                            <h3 style="color: #495057; margin-bottom: 1rem;">💡 How it works:</h3>
                            <ul style="color: #666; line-height: 1.6; padding-left: 1.5rem;">
                                <li><strong>Privacy First:</strong> All data stays local - your queries and information remain private</li>
                                <li><strong>Comprehensive Search:</strong> Searches through all city ordinances, building codes, and zoning regulations</li>
                                <li><strong>Transparent Sources:</strong> Every answer includes citations to specific ordinances and sections</li>
                                <li><strong>Learning System:</strong> Improves over time based on your feedback</li>
                            </ul>
                        </div>
                    </div>
                `;
            }

            async provideFeedback(queryId, helpful) {
                try {
                    // In production, this would call your feedback API
                    console.log(`Feedback for query ${queryId}: ${helpful ? 'helpful' : 'not helpful'}`);

                    // Update UI to show feedback was recorded
                    const buttons = document.querySelectorAll('.feedback-btn');
                    buttons.forEach(btn => {
                        btn.classList.remove('selected');
                        if ((helpful && btn.classList.contains('positive')) ||
                            (!helpful && btn.classList.contains('negative'))) {
                            btn.classList.add('selected');
                            btn.innerHTML = helpful ? '✓ Thank you!' : '✓ Feedback recorded';
                        }
                    });

                    // Simulate API call
                    await new Promise(resolve => setTimeout(resolve, 300));

                } catch (error) {
                    console.error('Feedback error:', error);
                }
            }

            loadStats() {
                // Simulate loading stats
                setTimeout(() => {
                    document.getElementById('docCount').textContent = '847';
                    document.getElementById('queryCount').textContent = localStorage.getItem('cogsQueryCount') || '0';
                }, 500);
            }

            incrementQueryCount() {
                const current = parseInt(localStorage.getItem('cogsQueryCount') || '0');
                const newCount = current + 1;
                localStorage.setItem('cogsQueryCount', newCount.toString());
                document.getElementById('queryCount').textContent = newCount.toString();
            }
        }

        // Initialize the interface when page loads
        let cogsInterface;
        document.addEventListener('DOMContentLoaded', () => {
            cogsInterface = new COGSInterface();
        });
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/query', methods=['POST'])
def api_query():
    data = request.get_json()
    query = data.get('query', '').strip()
    
    if not query:
        return jsonify({'error': 'Query required'}), 400
    
    try:
        result = cogs.query(query)
        
        return jsonify({
            'answer': result['answer'],
            'confidence': result['confidence'],
            'sources': [
                {
                    'title': s['title'],
                    'doc_type': s['doc_type'],
                    'similarity': s['similarity']
                }
                for s in result['sources'][:3]  # Limit to top 3
            ]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🌐 Starting COGS web interface...")
    print("📍 Visit: http://localhost:5000")
    app.run(debug=True, host='http://127.0.0.1', port=4040)
