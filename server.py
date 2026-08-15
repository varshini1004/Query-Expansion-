from flask import Flask, render_template, request, jsonify
import asyncio
from query_expander import NeuralExpander

app = Flask(__name__)
expander = NeuralExpander()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/expand', methods=['POST'])
def expand():
    data = request.json
    query = data.get('query', '')
    
    if not query:
        return jsonify({'error': 'No query provided'}), 400
    
    try:
        # Run the query expansion
        result = asyncio.run(expander.expand_query(query))
        
        # Return ALL expanded terms
        return jsonify({
            'expanded_terms': result.get('expanded_terms', []),
            'metrics': result.get('metrics', {}),
            'perplexity': result.get('perplexity'),
            'domain': result.get('detected_domain', 'general'),
            'total_candidates': result.get('total_candidates', 0),
            'total_expanded': result.get('total_expanded', 0)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 8000))
    app.run(host='0.0.0.0', port=port, debug=False)