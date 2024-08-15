from flask import Blueprint, render_template, request, jsonify
from .utils import generate_plots

main = Blueprint('main', __name__)

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    url = data.get('url')
    
    print(f"Received URL: {url}")  # Debugging line

    # Generate the plots and get the Plotly graph JSON
    graphData = generate_plots([url])
    
    print(f"Generated Graph JSON: {graphData}")  # Debugging line
    
    if graphData is not None:
        print(f"graphData2D: {graphData['graphData2D']}")  
        print(f"graphData3D: {graphData['graphData3D']}")  
        return jsonify({
            'success': True,
            'graphData2D': graphData['graphData2D'],
            'graphData3D': graphData['graphData3D']
        })
    else:
        return jsonify({
            'success': False,
            'error': "Failed to generate plots."
        })


