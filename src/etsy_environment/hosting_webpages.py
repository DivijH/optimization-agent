from flask import Flask, send_from_directory, request, jsonify, redirect, url_for
import os
import json
import random
from datetime import datetime

app = Flask(__name__)

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
ETSY_PAGES_DIR = os.path.join(DATA_DIR, "etsy_pages")
VARIANTS_DIR = os.path.join(ETSY_PAGES_DIR, "variants")
ANALYTICS_FILE = os.path.join(DATA_DIR, "ab_test_results.json")

# In-memory storage for user sessions
user_sessions = {}

def get_user_variant(user_id):
    """Assign user to control or test group"""
    if user_id not in user_sessions:
        # 50/50 split between control and test
        variant = random.choice(['control', 'test'])
        user_sessions[user_id] = {
            'variant': variant,
            'start_time': datetime.now().isoformat()
        }
    return user_sessions[user_id]['variant']

def save_event(event_data):
    """Save tracking event to file"""
    try:
        # Load existing data
        if os.path.exists(ANALYTICS_FILE):
            with open(ANALYTICS_FILE, 'r') as f:
                data = json.load(f)
        else:
            data = []
        
        # Add new event
        data.append(event_data)
        
        # Save back to file
        with open(ANALYTICS_FILE, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Error saving event: {e}")

@app.route('/')
def index():
    """Main entry point - redirect to appropriate variant"""
    user_id = request.args.get('user_id', 'default_user')
    variant = get_user_variant(user_id)
    return redirect(url_for('serve_variant', variant=variant, user_id=user_id))

@app.route('/variant/<variant>/', strict_slashes=False)
def serve_variant(variant):
    """Serve the appropriate variant"""
    user_id = request.args.get('user_id', 'default_user')
    if variant not in ['control', 'test']:
        variant = 'control'
    
    variant_dir = os.path.join(VARIANTS_DIR, variant)
    if not os.path.exists(variant_dir):
        return f"Variant {variant} not found", 404
    
    return send_from_directory(variant_dir, 'index.html')

@app.route('/variant/<variant>/<path:filename>')
def serve_variant_assets(variant, filename):
    """Serve static assets for variants"""
    variant_dir = os.path.join(VARIANTS_DIR, variant)
    return send_from_directory(variant_dir, filename)

@app.route('/etsy/<path:page_path>/', strict_slashes=False)
def serve_etsy_page(page_path):
    """Serve Etsy pages from the data directory"""
    page_dir = os.path.join(ETSY_PAGES_DIR, page_path)
    if not os.path.exists(page_dir):
        return f"Etsy page {page_path} not found", 404
    
    return send_from_directory(page_dir, 'index.html')

@app.route('/etsy/<path:page_path>/<path:filename>')
def serve_etsy_assets(page_path, filename):
    """Serve static assets for Etsy pages"""
    page_dir = os.path.join(ETSY_PAGES_DIR, page_path)
    return send_from_directory(page_dir, filename)

@app.route('/api/track-event', methods=['POST'])
def track_event():
    """API endpoint to receive tracking events"""
    try:
        event_data = request.json
        event_data['server_timestamp'] = datetime.now().isoformat()
        
        save_event(event_data)
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/analytics')
def view_analytics():
    """View analytics dashboard"""
    try:
        if not os.path.exists(ANALYTICS_FILE):
            return jsonify({'message': 'No analytics data available'})
        
        with open(ANALYTICS_FILE, 'r') as f:
            events = json.load(f)
        
        # Basic analytics
        control_events = [e for e in events if e.get('variant') == 'control']
        test_events = [e for e in events if e.get('variant') == 'test']
        
        analytics = {
            'total_events': len(events),
            'control_events': len(control_events),
            'test_events': len(test_events),
            'control_clicks': len([e for e in control_events if e.get('eventType') == 'click']),
            'test_clicks': len([e for e in test_events if e.get('eventType') == 'click']),
            'recent_events': events[-10:]  # Last 10 events
        }
        
        return jsonify(analytics)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/reset-analytics', methods=['POST'])
def reset_analytics():
    """Reset analytics data"""
    try:
        if os.path.exists(ANALYTICS_FILE):
            os.remove(ANALYTICS_FILE)
        user_sessions.clear()
        return jsonify({'status': 'Analytics reset successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs(VARIANTS_DIR, exist_ok=True)
    
    print("A/B Test Server starting...")
    print("Control variant: http://localhost:5000/?user_id=user1")
    print("Sample Etsy page: http://localhost:5000/etsy/search_infltable halloween spider_page_1")
    print("Analytics dashboard: http://localhost:5000/analytics")
    print("Reset analytics: POST to http://localhost:5000/reset-analytics")
    
    app.run(debug=True, port=5000)