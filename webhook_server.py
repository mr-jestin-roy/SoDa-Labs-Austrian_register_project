#!/usr/bin/env python3
"""
Simple webhook server for HandwritingOCR.com API results
Run this with: python webhook_server.py
Then use webhook URL: https://5400-110-174-183-131.ngrok-free.app/webhook/handwriting-ocr
"""

from flask import Flask, request, jsonify
import json
import threading
import time
from datetime import datetime

app = Flask(__name__)

# Store results in memory (use database in production)
webhook_results = {}
result_lock = threading.Lock()

@app.route('/webhook/handwriting-ocr', methods=['POST'])
def handwriting_ocr_webhook():
    """Receive HandwritingOCR.com webhook results"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data"}), 400
        
        document_id = data.get('id')
        status = data.get('status')
        
        print(f"\n📨 WEBHOOK RECEIVED")
        print(f"🆔 Document ID: {document_id}")
        print(f"📊 Status: {status}")
        print(f"🕐 Time: {datetime.now().strftime('%H:%M:%S')}")
        
        # Store result
        with result_lock:
            webhook_results[document_id] = data
        
        if status == 'processed':  # Correct status name
            # Extract text from results (single image)
            results = data.get('results', [])
            if results and len(results) > 0:
                transcript = results[0].get('transcript', '').strip()
                print(f"✅ SUCCESS! Text length: {len(transcript)} chars")
                print(f"📝 German handwriting preview: {transcript[:80]}{'...' if len(transcript) > 80 else ''}")
            else:
                print(f"⚠️ SUCCESS but no transcription found")
        elif status == 'failed':
            print(f"❌ FAILED: {data.get('error', 'Unknown error')}")
        
        return jsonify({"status": "received"}), 200
        
    except Exception as e:
        print(f"❌ Webhook error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/webhook/handwriting-ocr', methods=['GET'])
def webhook_info():
    """Info about the webhook endpoint"""
    return jsonify({
        "message": "HandwritingOCR.com webhook endpoint",
        "url": "https://5400-110-174-183-131.ngrok-free.app/webhook/handwriting-ocr",
        "pending_results": len(webhook_results)
    })

@app.route('/results/<document_id>')
def get_result(document_id):
    """Get result for specific document"""
    with result_lock:
        result = webhook_results.get(document_id)
    
    if result:
        return jsonify(result)
    else:
        return jsonify({"error": "Result not found"}), 404

@app.route('/results')
def list_results():
    """List all results"""
    with result_lock:
        return jsonify({
            "total": len(webhook_results),
            "results": list(webhook_results.keys())
        })

def wait_for_webhook_result(document_id, timeout=20):
    """Wait for webhook result (used by API function)"""
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        with result_lock:
            if document_id in webhook_results:
                result = webhook_results.pop(document_id)  # Remove after getting
                return result
        time.sleep(2)
    
    return None

if __name__ == '__main__':
    print("🚀 HandwritingOCR Webhook Server Starting...")
    print("📍 Local endpoint: http://localhost:5400/webhook/handwriting-ocr")
    print("🌐 Public endpoint: https://5400-110-174-183-131.ngrok-free.app/webhook/handwriting-ocr")
    print("\n📋 Configure this URL in HandwritingOCR.com settings:")
    print("   https://www.handwritingocr.com/settings/documents")
    print("\n⚡ Server ready! Waiting for webhooks...")
    
    app.run(host='0.0.0.0', port=5400, debug=False)  # Match ngrok port 