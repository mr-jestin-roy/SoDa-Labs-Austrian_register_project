# 🚀 HandwritingOCR.com Webhook Setup Guide

## 📋 Quick Setup Steps

### 1. Install Dependencies
```bash
pip install flask requests python-dotenv
```

### 2. Configure Environment Variables
Copy `config.example` to `.env` and add your API token:
```bash
cp config.example .env
```

Edit `.env`:
```bash
HANDWRITING_OCR_API_TOKEN=your_actual_token_here
```

### 3. Start Webhook Server
```bash
python webhook_server.py
```

This will start the server on port 5400 to match your ngrok configuration.

### 4. Configure Webhook in HandwritingOCR.com

1. Go to: https://www.handwritingocr.com/settings/documents
2. Set webhook URL to: `https://5400-110-174-183-131.ngrok-free.app/webhook/handwriting-ocr`
3. Save settings

## 🧪 Testing

### Test complete pipeline with CSV output:
```bash
python two_model_api_pipeline.py
```

This will:
- Process all images in `api_test_assets/test_input_images_folder/`
- Compare HandwritingOCR vs Transkribus results  
- Generate `ocr_compare_results.csv` with analysis

## 📊 Webhook vs Polling Comparison

| Method | Speed | Efficiency | Real-time |
|--------|-------|------------|-----------|
| **Webhook** | ⚡ Fast | ✅ High | ✅ Yes |
| **Polling** | 🐌 Slow | ❌ Low | ❌ No |

## 🔧 How It Works

1. **Submit Image**: API call with `webhook_url` parameter
2. **Get Document ID**: HandwritingOCR returns document ID immediately
3. **Wait for Webhook**: Your server receives result when processing completes
4. **Return Text**: Function returns transcribed text

## 🚨 Troubleshooting

### Webhook not receiving data:
- Check ngrok is running: `ngrok http 5400`
- Verify webhook URL in HandwritingOCR.com settings
- Check webhook server logs for errors

### API submission fails:
- Verify API token in `.env` file
- Check credit balance at HandwritingOCR.com
- Ensure image file exists and is readable

### Fallback to polling:
- If webhook fails, system automatically falls back to polling
- No data loss, just slower processing

## 📈 Monitoring

Watch webhook server logs for real-time updates:
```
📨 WEBHOOK RECEIVED
🆔 Document ID: abc123
📊 Status: completed
✅ SUCCESS! Text length: 245 chars
```

## 🔗 Endpoints

- **Webhook**: `https://5400-110-174-183-131.ngrok-free.app/webhook/handwriting-ocr`
- **Check Results**: `http://localhost:5400/results`
- **Health Check**: `http://localhost:5400/webhook/handwriting-ocr` (GET)

---
*Your ngrok URL: https://5400-110-174-183-131.ngrok-free.app* 