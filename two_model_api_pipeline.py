import os
import pandas as pd
import editdistance
import requests
import base64
import time
from transkribus_client import TranskribusClient
from dotenv import load_dotenv
import google.generativeai as genai
from concurrent.futures import ThreadPoolExecutor, as_completed
load_dotenv()

def cer(s1, s2):
    return editdistance.eval(s1, s2) / max(1, len(s2))

def call_handwritingocr_api(image_path, use_webhook=True):
    """
    Call HandwritingOCR.com API with webhook support for faster processing
    
    Args:
        image_path: Path to image file
        use_webhook: Whether to use webhook (faster) or polling
    """
    # Get API token
    api_token = os.getenv("HANDWRITING_OCR_API_TOKEN")
    if not api_token:
        raise Exception("HANDWRITING_OCR_API_TOKEN environment variable must be set.")
    
    # Webhook URL (your ngrok URL)
    webhook_url = "https://5400-110-174-183-131.ngrok-free.app/webhook/handwriting-ocr"
    
    # API endpoint
    url = "https://www.handwritingocr.com/api/v3/documents"
    
    # Headers
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Accept": "application/json"
    }
    
    # Import webhook function if available
    webhook_available = False
    try:
        from webhook_server import wait_for_webhook_result
        webhook_available = True
    except ImportError:
        pass
    
    # Prepare submission data
    with open(image_path, 'rb') as file:
        files = {
            'file': (os.path.basename(image_path), file, 'image/jpeg')
        }
        
        # Form data
        data = {
            'action': 'transcribe',
            'delete_after': '604800'  # 7 days
        }
        
        # Add webhook URL if using webhooks and webhook server is available
        if use_webhook and webhook_available:
            data['webhook_url'] = webhook_url
            print(f"📤 Submitting with webhook: {os.path.basename(image_path)}")
        else:
            print(f"📤 Submitting without webhook: {os.path.basename(image_path)}")
        
        # Submit document
        response = requests.post(url, headers=headers, files=files, data=data)
        
        if response.status_code not in [200, 201]:  # Accept both 200 OK and 201 Created
            raise Exception(f"HandwritingOCR API submission failed: {response.status_code} - {response.text}")
        
        result = response.json()
        document_id = result.get('id')
        
        if not document_id:
            raise Exception("No document ID returned from HandwritingOCR API")
        
        print(f"✅ Document submitted! ID: {document_id}")
    
    # Wait for result
    if use_webhook and webhook_available:
        print("⏳ Waiting for webhook result...")
        try:
            result = wait_for_webhook_result(document_id, timeout=20)  # 30 sec timeout
            
            if result:
                if result.get('status') == 'processed':  # Correct status name
                    # Extract text from results array (single image)
                    results = result.get('results', [])
                    if results and len(results) > 0:
                        # Get transcript from first (and only) page
                        transcript = results[0].get('transcript', '').strip()
                        print(f"✅ Webhook success! Text length: {len(transcript)} characters")
                        return transcript
                    else:
                        print("⚠️ No transcription results found in webhook")
                        return ""
                else:
                    raise Exception(f"Processing failed: {result.get('error', 'Unknown error')}")
            else:
                print("⚠️ Webhook timeout, falling back to polling...")
                return _poll_handwriting_ocr_result(document_id, headers)
        except Exception as e:
            print(f"⚠️ Webhook error ({e}), falling back to polling...")
            return _poll_handwriting_ocr_result(document_id, headers)
    else:
        # Use polling method
        return _poll_handwriting_ocr_result(document_id, headers)

def _poll_handwriting_ocr_result(document_id, headers):
    """Poll HandwritingOCR API for result (fallback method)"""
    status_url = f"https://www.handwritingocr.com/api/v3/documents/{document_id}"
    
    print("📊 Polling for HandwritingOCR result...")
    
    while True:
        response = requests.get(status_url, headers=headers)
        
        if response.status_code != 200:
            raise Exception(f"Status check failed: {response.status_code} - {response.text}")
        
        data = response.json()
        status = data.get('status', '')
        
        print(f"📋 HandwritingOCR status: {status}")
        
        if status == 'processed':  # Correct status name
            # Extract text from results array (single image)
            results = data.get('results', [])
            if results and len(results) > 0:
                # Get transcript from first (and only) page
                transcript = results[0].get('transcript', '').strip()
                print(f"✅ HandwritingOCR completed! Text length: {len(transcript)} characters")
                return transcript
            else:
                print("⚠️ No transcription results found")
                return ""
        elif status == 'failed':
            error = data.get('error', 'Processing failed')
            raise Exception(f"HandwritingOCR processing failed: {error}")
        elif status in ['processing', 'queued', 'pending']:
            print("⏳ Still processing... waiting 10 seconds")
            time.sleep(10)
        else:
            print(f"⚠️ Unknown status: {status}, waiting 10 seconds")
            time.sleep(10)

def call_transkribus_api(image_path):
    # ---- Get credentials from environment ----
    username = os.getenv("TRANSKRIBUS_USERNAME", None)
    password = os.getenv("TRANSKRIBUS_PASSWORD", None)
    access_token = os.getenv('TRANSKRIBUS_ACCESS_TOKEN', None)
    if access_token:
        print("Using access token from environment variable")
    else:
        print("No access token found, using username and password for authentication")
        if not username or not password:
         raise Exception("TRANSKRIBUS_USERNAME and TRANSKRIBUS_PASSWORD environment variables must be set.")

        # ---- Step 1: Authenticate ----
        token_url = 'https://account.readcoop.eu/auth/realms/readcoop/protocol/openid-connect/token'
        payload = {
            'grant_type': 'password',
            'username': username,
            'password': password,
            'client_id': 'processing-api-client'
        }
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}

        r = requests.post(token_url, data=payload, headers=headers)
        if r.status_code != 200:
            raise Exception(f"Authentication failed: {r.status_code} - {r.text}")
        
        token_data = r.json()
        access_token = token_data['access_token']
        

        refresh_token = token_data.get('refresh_token')  # Store for potential future use
        
        print(f"✅ Authentication successful! Token expires in: {token_data.get('expires_in', 'unknown')} seconds")
     

    # ---- Step 2: Encode image as base64 ----
    with open(image_path, 'rb') as f:
        img_bytes = f.read()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

    # ---- Step 3: Submit data for processing ----
    process_url = "https://transkribus.eu/processing/v1/processes"
    
    # Use the correct API payload format from official documentation
    process_payload = {
        "config": {
            "textRecognition": {
            "htrId": int(os.getenv('TRANSKRIBUS_HTR_MODEL_ID', "No HTR Model ID"))  # HTR Model ID
            }
        },
        "image": {
            "base64": img_b64
        }
    }
    process_headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    pr = requests.post(process_url, json=process_payload, headers=process_headers)
    if pr.status_code != 200:
        raise Exception(f"Image submission failed: {pr.status_code} - {pr.text}")
    process_id = pr.json()["processId"]
    print(f"Image submitted successfully! Process ID: {process_id}")

    # ---- Step 4: Poll for status and retrieve result ----
    status_url = f"https://transkribus.eu/processing/v1/processes/{process_id}"
    while True:
        status_r = requests.get(status_url, headers={"Authorization": f"Bearer {access_token}"})
        if status_r.status_code != 200:
            raise Exception(f"Status check failed: {status_r.status_code} - {status_r.text}")
        status_json = status_r.json()
        status = status_json.get("status", "")
        print(f"Processing status: {status}")
        
        if status == "FINISHED":
            text = status_json.get("content", {}).get("text", "")
            print(f"HTR processing completed! Text length: {len(text)} characters")
            return text  # Return just the text string for your pipeline
        elif status == "FAILED":
            raise Exception("Processing failed.")
        time.sleep(5)


def call_gemini_api(img_path):
    """
    Call Google Gemini 2.5 Pro model API for OCR functionality
    
    Args:
        img_path: Path to image file
        
    Returns:
        str: Extracted text from the image
    """
    # API Key for Google Gemini 2.5 Pro
    api_key = "AIzaSyAhE8j0_EArjdNDfevIbuY9OelGOpIW14c"
    if not api_key:
        raise Exception("GEMINI_API_KEY environment variable must be set.")
    
    print(f"📤 Submitting to Google Gemini 2.5 Pro: {os.path.basename(img_path)}")
    
    # Configure Gemini API
    genai.configure(api_key=api_key)
    
    # Read image file
    with open(img_path, "rb") as f:
        data = f.read()
        print(f"Read {len(data)} bytes from {img_path}")
    
    try:
        # Initialize Gemini model
        model = genai.GenerativeModel('gemini-2.5-pro')
        
        # Prepare the prompt
        prompt = """
                    Transcribe all text from the attached image based on the following rules:

                    Transcribe Everything: The image contains a mix of printed Fraktur (blackletter) and handwritten Kurrent/Sütterlin script. You must transcribe both.
                    Raw Text Only: Return the raw German transcription exactly as written.
                    Do NOT translate.
                    Do NOT modernize or "correct" any historical spelling.
                    Preserve all original abbreviations and punctuation.
                    Preserve Table Structure:
                    Use new lines for each new row in the register.
                    Use a single vertical pipe (|) to separate the text from each distinct column.

                    Handle Edge Cases:
                    Transcribe all text, including small-print instructions and any text printed vertically.
                    If a word, number, or section is completely unreadable, output [illegible].

                    Output only the raw transcription.
                """
        
        # Create image part
        image_part = {
            "mime_type": "image/jpeg",
            "data": data
        }
        
        # Generate content
        response = model.generate_content([prompt, image_part])
        
        # Extract the transcribed text
        transcribed_text = response.text.strip()
        print(f"API output: '{transcribed_text}'")
        print(f"✅ Google Gemini 2.5 Pro completed! Text length: {len(transcribed_text)} characters")
        
        # Rate limiting handled at batch level for parallel processing
        
        return transcribed_text
        
    except Exception as e:
        print(f"❌ Google Gemini 2.5 Pro API error: {str(e)}")
        raise Exception(f"Google Gemini 2.5 Pro API call failed: {str(e)}")


def word_match(w1, w2, cer_threshold=0.3):
    return cer(w1, w2) <= cer_threshold

def align_and_flag_illegible(text1, text2, cer_threshold=0.3):
    words1 = text1.strip().split()
    words2 = text2.strip().split()
    max_len = max(len(words1), len(words2))
    final_words = []
    for i in range(max_len):
        w1 = words1[i] if i < len(words1) else ""
        w2 = words2[i] if i < len(words2) else ""
        if w1 and w2 and word_match(w1, w2, cer_threshold):
            final_words.append(w1)  # or w2, since they're similar
        else:
            final_words.append("[illegible]")
    return " ".join(final_words)

# Settings
image_folder = "api_test_assets/test_input_images_folder/Althofen_TrauungsbuchTomIV_1907_1936_00015"
output_csv = "Althofen_TrauungsbuchTomIV_1907_1936_00015_ground_truth_gemini.csv"
line_cer_threshold = 0.10   # CER threshold for whole line (not used for word-level flagging)
word_cer_threshold = 0.3    # CER threshold for word-level matching
start = time.time()

images = sorted([f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])

def process_single_image(img_name):
    """Process a single image and return the result"""
    img_path = os.path.join(image_folder, img_name)
    print(f"{img_name}: {os.path.getsize(img_path)} bytes")
    
    try:
        gemini_text = call_gemini_api(img_path)  # Using Google Gemini 2.5 Pro model
        
        return {
            "image": 'Althofen_TrauungsbuchTomIV_1907_1936_00015_' + img_name,
            "gemini_text": gemini_text,  # Using Google Gemini 2.5 Pro
            "cer": 'N/A',
            "final_ground_truth": ''
        }
    except Exception as e:
        print(f"❌ Error processing {img_name}: {str(e)}")
        return {
            "image": 'Althofen_TrauungsbuchTomIV_1907_1936_00015_' + img_name,
            "gemini_text": f"ERROR: {str(e)}",
            "cer": 'N/A',
            "final_ground_truth": ''
        }

# Process images in parallel batches of 5
results = []
batch_size = 5
max_workers = 5

print(f"📊 Processing {len(images)} images in parallel batches of {batch_size}...")

for i in range(0, len(images), batch_size):
    batch = images[i:i + batch_size]
    print(f"\n🔄 Processing batch {i//batch_size + 1}: {len(batch)} images")
    
    # Process batch in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks in the batch
        future_to_img = {executor.submit(process_single_image, img_name): img_name for img_name in batch}
        
        # Collect results as they complete
        batch_results = []
        for future in as_completed(future_to_img):
            img_name = future_to_img[future]
            try:
                result = future.result()
                batch_results.append(result)
                print(f"✅ Completed: {img_name}")
            except Exception as e:
                print(f"❌ Failed: {img_name} - {str(e)}")
                batch_results.append({
                    "image": 'Althofen_TrauungsbuchTomIV_1907_1936_00015_' + img_name,
                    "gemini_text": f"BATCH_ERROR: {str(e)}",
                    "cer": 'N/A',
                    "final_ground_truth": ''
                })
    
    results.extend(batch_results)
    
    # No rate limiting - process batches immediately

df = pd.DataFrame(results)
df.to_csv(output_csv, index=False)
print(f"Saved auto-annotated CSV to {output_csv}")
end = time.time()
print(f"Time taken: {end - start} seconds")

# if __name__ == "__main__":
#     # Test with a single image
#     test_image = "api_test_assets/test_input_images_folder/Althofen_TrauungsbuchTomIV_1907_1936_00046_line_0010.jpg"
    
#     if os.path.exists(test_image):
#         print(f"🔍 Testing Transkribus API with: {test_image}")
#         try:
#             result = call_transkribus_api(test_image)  # Remove image_path= parameter name
#             print(f"✅ Success! Transcribed text:\n{result}")
#         except Exception as e:
#             print(f"❌ Error: {e}")
#     else:
#         print(f"❌ Test image not found: {test_image}")