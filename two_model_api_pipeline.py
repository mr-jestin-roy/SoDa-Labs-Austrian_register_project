import os
import pandas as pd
import editdistance
import requests
import base64
import time
from transkribus_client import TranskribusClient
from dotenv import load_dotenv
load_dotenv()

def cer(s1, s2):
    return editdistance.eval(s1, s2) / max(1, len(s2))

def call_handwritingocr_api(image_path):
    # Placeholder for actual API call
    return "recognized text hwocr"

def call_transkribus_api(image_path):
    # ---- Get credentials from environment ----
    username = os.getenv("TRANSKRIBUS_USERNAME", None)
    password = os.getenv("TRANSKRIBUS_PASSWORD", None)
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

    # r = requests.post(token_url, data=payload, headers=headers)
    # if r.status_code != 200:
    #     raise Exception(f"Authentication failed: {r.status_code} - {r.text}")
    
    # token_data = r.json()
    # access_token = token_data['access_token']
    access_token = os.getenv('TRANSKRIBUS_ACCESS_TOKEN', None)

    # refresh_token = token_data.get('refresh_token')  # Store for potential future use
    
    # print(f"✅ Authentication successful! Token expires in: {token_data.get('expires_in', 'unknown')} seconds")


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
image_folder = "api_test_assets/test_input_images_folder"
output_csv = "ocr_compare_results.csv"
line_cer_threshold = 0.10   # CER threshold for whole line (not used for word-level flagging)
word_cer_threshold = 0.3    # CER threshold for word-level matching

images = sorted([f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])

# results = []
# for img_name in images:
#     img_path = os.path.join(image_folder, img_name)
#     hwocr_text = call_handwritingocr_api(img_path)
#     transkribus_text = call_transkribus_api(img_path)
#     cer_score = cer(hwocr_text, transkribus_text)
#     final_gt = align_and_flag_illegible(hwocr_text, transkribus_text, cer_threshold=word_cer_threshold)
#     results.append({
#         "image": img_name,
#         "hwocr_text": hwocr_text,
#         "transkribus_text": transkribus_text,
#         "cer": cer_score,
#         "final_ground_truth": final_gt
#     })

# df = pd.DataFrame(results)
# df.to_csv(output_csv, index=False)
# print(f"Saved auto-annotated CSV to {output_csv}")

if __name__ == "__main__":
    # Test with a single image
    test_image = "api_test_assets/test_input_images_folder/Althofen_TrauungsbuchTomIV_1907_1936_00015_line_0010.jpg"
    
    if os.path.exists(test_image):
        print(f"🔍 Testing Transkribus API with: {test_image}")
        try:
            result = call_transkribus_api(test_image)  # Remove image_path= parameter name
            print(f"✅ Success! Transcribed text:\n{result}")
        except Exception as e:
            print(f"❌ Error: {e}")
    else:
        print(f"❌ Test image not found: {test_image}")