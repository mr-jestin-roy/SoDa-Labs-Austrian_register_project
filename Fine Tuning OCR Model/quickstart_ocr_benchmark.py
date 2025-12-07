"""
Quick Start OCR Benchmark - Minimal Version
Test with a single model first to verify setup
"""

import os
import pandas as pd
from PIL import Image
from jiwer import wer, cer
from tqdm import tqdm

# =============================================================================
# QUICK CONFIGURATION - EDIT THESE
# =============================================================================

INPUT_FOLDER = "api_test_assets/test_input_images_folder/Althofen_TrauungsbuchTomIV_1907_1936_00046"  # Your images folder
GROUND_TRUTH_CSV = "Althofen_TrauungsbuchTomIV_1907_1936_00046_ground_truth_gemini.csv"  # Your CSV file
OUTPUT_FILE = "Fine Tuning OCR Model/output_results/quick_deepseek_ocr_3b_results.csv"

# Choose ONE backend to start with
USE_VLLM = True  # Set to False to use Ollama instead

# =============================================================================
# vLLM SETUP (Recommended)
# =============================================================================

if USE_VLLM:
    print("Loading vLLM model...")
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    # Start with the smallest model - DeepSeek OCR 3B
    MODEL_NAME = "deepseek-ai/DeepSeek-OCR"

    llm = LLM(
        enable_prefix_caching=False,  # Required for DeepSeek-OCR
        mm_processor_cache_gb=0,  # Required for DeepSeek-OCR
        trust_remote_code=True,
        max_model_len=2048,
    )
    CUSTOM_PROMPT = """Transcribe all text from the attached image based on the following rules:

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

        Output only the raw transcription."""

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    def run_ocr(image_path):
        """Run OCR using vLLM"""
        image = Image.open(image_path).convert("RGB")

        prompt = f"<image>\n{CUSTOM_PROMPT}"  # DeepSeek format

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=1024
        )

        outputs = llm.generate(
            {
                "prompt": prompt,
                "multi_modal_data": {"image": image}
            },
            sampling_params=sampling_params
        )

        return outputs[0].outputs[0].text.strip()

# =============================================================================
# OLLAMA SETUP (Alternative - Easier but Slower)
# =============================================================================

else:
    print("Using Ollama...")
    import ollama

    MODEL_NAME = "llama3.2-vision"  # Change to your preferred model

    # Try to pull model if not available
    try:
        ollama.show(MODEL_NAME)
    except:
        print(f"Pulling {MODEL_NAME}...")
        ollama.pull(MODEL_NAME)

    def run_ocr(image_path):
        """Run OCR using Ollama"""
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[{
                'role': 'user',
                'content': 'Extract all text from this image exactly as it appears:',
                'images': [image_path]
            }]
        )
        return response['message']['content'].strip()

# =============================================================================
# BENCHMARK EXECUTION
# =============================================================================

def main():
    # Load ground truth
    print(f"Loading ground truth from {GROUND_TRUTH_CSV}...")
    df = pd.read_csv(GROUND_TRUTH_CSV)
    print(f"Found {len(df)} samples")

    # Results storage
    results = []

    # Process each image
    print(f"\nProcessing images with {MODEL_NAME}...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        image_filename = row['image']
        ground_truth = str(row['gemini_text'])

        image_path = os.path.join(INPUT_FOLDER, image_filename)

        if not os.path.exists(image_path):
            print(f"Warning: {image_path} not found")
            continue

        try:
            # Run OCR
            prediction = run_ocr(image_path)

            # Calculate metrics
            cer_score = cer(ground_truth, prediction)
            wer_score = wer(ground_truth, prediction)

            results.append({
                'image': image_filename,
                'ground_truth': ground_truth,
                'prediction': prediction,
                'cer': cer_score,
                'wer': wer_score
            })

        except Exception as e:
            print(f"Error on {image_filename}: {e}")
            results.append({
                'image': image_filename,
                'ground_truth': ground_truth,
                'prediction': "",
                'cer': 1.0,
                'wer': 1.0
            })

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_FILE, index=False)

    # Print summary
    avg_cer = results_df['cer'].mean()
    avg_wer = results_df['wer'].mean()

    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY - {MODEL_NAME}")
    print(f"{'='*60}")
    print(f"Samples processed: {len(results_df)}")
    print(f"Average CER: {avg_cer:.4f} ({avg_cer*100:.2f}%)")
    print(f"Average WER: {avg_wer:.4f} ({avg_wer*100:.2f}%)")
    print(f"\nResults saved to: {OUTPUT_FILE}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
