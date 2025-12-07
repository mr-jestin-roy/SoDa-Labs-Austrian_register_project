"""
OCR Benchmark Script for vLLM and Ollama
Evaluates Qwen3-VL-32B, Chandra OCR, and DeepSeek OCR 3B
Calculates CER and WER metrics
"""

import os
import pandas as pd
from PIL import Image
from pathlib import Path
from typing import Dict, List, Tuple
import json
from jiwer import wer, cer
from tqdm import tqdm
import torch

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Configuration for the benchmark"""
    # Paths
    INPUT_FOLDER = "api_test_assets/test_input_images_folder/Althofen_TrauungsbuchTomIV_1907_1936_00046"  # Folder with segmented images
    GROUND_TRUTH_CSV = "Althofen_TrauungsbuchTomIV_1907_1936_00046_ground_truth_gemini.csv"  # CSV with columns: image_filename, gemini_text
    OUTPUT_FOLDER = "Fine Tuning OCR Model/output_results"

    # YOUR CUSTOM PROMPT for historical German documents
    # This will be used for ALL models
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

    # Models to benchmark
    MODELS = {
        "qwen3_vl_32b_gguf": {
            "backend": "vllm",
            "model_name": "unsloth/Qwen3-VL-32B-Instruct-GGUF",
            "prompt_template": CUSTOM_PROMPT
        },
        "chandra_ocr": {
            "backend": "vllm", 
            "model_name": "datalab-to/chandra",
            "prompt_template": CUSTOM_PROMPT  # Same prompt
        },
        "deepseek_ocr_3b": {
            "backend": "vllm",
            "model_name": "deepseek-ai/DeepSeek-OCR",
            "prompt_template": f"<image>\n{CUSTOM_PROMPT}"  # DeepSeek format
        }
    }

    # vLLM parameters
    VLLM_PARAMS = {
        "temperature": 0.0,  # Deterministic for OCR
        "max_tokens": 512,
        "top_p": 1.0
    }

# =============================================================================
# VLLM INFERENCE CLASS
# =============================================================================

class VLLMInference:
    """Handles inference using vLLM for vision-language models"""

    def __init__(self, model_name: str, max_model_len: int = 4096):
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer

        print(f"Loading model: {model_name}")
        self.model_name = model_name

        # Initialize vLLM
        self.llm = LLM(
            model=model_name,
            max_model_len=max_model_len,
            max_num_seqs=16,
            enforce_eager=True,
            trust_remote_code=True
        )

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        print(f"Model {model_name} loaded successfully")

    def generate(self, image_path: str, prompt: str, temperature: float = 0.0, 
                max_tokens: int = 512) -> str:
        """Generate OCR output for a single image"""
        from vllm import SamplingParams

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Prepare sampling parameters
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=1.0
        )

        # For models that need chat template
        if "qwen" in self.model_name.lower() or "chandra" in self.model_name.lower():
            messages = [{
                'role': 'user',
                'content': prompt.replace("<image>", "")
            }]

            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            formatted_prompt = prompt

        # Generate
        outputs = self.llm.generate(
            {
                "prompt": formatted_prompt,
                "multi_modal_data": {"image": image}
            },
            sampling_params=sampling_params
        )

        # Extract text
        generated_text = outputs[0].outputs[0].text if outputs else ""
        return generated_text.strip()

# =============================================================================
# OLLAMA INFERENCE CLASS (Alternative)
# =============================================================================

class OllamaInference:
    """Handles inference using Ollama for vision models"""

    def __init__(self, model_name: str):
        import ollama
        self.model_name = model_name
        self.client = ollama
        print(f"Using Ollama model: {model_name}")

        # Pull model if not available
        try:
            self.client.show(model_name)
        except:
            print(f"Pulling model {model_name}...")
            self.client.pull(model_name)

    def generate(self, image_path: str, prompt: str, **kwargs) -> str:
        """Generate OCR output using Ollama"""
        response = self.client.chat(
            model=self.model_name,
            messages=[{
                'role': 'user',
                'content': prompt.replace("<image>", "Extract the text from this image:"),
                'images': [image_path]
            }]
        )
        return response['message']['content'].strip()

# =============================================================================
# METRICS CALCULATION
# =============================================================================

def calculate_metrics(ground_truth: str, prediction: str) -> Dict[str, float]:
    """
    Calculate CER and WER between ground truth and prediction

    Args:
        ground_truth: Reference text
        prediction: Predicted text from OCR model

    Returns:
        Dictionary with CER and WER scores
    """
    # Handle empty cases
    if not ground_truth and not prediction:
        return {"cer": 0.0, "wer": 0.0}

    if not ground_truth:
        return {"cer": 1.0, "wer": 1.0}

    if not prediction:
        return {"cer": 1.0, "wer": 1.0}

    # Calculate metrics
    cer_score = cer(ground_truth, prediction)
    wer_score = wer(ground_truth, prediction)

    return {
        "cer": cer_score,
        "wer": wer_score
    }

# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

class OCRBenchmark:
    """Main benchmark class"""

    def __init__(self, config: Config):
        self.config = config
        self.models = {}

        # Create output directory
        os.makedirs(config.OUTPUT_FOLDER, exist_ok=True)

    def load_ground_truth(self) -> pd.DataFrame:
        """Load ground truth CSV"""
        df = pd.read_csv(self.config.GROUND_TRUTH_CSV)
        print(f"Loaded {len(df)} ground truth samples")
        return df

    def initialize_models(self):
        """Initialize all models for benchmarking"""
        for model_id, model_config in self.config.MODELS.items():
            print(f"\nInitializing {model_id}...")

            if model_config["backend"] == "vllm":
                self.models[model_id] = VLLMInference(model_config["model_name"])
            elif model_config["backend"] == "ollama":
                self.models[model_id] = OllamaInference(model_config["model_name"])
            else:
                raise ValueError(f"Unknown backend: {model_config['backend']}")

    def run_inference(self, model_id: str, image_path: str) -> str:
        """Run inference on a single image"""
        model_config = self.config.MODELS[model_id]
        model = self.models[model_id]

        prompt = model_config["prompt_template"]

        return model.generate(
            image_path,
            prompt,
            **self.config.VLLM_PARAMS
        )

    def benchmark_single_model(self, model_id: str, ground_truth_df: pd.DataFrame) -> pd.DataFrame:
        """Benchmark a single model against ground truth"""
        print(f"\n{'='*80}")
        print(f"Benchmarking: {model_id}")
        print(f"{'='*80}")

        results = []

        for idx, row in tqdm(ground_truth_df.iterrows(), total=len(ground_truth_df)):
            image_filename = row['image_filename']
            gt_text = str(row['gemini_text'])

            image_path = os.path.join(self.config.INPUT_FOLDER, image_filename)

            # Check if image exists
            if not os.path.exists(image_path):
                print(f"Warning: Image not found: {image_path}")
                continue

            try:
                # Run inference
                prediction = self.run_inference(model_id, image_path)

                # Calculate metrics
                metrics = calculate_metrics(gt_text, prediction)

                results.append({
                    'image_filename': image_filename,
                    'ground_truth': gt_text,
                    'prediction': prediction,
                    'cer': metrics['cer'],
                    'wer': metrics['wer']
                })

            except Exception as e:
                print(f"Error processing {image_filename}: {str(e)}")
                results.append({
                    'image_filename': image_filename,
                    'ground_truth': gt_text,
                    'prediction': "",
                    'cer': 1.0,
                    'wer': 1.0,
                    'error': str(e)
                })

        # Create dataframe
        results_df = pd.DataFrame(results)

        # Calculate average metrics
        avg_cer = results_df['cer'].mean()
        avg_wer = results_df['wer'].mean()

        print(f"\nResults for {model_id}:")
        print(f"  Average CER: {avg_cer:.4f} ({avg_cer*100:.2f}%)")
        print(f"  Average WER: {avg_wer:.4f} ({avg_wer*100:.2f}%)")

        return results_df

    def run_benchmark(self):
        """Run complete benchmark on all models"""
        # Load ground truth
        ground_truth_df = self.load_ground_truth()

        # Initialize models
        self.initialize_models()

        # Results storage
        all_results = {}
        summary_stats = []

        # Benchmark each model
        for model_id in self.config.MODELS.keys():
            results_df = self.benchmark_single_model(model_id, ground_truth_df)
            all_results[model_id] = results_df

            # Save individual results
            output_file = os.path.join(
                self.config.OUTPUT_FOLDER,
                f"{model_id}_results.csv"
            )
            results_df.to_csv(output_file, index=False)
            print(f"Saved results to: {output_file}")

            # Collect summary stats
            summary_stats.append({
                'model': model_id,
                'avg_cer': results_df['cer'].mean(),
                'avg_wer': results_df['wer'].mean(),
                'median_cer': results_df['cer'].median(),
                'median_wer': results_df['wer'].median(),
                'std_cer': results_df['cer'].std(),
                'std_wer': results_df['wer'].std(),
                'num_samples': len(results_df)
            })

        # Create summary report
        summary_df = pd.DataFrame(summary_stats)
        summary_file = os.path.join(self.config.OUTPUT_FOLDER, "benchmark_summary.csv")
        summary_df.to_csv(summary_file, index=False)

        print(f"\n{'='*80}")
        print("BENCHMARK SUMMARY")
        print(f"{'='*80}")
        print(summary_df.to_string(index=False))
        print(f"\nSummary saved to: {summary_file}")

        return all_results, summary_df

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function"""
    # Initialize configuration
    config = Config()

    # Create benchmark instance
    benchmark = OCRBenchmark(config)

    # Run benchmark
    results, summary = benchmark.run_benchmark()

    print("\n" + "="*80)
    print("Benchmark completed successfully!")
    print("="*80)

if __name__ == "__main__":
    main()
