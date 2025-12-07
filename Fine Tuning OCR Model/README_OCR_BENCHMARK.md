# OCR Benchmark Script - Setup & Usage Guide

## Overview
This script benchmarks three OCR models on your custom dataset:
1. **Qwen3-VL-32B-Instruct-GGUF** (32B parameter vision-language model, GGUF format)
2. **Chandra OCR** (specialized OCR model from Datalab)
3. **DeepSeek OCR 3B** (efficient 3B MoE model)

The script evaluates models using **Character Error Rate (CER)** and **Word Error Rate (WER)**.

---

## Installation Requirements

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended: 24GB+ VRAM for Qwen2-VL-32B)
- At least 32GB system RAM

### Option 1: Using vLLM (Recommended for Best Performance)

```bash
# Install vLLM and dependencies
pip install vllm>=0.6.1
pip install transformers>=4.37.0
pip install accelerate
pip install jiwer  # For CER/WER calculation
pip install pandas pillow tqdm

# For Qwen2-VL support
pip install qwen-vl-utils

# Optional: Flash Attention for faster inference
pip install flash-attn --no-build-isolation
```

### Option 2: Using Ollama (Easier Setup, Slightly Slower)

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Install Python client
pip install ollama
pip install jiwer pandas pillow tqdm

# Pull models
ollama pull llama3.2-vision  # For testing Ollama setup
```

---

## Dataset Preparation

Your dataset should be organized as follows:

```
project/
├── input_images/              # Folder with segmented images
│   ├── image001.jpg
│   ├── image002.png
│   └── ...
├── ground_truth.csv           # CSV with ground truth
├── ocr_benchmark_script.py    # The benchmark script
└── output_results/            # Will be created automatically
```

### Ground Truth CSV Format
Your CSV should have exactly these two columns:

| image_filename | gemini_text |
|----------------|-------------|
| image001.jpg   | actual text content |
| image002.png   | actual text content |

Example:
```csv
image_filename,gemini_text
word_001.jpg,"Hello world"
line_002.png,"This is a test document"
```

---

## Configuration

Edit the `Config` class in `ocr_benchmark_script.py`:

```python
class Config:
    # Update these paths to match your setup
    INPUT_FOLDER = "path/to/your/input_images"
    GROUND_TRUTH_CSV = "path/to/your/ground_truth.csv"
    OUTPUT_FOLDER = "path/to/your/output_results"
```

---

## Model-Specific Setup

### 1. Qwen3-VL-32B-GGUF (vLLM)
```python
# The script will automatically download from Hugging Face
# Requires: ~32GB disk space (GGUF format is more efficient), 16GB+ VRAM
# Model: unsloth/Qwen3-VL-32B-Instruct-GGUF
```

**First-time setup:**
```bash
# Log in to Hugging Face (if model requires authentication)
huggingface-cli login
```

### 2. Chandra OCR (vLLM)
```python
# Model: datalab-to/chandra
# Requires: ~40GB disk space, 16GB+ VRAM
```

**Note:** Chandra uses a custom license. For commercial use >$2M revenue, check their pricing page.

### 3. DeepSeek OCR 3B (vLLM)
```python
# Model: deepseek-ai/DeepSeek-OCR
# Requires: ~12GB disk space, 8GB+ VRAM
# Most memory-efficient option
```

---

## Running the Benchmark

### Basic Usage

```bash
python ocr_benchmark_script.py
```

### GPU Memory Optimization

If you encounter OOM (Out of Memory) errors:

**For vLLM:**
```python
# Reduce max_model_len in VLLMInference.__init__
self.llm = LLM(
    model=model_name,
    max_model_len=2048,  # Reduce from 4096
    tensor_parallel_size=2,  # Use 2 GPUs if available
)
```

**For single model testing:**
Comment out models in Config.MODELS to test one at a time:
```python
MODELS = {
    # "qwen2_vl_32b": {...},  # Comment out to skip
    "deepseek_ocr_3b": {...},  # Start with smallest model
}
```

---

## Understanding the Output

### Individual Model Results
For each model, the script creates a CSV file:

**Example: `qwen3_vl_32b_gguf_results.csv`**
```csv
image_filename,ground_truth,prediction,cer,wer
image001.jpg,"hello world","hello world",0.0,0.0
image002.jpg,"test data","test date",0.1,0.5
```

### Summary Report
**`benchmark_summary.csv`**
```csv
model,avg_cer,avg_wer,median_cer,median_wer,std_cer,std_wer,num_samples
qwen3_vl_32b_gguf,0.0234,0.0456,0.0200,0.0400,0.0123,0.0234,100
chandra_ocr,0.0189,0.0378,0.0150,0.0350,0.0098,0.0189,100
deepseek_ocr_3b,0.0267,0.0523,0.0230,0.0480,0.0145,0.0267,100
```

### Interpreting Metrics

**Character Error Rate (CER):**
- 0.00 - 0.02 (0-2%): Excellent
- 0.02 - 0.05 (2-5%): Good
- 0.05 - 0.10 (5-10%): Acceptable for some use cases
- >0.10 (>10%): Poor, needs improvement

**Word Error Rate (WER):**
- Typically 2-4x higher than CER
- 0.00 - 0.05 (0-5%): Excellent
- 0.05 - 0.15 (5-15%): Good
- >0.15 (>15%): May need post-processing

---

## Troubleshooting

### Issue: CUDA Out of Memory

**Solution 1:** Test with smaller model first
```bash
# Edit script to only include DeepSeek OCR 3B
# Then gradually add larger models
```

**Solution 2:** Process fewer images at a time
```python
# In Config class, limit dataset size for testing
ground_truth_df = ground_truth_df.head(10)  # Test with 10 images first
```

**Solution 3:** Use Ollama instead of vLLM
```python
# Change backend in Config.MODELS
"backend": "ollama",  # Instead of "vllm"
```

### Issue: Model Download Fails

**Solution:**
```bash
# Pre-download models
huggingface-cli download unsloth/Qwen3-VL-32B-Instruct-GGUF
huggingface-cli download datalab-to/chandra
huggingface-cli download deepseek-ai/DeepSeek-OCR
```

### Issue: jiwer Import Error

**Solution:**
```bash
pip install jiwer --upgrade
# Or use alternative:
pip install fastwer  # Faster alternative
```

### Issue: Image Loading Errors

**Solution:**
```python
# Check image format support
from PIL import Image
Image.open("test.jpg").convert("RGB")  # Ensure RGB format
```

---

## Advanced Configuration

### Custom Prompts
Modify prompts in `Config.MODELS` for better results:

```python
"prompt_template": """<image>
You are an expert OCR system. Extract ALL text from this image.
Rules:
1. Preserve exact spacing and line breaks
2. Include punctuation marks
3. Do not add explanations
4. Output only the text content
"""
```

### Adjusting Inference Parameters
```python
VLLM_PARAMS = {
    "temperature": 0.0,  # 0.0 = deterministic (recommended for OCR)
    "max_tokens": 1024,  # Increase for longer texts
    "top_p": 1.0,
}
```

### Batch Processing
For large datasets:
```python
# Process in chunks
chunk_size = 50
for i in range(0, len(ground_truth_df), chunk_size):
    chunk = ground_truth_df.iloc[i:i+chunk_size]
    # Process chunk
```

---

## Performance Benchmarks

Expected inference times (per image, on RTX 4090):

| Model | VRAM Usage | Time/Image | CER (typical) |
|-------|------------|------------|---------------|
| DeepSeek OCR 3B | 8-10GB | ~0.5s | 2-5% |
| Chandra OCR | 16-20GB | ~1.0s | 1.5-4% |
| Qwen3-VL-32B-GGUF | 16-24GB | ~1.5s | 1-3% |

---

## Alternative: Ollama Setup

If vLLM is too complex, use Ollama:

```python
# Modify Config.MODELS to use Ollama
"qwen2_vl_7b": {  # Use smaller 7B version
    "backend": "ollama",
    "model_name": "qwen2-vl:7b",
    "prompt_template": "Extract text from this image:"
}
```

```bash
# Pull Ollama models
ollama pull qwen2-vl:7b
ollama pull llama3.2-vision
```

---

## Citation & License

### Models
- **Qwen3-VL-GGUF**: Apache 2.0 License
- **Chandra OCR**: Apache 2.0 + Custom License (see model card)
- **DeepSeek OCR**: MIT License

### Libraries
- **jiwer**: Apache 2.0
- **vLLM**: Apache 2.0

---

## Support

For issues:
1. Check GPU memory: `nvidia-smi`
2. Verify Python packages: `pip list | grep vllm`
3. Test with single small image first
4. Check model-specific documentation on Hugging Face

Common resources:
- vLLM Docs: https://docs.vllm.ai
- Ollama Docs: https://ollama.com/docs
- jiwer: https://github.com/jitsi/jiwer
