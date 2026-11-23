#!/bin/bash
# =============================================================================
# Nectar OCR Environment Setup Script
# =============================================================================
# Usage: bash setup_env.sh
# =============================================================================

set -e  # Exit on error

echo "🚀 Setting up Nectar OCR Environment..."
echo "========================================"

# -----------------------------------------------------------------------------
# OPTION 1: Conda Environment (Recommended for GPU)
# -----------------------------------------------------------------------------
setup_conda() {
    echo "📦 Creating conda environment 'nectar-ocr'..."

    # Create environment from YAML
    conda env create -f environment.yml

    echo ""
    echo "✅ Conda environment created!"
    echo ""
    echo "To activate:"
    echo "    conda activate nectar-ocr"
}

# -----------------------------------------------------------------------------
# OPTION 2: Minimal pip-only install (for HTML to XLSX only)
# -----------------------------------------------------------------------------
setup_pip_minimal() {
    echo "📦 Installing minimal dependencies (HTML → XLSX only)..."

    pip install beautifulsoup4 openpyxl lxml pandas pillow

    echo ""
    echo "✅ Minimal install complete!"
    echo "   You can now run: python convert_html_to_xlsx.py"
}

# -----------------------------------------------------------------------------
# OPTION 3: Full pip install (requires NVIDIA GPU for vLLM)
# -----------------------------------------------------------------------------
setup_pip_full() {
    echo "📦 Installing full dependencies..."

    pip install -r requirements.txt

    echo ""
    echo "✅ Full install complete!"
}

# -----------------------------------------------------------------------------
# Main Menu
# -----------------------------------------------------------------------------
echo ""
echo "Select installation option:"
echo "  1) Conda environment (full GPU support) - RECOMMENDED"
echo "  2) Pip minimal (HTML→XLSX conversion only)"
echo "  3) Pip full (requires NVIDIA GPU)"
echo ""
read -p "Enter choice [1-3]: " choice

case $choice in
    1) setup_conda ;;
    2) setup_pip_minimal ;;
    3) setup_pip_full ;;
    *) echo "Invalid choice. Exiting."; exit 1 ;;
esac

echo ""
echo "========================================"
echo "🎉 Setup complete!"
echo ""
echo "Quick Start Commands:"
echo "  - Convert HTML to XLSX:  python chandra_html_output_8758/convert_html_to_xlsx.py"
echo "  - Run batch OCR:         python batch_chandra_ocr.py --input-dir ./images --output-dir ./output"
echo "  - Start vLLM server:     vllm serve datalab-to/chandra --port 8010 --trust-remote-code"
echo "========================================"
