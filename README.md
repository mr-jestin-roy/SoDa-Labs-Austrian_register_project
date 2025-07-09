# Optical character recognition on historical handwritten texts

Repository for the Austrian Register Data Project 📖

---

**Given**: Dataset of historical handwritten text from Austrian birth, baptism, marriage, and death registers. Archives are present in tabular form (table being hand-drawn or block-printed) and text is cursive handwriting. [Metricula online dataset](https://data.matricula-online.eu/en/bestande/).

**Overview**

- Text dates 1625-2006. More info [here](https://data.matricula-online.eu/en/allgemeine-infos/).
- 183,312 church registers; 4,502,868 images scraped and stored on AWS S3.

Sample ground truth:
![sample ground truth](./assets/images/04-Trauung_0005.jpg)

> **Notes on dataset:**

- Tables are not uniform in structure (variation in fields/columns), owing to the range of periods and geographic locations the dataset comes from
- Handwriting styles vary due to differences in human-writers and periods
- Noise is present in the form of ink blots, parchment texture, or general wear and tear of the physical documents

> Matricula images were scraped using [austrian_register_project/Scraper at main · sodalabsio/austrian_register_project (github.com)](https://github.com/sodalabsio/austrian_register_project/tree/main/Scraper); originally <https://github.com/1fge/matricula-online-scraper>

> Sample Matricula dataset present at: https://drive.google.com/drive/folders/15eqsj8LIIz9bv_VLS8pzlua7WvbymnSn?usp=sharing

**\*Full dataset**: on AWS S3 bucket “austrian-register-data”. Ask access from Satya Borgohain. (Some part of it) transferred to Paul’s M3; 496 images at /projects/oc23/mini_images/ and transcriptions at /projects/oc23/austrian_images.txt.\*

---

## 🚀 Quick Start Guide

### Project Structure

```
SoDa-Labs-Austrian_register_project/
├── two_model_api_pipeline.py           # Main OCR comparison pipeline
├── webhook_server.py                   # Webhook server for faster processing
├── transkribus_client.py               # Transkribus API client
├── requirements.txt                    # Python dependencies
├── config.example                      # Environment variables template
├── .env                               # Your API keys (create from config.example)
├── api_test_assets/                   # Test images and results
│   ├── test_input_images_folder/      # Sample Austrian parish records
│   └── test_output_images_folder/     # Processed results
├── input_images_sample/               # Quick test images
└── README.md                          # This file
```

### Key Files

- **`two_model_api_pipeline.py`**: Main script that compares HandwritingOCR, Transkribus, and OpenAI o3 APIs
- **`webhook_server.py`**: Optional webhook server for faster HandwritingOCR processing
- **`transkribus_client.py`**: Dedicated client for Transkribus API interactions
- **`config.example`**: Template for environment variables - copy to `.env` and fill in your API keys

### Prerequisites

- Python 3.8+
- pip package manager
- Git

### 1. Clone and Setup

```bash
# Clone the repository
git clone <repository-url>
cd SoDa-Labs-Austrian_register_project

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. API Configuration

Copy the example config file and add your API credentials:

```bash
cp config.example .env
```

Edit the `.env` file with your actual API keys:

```plaintext
# Transkribus API Configuration
TRANSKRIBUS_USERNAME=your_email@example.com
TRANSKRIBUS_PASSWORD=your_password
TRANSKRIBUS_HTR_MODEL_ID=38230

# HandwritingOCR.com API Configuration
HANDWRITING_OCR_API_TOKEN=your_handwriting_ocr_api_token_here

# OpenAI API Configuration (for o3 model)
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. API Key Setup Instructions

#### **Transkribus API**
1. Create account at [Transkribus](https://transkribus.eu/)
2. Use your email and password in the `.env` file
3. Model ID 38230 is pre-configured for German handwriting

#### **HandwritingOCR.com API**
1. Sign up at [HandwritingOCR.com](https://www.handwritingocr.com/)
2. Go to API settings and generate an API token
3. Add the token to your `.env` file

#### **OpenAI API (o3 Model)**
1. Create account at [OpenAI Platform](https://platform.openai.com/)
2. Go to [API Keys](https://platform.openai.com/account/api-keys) and create a new key
3. **Important**: Your organization must be verified to use o3 models
4. Go to [Organization Settings](https://platform.openai.com/settings/organization/general) and click "Verify Organization"
5. Wait up to 15 minutes for access to propagate

### 4. Running the OCR Pipeline

The main pipeline compares three OCR services: HandwritingOCR, Transkribus, and OpenAI o3.

```bash
# Activate virtual environment
source venv/bin/activate

# Run the comparison pipeline
python two_model_api_pipeline.py
```

#### **Single API Mode**

To test individual APIs, uncomment/comment the relevant lines in `two_model_api_pipeline.py`:

```python
# Test only HandwritingOCR
hwocr_text = call_handwritingocr_api(img_path, use_webhook=False)
# transkribus_text = call_transkribus_api(img_path)
# openO3_text = call_openO3_api(img_path)

# Test only Transkribus
# hwocr_text = call_handwritingocr_api(img_path, use_webhook=False)
transkribus_text = call_transkribus_api(img_path)
# openO3_text = call_openO3_api(img_path)

# Test only OpenAI o3
# hwocr_text = call_handwritingocr_api(img_path, use_webhook=False)
# transkribus_text = call_transkribus_api(img_path)
openO3_text = call_openO3_api(img_path)
```

#### **Batch Processing Options**

```python
# Process first 10 images (for testing)
for img_name in images[:10]:

# Process all images in folder
for img_name in images:

# Process specific range
for img_name in images[20:50]:
```

### 5. Configuration Options

Edit the settings in `two_model_api_pipeline.py`:

```python
# Settings
image_folder = "api_test_assets/test_input_images_folder/Althofen_TrauungsbuchTomIV_1907_1936_00015"
output_csv = "ocr_results.csv"                # Output CSV file
word_cer_threshold = 0.3                      # CER threshold for word matching
line_cer_threshold = 0.10                     # CER threshold for whole line
```

#### **Available Test Data**

The repository includes sample Austrian parish record images:

- `api_test_assets/test_input_images_folder/Althofen_TrauungsbuchTomIV_1907_1936_00015/` - 149 line images
- `api_test_assets/test_input_images_folder/Althofen_TrauungsbuchTomIV_1907_1936_00046/` - 162 line images
- `input_images_sample/` - Sample images for quick testing

#### **Webhook Configuration**

For faster HandwritingOCR processing, the pipeline supports webhook notifications:

```python
# Enable webhook mode (faster)
hwocr_text = call_handwritingocr_api(img_path, use_webhook=True)

# Use polling mode (slower but more reliable)
hwocr_text = call_handwritingocr_api(img_path, use_webhook=False)
```

### 6. Pipeline Features

- **Multi-API comparison**: Compare HandwritingOCR, Transkribus, and OpenAI o3
- **Batch processing**: Process entire folders of images
- **Webhook support**: Faster processing with webhook notifications
- **German handwriting optimization**: Specialized prompts for Austrian parish records
- **Error handling**: Robust fallback mechanisms
- **Progress tracking**: Real-time processing updates

### 7. Expected Output

The pipeline generates a CSV file with columns:
- `image`: Image filename
- `hwocr_text`: HandwritingOCR transcription
- `transkribus_text`: Transkribus transcription  
- `openO3_text`: OpenAI o3 transcription
- `cer`: Character Error Rate comparison
- `final_ground_truth`: Aligned ground truth text

### 8. Performance

- **HandwritingOCR**: ~2 seconds per image
- **Transkribus**: ~30 seconds per image  
- **OpenAI o3**: ~13 seconds per image
- **Batch processing**: Supports hundreds of images

### 9. Optional: Webhook Setup

For faster HandwritingOCR processing, set up webhook notifications:

```bash
# Install ngrok (for local development)
# Run webhook server
python webhook_server.py

# In another terminal, expose webhook
ngrok http 5400
```

See `WEBHOOK_SETUP.md` for detailed instructions.

### 10. Troubleshooting

#### **Common Issues**

**OpenAI o3 Model Access Error**
```
Error: Your organization must be verified to use the model `o3-2025-04-16`
```
**Solution**: Go to [Organization Settings](https://platform.openai.com/settings/organization/general) and verify your organization. Wait up to 15 minutes for access.

**Environment Variables Not Found**
```
Exception: OPENAI_API_KEY environment variable must be set
```
**Solution**: Ensure your `.env` file exists and contains all required API keys.

**Transkribus Authentication Failed**
```
Authentication failed: 401 - Unauthorized
```
**Solution**: Check your Transkribus username and password in the `.env` file.

**HandwritingOCR Webhook Timeout**
```
Webhook timeout, falling back to polling...
```
**Solution**: This is normal behavior. The system automatically falls back to polling mode.

**No Images Found**
```
Empty image folder or no supported formats (.jpg, .png, .jpeg)
```
**Solution**: Check that your image folder path is correct and contains supported image formats.

#### **Performance Tips**

- Use webhook mode for HandwritingOCR when processing many images
- Process images in smaller batches (10-50) to avoid API rate limits
- For large datasets, run overnight processing sessions
- Monitor your API usage to avoid hitting quota limits

#### **Getting Help**

If you encounter issues:

1. Check the console output for detailed error messages
2. Verify all API keys are correctly set in the `.env` file
3. Test with a small batch of images first
4. Check API service status pages for outages

---

## Aim
To perform layout analysis and handwritten text recognition (HTR) on the images, extract tabular data from the parish records, and identify key figures from the regions, tracking their descendants' economic progress over the decades that followed, including key periods such as World War I and many more.

---

### Sub-aim 1 
To ***Using LLM toools transcribe some real Austrian birth certificates*** from the archives to obtain rich ground truth (limited in size due to human annotation), hence reducing the CER rate further.

Path of images: [./assets/images/mini_images/](./assets/images/mini_images/)  
Path of corresponding transcriptions: [./assets/docs/austrian_images.txt](./assets/docs/austrian_images.txt)  
No. of word-level images transcribed: 496

**Work done by past researchers like Yash & Ali are as follows**,
For this, we used [Transkribus](https://readcoop.eu/transkribus/) desktop app. Scripts in place:
- In Yashdeep’s local (written by Ali): transkribus_API.py, word polygon extraction.ipynb. These are for generating a CSV containing the prediction for the word, ground-truth, and character error rate (CER). 
- **Model**: PyLaia HTR 27457. We used this model hosted on transkribus servers to get some rudimentary transcriptions; later we refined these transcriptions by hand (done by Paul for ~500 word-images).
    - Net Name: **German_Kurrent_17th-18th (10th Nov, 2020)**
    - Language: **german**, latin, french
    - No. of words trained on: 1839841
    - CER Train: 6.00%; CER Val: 5.50%

---

### Sub-aim 2
To ***train an HTR model*** using both the above labelled opensource dataset (HANA) and the hand-transcribed data by Paul with the help of Transkribus above.

Current model being explored: [TrOCR](https://paperswithcode.com/method/trocr)  
Path in M3: `/oc23/trocr/`

---

### Sub-aim 3
To ***infer (test) the above HTR model*** on our Austrian birth dataset (transcribed using Transkribus by Paul and Sascha)

---

### **Sub-aim 4**

To **extract and classify key entities**—specifically names, professions, and place names—from the HTR model outputs. This involves applying **named entity recognition (NER)** and **text classification techniques** on each structured row of the digitized records. The objective is to systematically segment out individual names, recorded professions (occupations), and geographical locations (villages, towns, parishes, etc.) present in the historical documents. The extracted entities will be further categorized and aggregated to provide insights into:

- The **distribution and frequency** of given names and surnames
- **Common and rare professions** across different periods and regions
- **Geographic clustering** of families and occupations

This sub-aim supports downstream analysis such as **demographic mapping, social mobility studies,** and **regional economic profiling**, enabling a richer understanding of population structure and occupational trends in historical Austrian society.

---

### **Sub-aim 5**

To **construct historical social networks** by mapping relationships and associations extracted from each record in the birth, marriage, and death registers. Using the classified entities from Sub-aim 5, this stage involves identifying and linking individuals based on documented **family ties** (parents, children, spouses), **godparent relationships** (as found in baptismal records), and other recorded associations. The goal is to create a comprehensive **connection graph** for each individual, revealing patterns of social connectivity, kinship, and community structure over time.

This networked approach will enable:

- **Tracing lineage and familial relationships** across generations
- **Analyzing the roles of godparents, witnesses,** and other community members in social cohesion
- **Studying the evolution** of interconnected families, migration patterns, and the spread of professions within regions

Ultimately, Sub-aim 6 aims to provide a **dynamic, searchable model of historical social networks**, offering researchers new ways to explore the interconnectedness of individuals and families in Austrian society from **1625 to 2006**.

---

### **Sub-aim 6**

To **conduct demographic and longitudinal analyses** leveraging the classified names, professions, and locations, as well as the social networks constructed in previous sub-aims. This phase focuses on examining how individuals and families were **clustered by profession, geographic origin, and social connectivity**. By linking historical register data to broader historical contexts—such as key events or periods of social change (e.g., radical movements)—the analysis can trace lineages back to specific towns or regions, revealing **ancestral connections and patterns of association**.

The aim is to:

- **Identify demographic clusters** and the spatial distribution of professions and families
- **Trace economic and occupational trajectories** of individuals and families across generations
- **Analyze how factors** such as geographic origin, profession, and social ties influenced upward mobility and livelihood improvements over time

Through this sub-aim, the project aspires to **uncover stories of economic progress, shifts in social status,** and the **impact of historical events on community development**, providing valuable insights for historians, genealogists, and social scientists.

 
