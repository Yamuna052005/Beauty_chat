# Beauty Buddy: AI-Powered Cosmetic Compliance & Risk Intelligence System

## Overview
Beauty Buddy is an AI-driven document intelligence platform designed to automate cosmetic ingredient analysis, safety assessment, and regulatory compliance validation.

The system processes unstructured product data such as PDFs, images, and web pages, extracts ingredient-level information, and performs multi-dimensional analysis including toxicological risk evaluation and region-specific regulatory checks.

---

## Problem Statement
Cosmetic formulations often contain dense, non-standardized ingredient lists that are difficult to interpret.

Challenges:
- Regulatory requirements differ across regions (EU, US, India, etc.)
- Ingredient safety data is fragmented
- Manual compliance checks are slow and error-prone
- Startups lack access to regulatory intelligence tools

---

## Solution
Beauty Buddy automates the end-to-end compliance workflow:

- Extracts ingredient data from multiple input formats
- Identifies hazardous or controversial compounds
- Maps ingredients to regulatory frameworks
- Flags banned or restricted substances
- Generates structured compliance reports

---

## Core Features

### Multi-Source Data Ingestion
- Product PDFs
- Web pages
- Ingredient label images

### Intelligent Ingredient Extraction
- Handles unstructured text
- Normalizes ingredient names

### Risk & Safety Analysis
- Detects allergens, irritants, and toxic compounds
- Provides contextual explanations

### Regulatory Compliance Engine
- Identifies banned/restricted ingredients
- Supports region-specific validation

### Retrieval-Augmented Generation (RAG)
- Uses vector similarity search
- Enhances LLM response accuracy

---

## System Architecture

1. Input Layer
   - PDF / URL / Image

2. Extraction Layer
   - PyPDF Loader
   - BeautifulSoup
   - Tesseract OCR

3. Processing Layer
   - Text chunking
   - Embedding generation

4. Retrieval Layer
   - FAISS vector database

5. AI Analysis Layer
   - Google Gemini (via LangChain)
   - Safety and compliance reasoning

6. Output Layer
   - Structured compliance report

---

## Technology Stack

### Frontend
- Streamlit

### AI / NLP
- LangChain
- Google Gemini
- Sentence Transformers

### Data & Retrieval
- FAISS

### Data Extraction
- BeautifulSoup
- PyPDF Loader
- Tesseract OCR

---

## Example Use Case
Input: Upload a cosmetic product label (image or PDF)

Output:
- Ingredient list
- Risk classification (Low / Moderate / High)
- Compliance flags
- Region-specific insights
- Explanation for flagged ingredients

---

## Installation & Setup

### Clone Repository
git clone <your-repository-url>
cd beauty-compliance-bot

### Create Virtual Environment
python -m venv venv
venv\Scripts\activate

### Install Dependencies
pip install -r requirements.txt

### Run Application
streamlit run app.py

---

## Limitations
- Limited regulatory datasets
- LLM outputs may need validation
- OCR accuracy depends on image quality

---

## Future Enhancements
- Integration with official regulatory databases
- Real-time compliance API
- Ingredient substitution suggestions
- Multilingual support
- Mobile application

---

## Project Value
- Demonstrates RAG architecture
- Applies AI to real-world compliance problems
- Handles unstructured data pipelines
- Uses LLMs for decision support
