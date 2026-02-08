#  Beauty Buddy: Compliance & Product Intelligence Bot

## overview
Beauty Buddy is an AI-powered document intelligence system designed to analyze cosmetic and personal care products.
The application extracts ingredient information from product documents, labels, and web pages, and evaluates safety risks, regulatory compliance, and potential restrictions or bans across different regions

This project demonstrates how AI can support cosmetic compliance, product safety assessment, and decision support using real-world data sources.

## Problem Statement
Cosmetic products often contain complex ingredient lists that are difficult for consumers, startups, and compliance teams to interpret.
Manual verification of safety risks and regulatory restrictions is time-consuming and prone to errors.

## What This System Automates
-Ingredient extraction

-Risk identification

-Compliance analysis

-Regulatory flagging

## Technology Stack
### Frontend
streamlit
### AI/NLP
-LangChain

-Google Gemini (LLM)

-Sentence Transformers (Embeddings)
### Vector Database
FAISS
### Data Processing
-BeautifulSoup (Web Scraping)

-PyPDF Loader

-Tesseract OCR (Image Text Extraction)

 ## System Workflow
 1.User provides input through:
Product PDF,
Product URL,
Ingredient label image

2.The system extracts text from the provided sources.

3.Extracted text is split into smaller chunks and converted into embeddings.

4.FAISS retrieves the most relevant context.

5.The LLM analyzes:
Ingredients,
Safety risks,
Compliance issues,
Regulatory concerns

6.A structured response with safety and compliance insights is generated.

## Example Analysis Question
Analyze the ingredients for safety risks, compliance issues, and country-specific bans.

## Installation
### clone the repository
-git clone <your-repository-url>

-cd beauty-compliance-bot
### create and activate virtual environment
-python -m venv venv

-venv\Scripts\activate
### install dependencies
-pip install -r requirements.txt
### Run the application
-streamlit run app.py

