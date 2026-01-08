# Tax Audit & Research Copilot

By: Abdulla Ahmed Alaydaroos

This repository contains my capstone project - a tax audit support tool designed to assist tax auditors in researching UAE tax laws, regulations, and official guidance.

## Repository Structure

```
├── v1_original/                    # Version 1 - Initial Implementation
│   └── Problem_statement_1_*.ipynb # Original prototype notebook
│
├── v2_enhanced/                    # Version 2 - Final Enhanced Version
│   ├── tax_audit_copilot_enhanced.ipynb    # Full enhanced notebook
│   ├── COLAB_CELLS_COPY_PASTE.md           # Copy-paste ready code cells
│   ├── ENHANCEMENT_PLAN.md                  # Development documentation
│   ├── requirements.txt                     # Python dependencies
│   └── .env.example                         # Environment variables template
│
└── README.md                       # This file
```

## Version Comparison

### Version 1 (Original)
- Basic RAG pipeline with FAISS vector store
- Simple Gradio interface
- Local document retrieval only

### Version 2 (Enhanced - Final)
- Structured state management with dataclasses
- Context understanding with LLM classification
- Hybrid retrieval (local FAISS + Tavily web search)
- Trusted UAE tax source prioritization (FTA, MOF, official portals)
- LLM-based relevance filtering
- LangGraph workflow orchestration
- Enhanced Gradio UI with multiple tabs
- Iterative refinement support
- Batch query processing
- PDF and Markdown export

## How to Run

### Option 1: Google Colab (Recommended)
1. Open `v2_enhanced/tax_audit_copilot_enhanced.ipynb` in Google Colab
2. Add API keys using Colab Secrets:
   - `OPEN_AI_API` (required)
   - `TAVILY_API_KEY` (optional, enables web search)
3. Run all cells from top to bottom
4. Upload your tax documents or use provided samples

### Option 2: Local Environment
1. Clone this repository
2. Copy `v2_enhanced/.env.example` to `.env` and fill in your keys
3. Install dependencies: `pip install -r v2_enhanced/requirements.txt`
4. Run the notebook with Jupyter

## Features

- **Document Processing**: Upload PDFs and Word documents for indexing
- **Smart Retrieval**: Combines local knowledge base with live web search
- **UAE Tax Focus**: Prioritizes official FTA and MOF sources
- **Structured Output**: Generates audit-ready memorandums with citations
- **Export Options**: Download results as PDF or Markdown

## Notes
- Demo documents are synthetic and used for illustration only
- The tool supports professional judgment and improves efficiency
- Web search requires Tavily API key (graceful fallback if not available)
