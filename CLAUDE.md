# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

AI-Powered Audit Readiness Assistant - A Google Colab notebook-based system that helps regulatory authorities (like ADAA) assess whether organizations are ready for formal audits. Uses LangChain/LangGraph for LLM orchestration, FAISS for document retrieval, and Tavily for web search.

## Repository Structure

- `v1_original/` - Initial prototype (archived)
- `v2_enhanced/` - Current version with full feature set

## Running the Notebook

The project runs in Google Colab. No local development setup required.

**In Colab:**
1. Open `v2_enhanced/audit_readiness_assistant.ipynb`
2. Add secrets via Colab Secrets sidebar:
   - `OPEN_AI_API` (required) - OpenAI-compatible API key
   - `TAVILY_API_KEY` (optional) - enables web search
3. Run all cells sequentially
4. Use the interactive interface tabs

**API Gateway:** Uses `https://aibe.mygreatlearning.com/openai/v1` as the OpenAI-compatible base URL.

## Architecture

### Workflow Pipeline (LangGraph StateGraph)

```
Entity Intake → Standards Retrieval → Gap Analysis → Risk Scoring → Report Generation
```

1. **Entity Intake**: Collects organization profile, compliance indicators, prior findings
2. **Standards Retrieval**: Loads relevant IFRS/ADAA/control requirements from knowledge base
3. **Gap Analysis**: LLM compares inputs against requirements to identify gaps
4. **Risk Scoring**: Prioritizes gaps as Critical, High, Medium, Low
5. **Report Generation**: Produces structured audit readiness report with recommendations

### Key Data Structures

- `EntityProfile`: Organization info (name, type, sector, size, framework)
- `ComplianceIndicator`: Self-assessment status for each compliance area
- `PriorFinding`: Historical audit observations with severity and status
- `IdentifiedGap`: Gap with risk level, requirement reference, recommendation
- `UnifiedSource`: Document source with provenance tracking

### Compliance Areas

- Financial Reporting
- Internal Controls
- Asset Management
- Procurement & Contracts
- HR & Payroll
- IT Systems & Security
- Regulatory Compliance
- Governance & Oversight

### UI (Gradio)

Five tabs:
1. Document Setup - Upload standards documents or use demos
2. Organization Profile - Entity details input
3. Compliance Self-Assessment - Checklist by area
4. Prior Audit Findings - Historical observations
5. Run Assessment - Execute analysis and view results

## Key Files in v2_enhanced/

- `audit_readiness_assistant.ipynb` - Main notebook (18 cells)
- `COLAB_CELLS_COPY_PASTE.md` - Copy-paste ready cells for manual setup
- `requirements.txt` - Python dependencies
- `.env.example` - Environment template

## LLM Configuration

Uses `gpt-4o-mini` via the OpenAI-compatible gateway:
- Main LLM: temperature 0.1 for analysis/generation
- Fast LLM: temperature 0.0 for classification

Responses are structured JSON with fallback parsing for malformed outputs.
