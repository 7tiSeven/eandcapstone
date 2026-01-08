# Enhancement Plan: Tax Audit & Research Copilot

## Document Metadata
- **Notebook Analyzed:** `Problem_statement_1_AI_Assisted_Tax_Policy_&_Research_final.ipynb`
- **Analysis Date:** 2026-01-08
- **Author:** ML Engineer / Solution Architect Review

---

## 1. Notebook Summary (Current Architecture)

### 1.1 Architecture Overview (15 Bullets)

1. **Dependencies:** `openai`, `langchain`, `langchain-openai`, `langchain-community`, `faiss-cpu`, `sentence-transformers`, `pypdf`, `python-docx`, `gradio`, `pandas`, `reportlab`
2. **LLM Provider:** OpenAI-compatible gateway at `https://aibe.mygreatlearning.com/openai/v1` using `gpt-4o-mini` model
3. **Credentials:** Loaded securely via `google.colab.userdata.get("OPEN_AI_API")` - no hardcoding
4. **Embeddings:** HuggingFace `sentence-transformers/all-MiniLM-L6-v2` for vector representations
5. **Vector Store:** FAISS in-memory index for similarity search
6. **Text Splitter:** `RecursiveCharacterTextSplitter` (chunk_size=900, overlap=150) with legal-aware separators
7. **Document Ingestion:** Supports PDF (pypdf), DOCX (python-docx), and plain text files
8. **Demo Documents:** Auto-generates synthetic UAE VAT law/regulation/guidance PDFs for demonstration
9. **Retrieval:** Similarity search with scores, returns top-K chunks with metadata
10. **Prompt Contract:** Strict JSON-only output with mandatory citations `[source | chunk_id]`
11. **Refusal Behavior:** System prompt enforces "INSUFFICIENT EVIDENCE" response when sources inadequate
12. **Output Structure:** JSON with keys: `summary`, `key_provisions`, `obligations`, `exemptions`, `penalties`, `audit_checklist`, `assumptions`, `sources_used`
13. **UI Framework:** Gradio Blocks with 2 tabs (Upload & Index, Ask)
14. **Evidence Display:** DataFrame showing retrieved sources with preview, evidence strength badge
15. **Export:** PDF download via reportlab with markdown-to-plain conversion

### 1.2 Cell-by-Cell Breakdown

| Cell | Purpose | Key Functions/Classes |
|------|---------|----------------------|
| 0-1 | Markdown intro & business problem | — |
| 2 | Dependency installation | pip install |
| 3-4 | Credentials setup | `userdata.get("OPEN_AI_API")` |
| 5 | Core imports | os, io, re, json, typing |
| 6 | Approach overview markdown | — |
| 7 | File reading | `read_file_bytes()` |
| 8-10 | Demo document generation | `make_pdf()`, reportlab |
| 11 | Workflow markdown | — |
| 12 | Indexing pipeline | `build_index()`, FAISS, embeddings |
| 13 | LLM & Prompt setup | `ChatOpenAI`, `ChatPromptTemplate`, PROMPT |
| 14 | Query answering | `_safe_json_parse()`, `retrieve_with_scores()`, `answer_query()` |
| 15 | Memo formatting | `json_to_memo_md()` |
| 16 | Demo file loader | `initial_uploaded_files_store` |
| 17 | Upload handlers | `ui_upload()`, `ui_build_index()` |
| 18 | Interface markdown | — |
| 19 | Gradio UI | `ui_ask()`, `export_memo_pdf()`, `evidence_badge()`, demo |
| 20 | Closing reflection | — |

---

## 2. Coverage Matrix

| # | Required Approach Step | Current State | Evidence (Cell/Function) | Gap | Risk/Notes | Enhancement Proposal |
|---|------------------------|---------------|--------------------------|-----|------------|---------------------|
| **1** | **Input Handling** - Accept structured query with optional context (taxpayer type, sector, prior refs) | **Partial** | Cell 19: `query`, `as_of` textboxes only | No structured state object; no taxpayer type/sector fields; no constraints input | Users cannot specify focus areas or context | Add `QueryState` dataclass with tax_area, taxpayer_type, sector, constraints fields; add UI inputs |
| **2** | **Context Understanding** - Single agent interprets tax area, request type, detail level | **Missing** | None | No classification step before retrieval | Retrieval is unguided; may return irrelevant chunks | Add pre-retrieval LLM classification call returning `ContextAnalysis` JSON |
| **3** | **Information Retrieval** - Tavily search for UAE tax laws, ministerial decisions, official guidance | **Missing** | None - only local FAISS | No web search capability; cannot access live regulations | System limited to uploaded docs; cannot get latest updates | Add `tavily-python`, implement `web_search()` with domain filtering |
| **4** | **Law Interpretation & Relevance Filtering** - Filter to match scenario; highlight exemptions/thresholds/penalties; ignore outdated | **Partial** | Cell 14: similarity score only | No semantic filtering; no date-based recency; no explicit relevance check | May include tangentially related chunks | Add LLM-based relevance filter post-retrieval; add date metadata |
| **5** | **Structured Summary Generation** - Output: Tax Area, Relevant Laws, Key Provisions, Audit considerations, Plain summary | **Implemented** | Cell 13: PROMPT, Cell 15: `json_to_memo_md()` | Missing explicit "Tax Area" field; "Audit/Compliance considerations" is present as checklist | Minor gap only | Add `tax_area` to JSON schema; rename for clarity |
| **6** | **Iterative Refinement** - User clarifications refine output (focus VAT only, include CT penalties) | **Missing** | None | No state carryover; no refinement UI | Users must re-enter full query each time | Add "Refine last answer" toggle; carry previous `QueryState`; add focus dropdowns |
| **7** | **Tool & Environment Integration** - Env vars (OPENAI keys, TAVILY_API_KEY); LangGraph workflow | **Partial** | Cell 4: OpenAI key only | No TAVILY_API_KEY; no LangGraph; linear function calls | No orchestration framework; harder to extend | Add TAVILY_API_KEY secret; implement LangGraph StateGraph |
| **8** | **Output & Usage** - Display summary, export PDF/Markdown, reference laws, optional batch mode | **Partial** | Cell 19: PDF export, markdown memo | No clickable web URLs; no batch mode; no markdown export button | Limited export options; no multi-query support | Add web source URLs; markdown export; batch query textarea |

### Coverage Summary
- **Implemented:** 1 step (5)
- **Partial:** 4 steps (1, 4, 7, 8)
- **Missing:** 3 steps (2, 3, 6)

---

## 3. Enhancement Roadmap

### Goals
1. Achieve full alignment with the 8-step required approach
2. Enable hybrid retrieval (local docs + live web search)
3. Add intelligent context understanding and relevance filtering
4. Support iterative refinement workflows
5. Maintain audit-grade traceability and citation integrity

### Non-Goals
1. Complete notebook rewrite (preserve working code)
2. Production deployment (remains Colab prototype)
3. Multi-user authentication
4. Real-time collaboration features

---

### Phase 1: Foundation & State Management (Priority: High)

**Duration Estimate:** N/A (no timeline estimates per guidelines)

#### P1.1 - Structured Query State Object

**Why Needed:** Current implementation passes loose parameters; no way to track constraints or carry state between refinements.

**Implementation:**
```python
# Insert after Cell 5 (imports)
from dataclasses import dataclass, field, asdict
from typing import Optional, List
from datetime import date

@dataclass
class QueryState:
    """Structured state object for tax queries."""
    query_text: str
    tax_area: Optional[str] = None  # VAT, CT, Excise, Transfer Pricing, etc.
    request_type: Optional[str] = None  # penalty_inquiry, threshold_check, audit_checklist, general_research
    detail_level: str = "standard"  # brief, standard, comprehensive
    taxpayer_type: Optional[str] = None  # individual, SME, large_corporate, government
    sector: Optional[str] = None  # real_estate, financial_services, manufacturing, etc.
    as_of_date: str = field(default_factory=lambda: date.today().isoformat())
    constraints: List[str] = field(default_factory=list)  # ["VAT only", "include penalties", "exclude historical"]
    prior_refs: List[str] = field(default_factory=list)  # Previous query IDs for context

    def to_prompt_context(self) -> str:
        """Format state for LLM prompt injection."""
        parts = [f"Query: {self.query_text}"]
        if self.tax_area:
            parts.append(f"Tax Area: {self.tax_area}")
        if self.taxpayer_type:
            parts.append(f"Taxpayer Type: {self.taxpayer_type}")
        if self.sector:
            parts.append(f"Sector: {self.sector}")
        if self.constraints:
            parts.append(f"Constraints: {', '.join(self.constraints)}")
        return "\n".join(parts)
```

**Insertion Point:** New Cell 5a (after imports, before file reading)

**UX Change:** No visible change yet; foundation for later UI additions.

**Testing:**
- Unit: Verify `QueryState` serialization/deserialization
- Unit: Verify `to_prompt_context()` formatting

---

#### P1.2 - Context Understanding Step

**Why Needed:** Required approach step 2 mandates interpretation before retrieval to guide search.

**Implementation:**
```python
# Insert after QueryState definition

CONTEXT_ANALYSIS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a tax query classifier. Analyze the user's tax query and return a JSON object with:
- tax_area: one of [VAT, Corporate Tax, Excise Tax, Transfer Pricing, Customs, General, Unknown]
- request_type: one of [penalty_inquiry, threshold_check, compliance_checklist, exemption_check, filing_deadline, general_research]
- detail_level: one of [brief, standard, comprehensive] based on query complexity
- key_entities: list of specific items mentioned (amounts, dates, company types)
- search_keywords: list of 3-5 optimal search terms for retrieval

Return ONLY valid JSON. No markdown."""),
    ("human", "{query}")
])

@dataclass
class ContextAnalysis:
    tax_area: str
    request_type: str
    detail_level: str
    key_entities: List[str]
    search_keywords: List[str]

def analyze_context(query_text: str) -> ContextAnalysis:
    """Pre-retrieval classification step."""
    try:
        msg = CONTEXT_ANALYSIS_PROMPT.format_messages(query=query_text)
        response = llm.invoke(msg).content
        data = json.loads(response)
        return ContextAnalysis(**data)
    except Exception as e:
        # Fallback to defaults if classification fails
        return ContextAnalysis(
            tax_area="Unknown",
            request_type="general_research",
            detail_level="standard",
            key_entities=[],
            search_keywords=[query_text.split()[:5]]
        )
```

**Insertion Point:** New Cell 13a (after LLM setup, before retrieval functions)

**UX Change:** Behind-the-scenes; analysis shown in JSON output.

**Testing:**
- Scenario: "What are VAT penalties for late filing?" → tax_area=VAT, request_type=penalty_inquiry
- Scenario: "Corporate tax rate in UAE" → tax_area=Corporate Tax, request_type=general_research
- Edge: Empty query → graceful fallback to defaults

---

### Phase 2: Hybrid Retrieval (Priority: High)

#### P2.1 - Tavily Web Search Integration

**Why Needed:** Required approach step 3 mandates Tavily for live UAE tax law retrieval.

**Implementation:**
```python
# Insert in Cell 2 (dependencies)
# Add: tavily-python

# Insert after Cell 4 (credentials)
try:
    TAVILY_API_KEY = userdata.get("TAVILY_API_KEY")
    assert TAVILY_API_KEY, "Missing Colab Secret: TAVILY_API_KEY"
    print("✅ Tavily API key loaded")
except:
    TAVILY_API_KEY = None
    print("⚠️ Tavily API key not found - web search disabled")

# New Cell 14a - Web Search Module
from tavily import TavilyClient

# Official UAE tax source domains (whitelist)
TRUSTED_DOMAINS = [
    "tax.gov.ae",           # Federal Tax Authority
    "mof.gov.ae",           # Ministry of Finance
    "economy.ae",           # Ministry of Economy
    "government.ae",        # UAE Government Portal
    "u.ae",                 # Official UAE portal
    "gcc-sg.org",           # GCC Secretariat
    "pwc.com",              # Big 4 (authoritative commentary)
    "ey.com",
    "kpmg.com",
    "deloitte.com",
]

@dataclass
class WebSource:
    """Structured web search result."""
    url: str
    domain: str
    title: str
    snippet: str
    score: float
    published_date: Optional[str] = None

def search_web(query: str, max_results: int = 5, include_domains: List[str] = None) -> List[WebSource]:
    """
    Search web for UAE tax information using Tavily.
    Prioritizes official/trusted sources.
    """
    if not TAVILY_API_KEY:
        return []

    try:
        client = TavilyClient(api_key=TAVILY_API_KEY)

        # Enhance query for UAE tax context
        enhanced_query = f"UAE tax {query}"

        response = client.search(
            query=enhanced_query,
            search_depth="advanced",
            max_results=max_results * 2,  # Fetch more, filter later
            include_domains=include_domains or TRUSTED_DOMAINS,
            include_answer=False,
            include_raw_content=False,
        )

        results = []
        for item in response.get("results", []):
            url = item.get("url", "")
            domain = url.split("/")[2] if "/" in url else url

            # Score boost for official sources
            base_score = item.get("score", 0.5)
            if any(d in domain for d in ["tax.gov.ae", "mof.gov.ae", "government.ae"]):
                base_score *= 1.3  # 30% boost for official

            results.append(WebSource(
                url=url,
                domain=domain,
                title=item.get("title", ""),
                snippet=item.get("content", "")[:800],
                score=min(base_score, 1.0),
                published_date=item.get("published_date")
            ))

        # Sort by score and take top results
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:max_results]

    except Exception as e:
        print(f"⚠️ Web search failed: {e}")
        return []
```

**Insertion Point:**
- Cell 2: Add `tavily-python` to pip install
- New Cell 4a: Tavily credential loading
- New Cell 14a: Web search implementation

**UX Change:** New "Enable Web Search" checkbox in UI.

**Testing:**
- Normal: Search "UAE VAT registration threshold" → returns tax.gov.ae results
- Normal: Search "corporate tax penalty" → returns relevant ministry sources
- Edge: Invalid API key → graceful degradation, returns empty list
- Edge: Network timeout → returns empty list with warning

---

#### P2.2 - Hybrid Retrieval Merger

**Why Needed:** Combine local FAISS results with web results into unified evidence block.

**Implementation:**
```python
# Modify Cell 14 - Extend answer_query()

@dataclass
class UnifiedSource:
    """Unified source with provenance tracking."""
    provenance: str  # "local" or "web"
    source_id: str   # filename or URL
    chunk_id: str
    content: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

def retrieve_hybrid(
    query: str,
    context: ContextAnalysis,
    k_local: int = 4,
    k_web: int = 3,
    enable_web: bool = True
) -> List[UnifiedSource]:
    """
    Hybrid retrieval combining local FAISS and web search.
    Returns unified source list with clear provenance.
    """
    unified = []

    # Local retrieval
    if VECTORSTORE is not None:
        local_results = retrieve_with_scores(query, k=k_local)
        for i, (doc, score) in enumerate(local_results, start=1):
            unified.append(UnifiedSource(
                provenance="local",
                source_id=doc.metadata.get("source", "unknown"),
                chunk_id=f"L{i}",
                content=doc.page_content[:1200],
                score=float(score),
                metadata={"filename": doc.metadata.get("source")}
            ))

    # Web retrieval (if enabled and API key available)
    if enable_web and TAVILY_API_KEY:
        # Use search keywords from context analysis
        search_terms = " ".join(context.search_keywords) if context.search_keywords else query
        web_results = search_web(search_terms, max_results=k_web)

        for i, ws in enumerate(web_results, start=1):
            unified.append(UnifiedSource(
                provenance="web",
                source_id=ws.domain,
                chunk_id=f"W{i}",
                content=ws.snippet,
                score=ws.score,
                metadata={
                    "url": ws.url,
                    "title": ws.title,
                    "published_date": ws.published_date
                }
            ))

    # Sort by score (local and web comparable after normalization)
    unified.sort(key=lambda x: x.score, reverse=True)

    return unified

def format_sources_block(sources: List[UnifiedSource]) -> str:
    """Format unified sources for LLM prompt."""
    blocks = []
    for src in sources:
        if src.provenance == "local":
            header = f"[local | {src.source_id} | {src.chunk_id}]"
        else:
            header = f"[web | {src.source_id} | {src.chunk_id}]"
        blocks.append(f"SOURCE {src.chunk_id} {header} (score={src.score:.3f}):\n{src.content}")
    return "\n\n---\n\n".join(blocks)
```

**Insertion Point:** Modify Cell 14, add helper functions

**UX Change:** Sources table shows provenance column (local/web), web sources have clickable URLs.

**Testing:**
- Normal: Query with both local and web enabled → mixed results
- Normal: Web-only mode (no docs uploaded) → web results only
- Edge: Both disabled → returns error gracefully

---

#### P2.3 - Relevance Filtering

**Why Needed:** Required approach step 4 mandates filtering irrelevant/outdated results.

**Implementation:**
```python
# New Cell 14b - Relevance Filter

RELEVANCE_FILTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a relevance filter for tax research. Given a query and a source snippet, determine if the source is relevant.

Return JSON with:
- relevant: boolean (true if source directly addresses the query)
- confidence: float 0-1
- reason: brief explanation
- outdated: boolean (true if content appears outdated based on dates/references)

Be strict: only mark relevant if the source DIRECTLY addresses the query topic."""),
    ("human", "Query: {query}\n\nSource [{source_id}]:\n{content}")
])

def filter_relevance(
    query: str,
    sources: List[UnifiedSource],
    threshold: float = 0.6
) -> List[UnifiedSource]:
    """
    LLM-based relevance filtering.
    Removes sources below confidence threshold or marked outdated.
    """
    filtered = []

    for src in sources:
        try:
            msg = RELEVANCE_FILTER_PROMPT.format_messages(
                query=query,
                source_id=src.chunk_id,
                content=src.content[:600]
            )
            response = llm.invoke(msg).content
            result = json.loads(response)

            if result.get("relevant", False) and result.get("confidence", 0) >= threshold:
                if not result.get("outdated", False):
                    src.metadata["relevance_score"] = result.get("confidence")
                    src.metadata["relevance_reason"] = result.get("reason", "")
                    filtered.append(src)

        except Exception as e:
            # On filter failure, include source (fail open)
            filtered.append(src)

    return filtered
```

**Insertion Point:** New Cell 14b (after hybrid retrieval)

**UX Change:** Fewer but higher-quality sources; metadata shows relevance reasoning.

**Testing:**
- Normal: Mix of relevant/irrelevant sources → only relevant pass
- Edge: All sources irrelevant → returns empty, triggers INSUFFICIENT EVIDENCE
- Edge: Filter LLM call fails → fails open, includes source

**Potential Failure Modes:**
- **False negatives:** Overly strict filtering removes valid sources
  - Mitigation: Lower threshold, add manual override
- **Latency:** Multiple LLM calls add delay
  - Mitigation: Batch filtering, use faster model for filter

---

### Phase 3: LangGraph Integration (Priority: Medium)

#### P3.1 - LangGraph State Machine

**Why Needed:** Required approach step 7 mandates LangGraph for workflow orchestration.

**Implementation:**
```python
# New Cell 6a - LangGraph Setup

# Add to Cell 2 dependencies:
# langgraph>=0.0.40

from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

class TaxCopilotState(TypedDict):
    """LangGraph state schema."""
    # Inputs
    query_text: str
    as_of_date: str
    enable_web: bool
    constraints: List[str]

    # Intermediate
    context_analysis: Optional[Dict]
    raw_sources: List[Dict]
    filtered_sources: List[Dict]

    # Outputs
    response_json: Dict
    memo_markdown: str
    error: Optional[str]

def node_input(state: TaxCopilotState) -> TaxCopilotState:
    """Input validation and state initialization."""
    if not state.get("query_text"):
        state["error"] = "No query provided"
    return state

def node_context_understanding(state: TaxCopilotState) -> TaxCopilotState:
    """Classify query and extract context."""
    if state.get("error"):
        return state

    analysis = analyze_context(state["query_text"])
    state["context_analysis"] = asdict(analysis)
    return state

def node_retrieval(state: TaxCopilotState) -> TaxCopilotState:
    """Hybrid retrieval: local + web."""
    if state.get("error"):
        return state

    context = ContextAnalysis(**state["context_analysis"])
    sources = retrieve_hybrid(
        state["query_text"],
        context,
        enable_web=state.get("enable_web", True)
    )
    state["raw_sources"] = [asdict(s) for s in sources]
    return state

def node_filtering(state: TaxCopilotState) -> TaxCopilotState:
    """Relevance filtering."""
    if state.get("error"):
        return state

    sources = [UnifiedSource(**s) for s in state["raw_sources"]]
    filtered = filter_relevance(state["query_text"], sources)
    state["filtered_sources"] = [asdict(s) for s in filtered]
    return state

def node_summary(state: TaxCopilotState) -> TaxCopilotState:
    """Generate structured summary."""
    if state.get("error"):
        return state

    sources = [UnifiedSource(**s) for s in state["filtered_sources"]]
    sources_text = format_sources_block(sources)

    # Use existing answer generation logic
    msg = PROMPT.format_messages(
        query=state["query_text"],
        as_of_date=state["as_of_date"],
        sources=sources_text
    )
    response = llm.invoke(msg).content
    parsed, _ = _safe_json_parse(response)

    state["response_json"] = parsed
    state["memo_markdown"] = json_to_memo_md(parsed)
    return state

# Build graph
workflow = StateGraph(TaxCopilotState)

workflow.add_node("input", node_input)
workflow.add_node("context_understanding", node_context_understanding)
workflow.add_node("retrieval", node_retrieval)
workflow.add_node("filtering", node_filtering)
workflow.add_node("summary", node_summary)

workflow.set_entry_point("input")
workflow.add_edge("input", "context_understanding")
workflow.add_edge("context_understanding", "retrieval")
workflow.add_edge("retrieval", "filtering")
workflow.add_edge("filtering", "summary")
workflow.add_edge("summary", END)

tax_copilot_graph = workflow.compile()

"""
WORKFLOW DIAGRAM:
┌─────────┐    ┌─────────────────────┐    ┌───────────┐    ┌───────────┐    ┌─────────┐
│  Input  │───▶│ Context Understanding│───▶│ Retrieval │───▶│ Filtering │───▶│ Summary │
└─────────┘    └─────────────────────┘    └───────────┘    └───────────┘    └─────────┘
                                                │                              │
                                           ┌────┴────┐                         │
                                           │  Local  │                         │
                                           │ (FAISS) │                         ▼
                                           └────┬────┘                    ┌─────────┐
                                                │                         │   END   │
                                           ┌────┴────┐                    └─────────┘
                                           │   Web   │
                                           │(Tavily) │
                                           └─────────┘
"""
```

**Insertion Point:** New Cell 6a (dedicated LangGraph cell with diagram)

**UX Change:** No visible change; internal orchestration improvement.

**Testing:**
- Normal: Full workflow execution → produces valid output
- Edge: Error at any node → error propagates cleanly
- Edge: Empty sources after filtering → INSUFFICIENT EVIDENCE response

---

### Phase 4: UI Enhancements (Priority: Medium)

#### P4.1 - Enhanced Input UI

**Why Needed:** Support structured input from required approach step 1.

**Implementation:** Modify Cell 19 Gradio UI

```python
# Add to Tab "2) Ask" section:

with gr.Row():
    with gr.Column(scale=2):
        query = gr.Textbox(
            label="Audit question / scenario",
            lines=4,
            placeholder="Example: Late filing VAT return — penalties, thresholds, exemptions…"
        )

        with gr.Accordion("Advanced Options", open=False):
            with gr.Row():
                tax_area = gr.Dropdown(
                    choices=["Auto-detect", "VAT", "Corporate Tax", "Excise Tax", "Transfer Pricing", "Customs", "General"],
                    value="Auto-detect",
                    label="Tax Area"
                )
                taxpayer_type = gr.Dropdown(
                    choices=["Not specified", "Individual", "SME", "Large Corporate", "Government Entity"],
                    value="Not specified",
                    label="Taxpayer Type"
                )

            with gr.Row():
                sector = gr.Dropdown(
                    choices=["Not specified", "Real Estate", "Financial Services", "Manufacturing", "Retail", "Technology", "Healthcare"],
                    value="Not specified",
                    label="Sector"
                )
                detail_level = gr.Radio(
                    choices=["Brief", "Standard", "Comprehensive"],
                    value="Standard",
                    label="Detail Level"
                )

            constraints = gr.CheckboxGroup(
                choices=["VAT only", "Corporate Tax only", "Include penalties", "Include exemptions", "Exclude historical"],
                label="Focus Constraints"
            )

        enable_web = gr.Checkbox(
            value=True,
            label="Enable Web Search (Tavily)",
            info="Search official UAE tax sources for latest guidance"
        )
```

**Insertion Point:** Modify Cell 19

**UX Change:** Collapsible advanced options; web search toggle.

---

#### P4.2 - Iterative Refinement

**Why Needed:** Required approach step 6 mandates refinement capability.

**Implementation:**
```python
# Add to Cell 19:

# State to track conversation
conversation_history = gr.State([])
last_query_state = gr.State(None)

refine_mode = gr.Checkbox(
    value=False,
    label="Refine previous answer",
    info="Build on the last query results"
)

refinement_instruction = gr.Textbox(
    label="Refinement instruction",
    placeholder="e.g., Focus on penalties only, Exclude VAT, Add more detail on thresholds",
    visible=False
)

def toggle_refinement(refine):
    return gr.update(visible=refine)

refine_mode.change(toggle_refinement, inputs=refine_mode, outputs=refinement_instruction)

def ui_ask_refined(query, as_of_date, top_k, refine, refinement, last_state, ...):
    if refine and last_state:
        # Append refinement to previous query
        enhanced_query = f"{last_state['query_text']}\n\nREFINEMENT: {refinement}"
        # Carry forward constraints
        ...
```

**Insertion Point:** Modify Cell 19

**UX Change:** Checkbox to enable refinement mode; text input for refinement instructions.

---

#### P4.3 - Enhanced Output with Web Sources

**Why Needed:** Required approach step 8 mandates clickable URLs for web evidence.

**Implementation:**
```python
# Modify json_to_memo_md() in Cell 15:

def json_to_memo_md(result: dict) -> str:
    # ... existing code ...

    # Add Web Sources section
    md.append("## Web Sources")
    web_sources = [s for s in result.get("_retrieved_sources", []) if s.get("provenance") == "web"]
    if web_sources:
        for ws in web_sources:
            url = ws.get("metadata", {}).get("url", "")
            title = ws.get("metadata", {}).get("title", ws.get("source_id", ""))
            md.append(f"- [{title}]({url})")
    else:
        md.append("_No web sources used._")

    return "\n".join(md)

# Add to UI - Markdown export button
def export_memo_md(state_result: dict) -> str:
    """Export memo as Markdown file."""
    if not state_result:
        raise gr.Error("No memo available")

    memo_md = json_to_memo_md(state_result)
    export_dir = "exports"
    os.makedirs(export_dir, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    md_path = os.path.join(export_dir, f"audit_memo_{ts}.md")

    with open(md_path, "w") as f:
        f.write(memo_md)

    return md_path
```

**Insertion Point:** Modify Cells 15 and 19

**UX Change:** Web sources with clickable links; Markdown download button.

---

#### P4.4 - Batch Query Mode

**Why Needed:** Required approach step 8 mentions optional batch analysis.

**Implementation:**
```python
# Add new Tab to Cell 19:

with gr.Tab("3) Batch Analysis"):
    gr.Markdown("""
    ### Batch Query Processing
    Enter multiple queries (one per line) or upload a CSV with a 'query' column.
    """)

    batch_input = gr.Textbox(
        label="Queries (one per line)",
        lines=6,
        placeholder="Query 1: What are VAT penalties?\nQuery 2: Corporate tax rate in UAE?\nQuery 3: Excise tax on tobacco?"
    )

    batch_csv = gr.File(label="Or upload CSV", file_types=[".csv"])

    batch_btn = gr.Button("Process Batch", variant="primary")
    batch_output = gr.Dataframe(label="Results", wrap=True)
    batch_download = gr.File(label="Download Results CSV")

    def process_batch(queries_text, csv_file):
        queries = []
        if queries_text:
            queries = [q.strip() for q in queries_text.strip().split("\n") if q.strip()]
        elif csv_file:
            df = pd.read_csv(csv_file)
            queries = df["query"].tolist()

        results = []
        for q in queries:
            result = answer_query(q)
            results.append({
                "query": q,
                "tax_area": result.get("context_analysis", {}).get("tax_area", ""),
                "summary": result.get("summary", "")[:200],
                "sources_count": len(result.get("_retrieved_sources", []))
            })

        return pd.DataFrame(results)

    batch_btn.click(process_batch, inputs=[batch_input, batch_csv], outputs=batch_output)
```

**Insertion Point:** Add new Tab in Cell 19

**UX Change:** New "Batch Analysis" tab for multi-query processing.

---

## 4. Implementation Details Summary

### Dependencies to Add (Cell 2)
```bash
!pip -q install --upgrade \
  tavily-python \
  langgraph>=0.0.40
```

### New Colab Secrets Required
- `TAVILY_API_KEY` - Tavily API key for web search

### New Cells to Add
| New Cell | After Cell | Content |
|----------|------------|---------|
| 4a | 4 | Tavily credential loading |
| 5a | 5 | QueryState dataclass |
| 6a | 6 | LangGraph workflow definition |
| 13a | 13 | Context analysis prompt and function |
| 14a | 14 | Web search implementation |
| 14b | 14a | Relevance filtering |

### Cells to Modify
| Cell | Modification |
|------|--------------|
| 2 | Add tavily-python, langgraph |
| 13 | Update PROMPT to include tax_area in output schema |
| 14 | Integrate hybrid retrieval into answer_query() |
| 15 | Add web sources section to memo formatting |
| 19 | Add advanced options, refinement, batch mode |

---

## 5. Test Plan

### Normal Scenarios (5 tests)

| # | Test Name | Input | Expected Output | Validation |
|---|-----------|-------|-----------------|------------|
| T1 | VAT Penalty Lookup | "What are the penalties for late VAT filing in UAE?" | JSON with penalties array, citations to VAT_Law_Demo.pdf | penalties array non-empty; citations format correct |
| T2 | Web Search Integration | "Latest UAE corporate tax updates 2024" (web enabled) | Results include tax.gov.ae sources | _retrieved_sources contains provenance="web" |
| T3 | Hybrid Retrieval | "VAT registration threshold" (local docs + web) | Mixed local and web sources | Both L* and W* chunk_ids present |
| T4 | Context Classification | "What exemptions apply to SME taxpayers?" | context_analysis.tax_area = "VAT" or "General" | tax_area field populated |
| T5 | Batch Processing | 3 queries via batch tab | DataFrame with 3 rows, summaries for each | All rows have non-empty summary |

### Edge Cases (3 tests)

| # | Test Name | Input | Expected Output | Validation |
|---|-----------|-------|-----------------|------------|
| E1 | No Sources Available | "Cryptocurrency tax regulations in Mars" | INSUFFICIENT EVIDENCE response | response contains "INSUFFICIENT" |
| E2 | Tavily API Failure | Web search with invalid key | Graceful fallback to local only | No crash; local results returned |
| E3 | Malformed Query | Empty string / special characters only | Error message, no crash | error key in response |

### Test Execution Commands
```python
# Add test cell at end of notebook:

def run_tests():
    tests_passed = 0
    tests_failed = 0

    # T1: VAT Penalty Lookup
    result = answer_query("What are the penalties for late VAT filing?")
    if result.get("penalties") and len(result["penalties"]) > 0:
        print("✅ T1 Passed: VAT Penalty Lookup")
        tests_passed += 1
    else:
        print("❌ T1 Failed: VAT Penalty Lookup")
        tests_failed += 1

    # ... additional tests ...

    print(f"\n{'='*40}")
    print(f"Tests: {tests_passed} passed, {tests_failed} failed")

run_tests()
```

---

## 6. Demo Script Update (2 minutes)

### Updated Demo Flow

**[0:00 - 0:20] Introduction**
- Show notebook title and business problem
- Highlight: "Now with hybrid retrieval and intelligent filtering"

**[0:20 - 0:40] Document Setup**
- Show demo docs auto-generation
- Click "Build Index" → show success message
- Point out: "Local knowledge base ready"

**[0:40 - 1:10] Basic Query with Web Search**
- Enter: "What are the current VAT registration thresholds and penalties for late filing?"
- Toggle ON "Enable Web Search"
- Click "Generate Audit Summary"
- Show:
  - Context analysis in JSON (tax_area: VAT)
  - Mixed sources (local L1-L4, web W1-W2)
  - Structured memo with citations
  - Evidence strength badge

**[1:10 - 1:30] Advanced Options Demo**
- Expand "Advanced Options"
- Select: Tax Area = VAT, Taxpayer Type = SME
- Check: "Include penalties"
- Re-run query
- Show: More focused results

**[1:30 - 1:50] Refinement Demo**
- Check "Refine previous answer"
- Enter: "Focus only on penalty amounts, exclude thresholds"
- Re-run → show refined output

**[1:50 - 2:00] Export**
- Click "Download Audit Memo (PDF)"
- Show downloaded file
- Highlight clickable web source links in memo

---

## 7. PR Checklist for GitHub Publishing

### Pre-Publish Checklist

- [ ] **Secrets Removed**
  - [ ] No API keys in code cells
  - [ ] All secrets loaded via `userdata.get()` or `os.environ`
  - [ ] Add `.env.example` file listing required secrets

- [ ] **Dependencies Documented**
  - [ ] All pip packages in Cell 2 with version pins
  - [ ] requirements.txt file created

- [ ] **Code Cleanup**
  - [ ] Remove debug print statements
  - [ ] Clear all cell outputs before commit
  - [ ] Remove personal identifiers if any

- [ ] **Documentation**
  - [ ] README.md with setup instructions
  - [ ] Colab Secrets setup guide
  - [ ] Demo video link (optional)

- [ ] **Testing**
  - [ ] All 8 tests pass
  - [ ] Tested with fresh Colab runtime
  - [ ] Tested without Tavily key (graceful degradation)

- [ ] **License**
  - [ ] LICENSE file added
  - [ ] Third-party license compliance checked

### Recommended Git Commands
```bash
# Initialize if needed
git init

# Add files
git add Problem_statement_1_AI_Assisted_Tax_Policy_&_Research_final.ipynb
git add ENHANCEMENT_PLAN.md
git add requirements.txt
git add README.md
git add .env.example

# Commit
git commit -m "feat: add hybrid retrieval and LangGraph orchestration

- Add Tavily web search integration with trusted domain filtering
- Implement LangGraph state machine for workflow orchestration
- Add context understanding pre-retrieval classification
- Add iterative refinement and batch processing UI
- Enhance output with clickable web source links

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"

# Push
git push origin main
```

---

## 8. Optional: Patch Snippets

### Snippet 1: Tavily Integration (Cell 4a)
```python
# === TAVILY WEB SEARCH SETUP ===
try:
    TAVILY_API_KEY = userdata.get("TAVILY_API_KEY")
    if TAVILY_API_KEY:
        print("✅ Tavily API key loaded - web search enabled")
    else:
        print("⚠️ TAVILY_API_KEY not set - web search disabled")
except Exception:
    TAVILY_API_KEY = None
    print("⚠️ Tavily API key not found - web search disabled")
```

### Snippet 2: LangGraph Skeleton (Cell 6a)
```python
# === LANGGRAPH WORKFLOW ===
from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional, List, Dict

class TaxCopilotState(TypedDict):
    query_text: str
    as_of_date: str
    enable_web: bool
    context_analysis: Optional[Dict]
    sources: List[Dict]
    response: Dict
    error: Optional[str]

def build_workflow():
    graph = StateGraph(TaxCopilotState)

    graph.add_node("input", lambda s: s)
    graph.add_node("context", node_context_understanding)
    graph.add_node("retrieve", node_retrieval)
    graph.add_node("filter", node_filtering)
    graph.add_node("summarize", node_summary)

    graph.set_entry_point("input")
    graph.add_edge("input", "context")
    graph.add_edge("context", "retrieve")
    graph.add_edge("retrieve", "filter")
    graph.add_edge("filter", "summarize")
    graph.add_edge("summarize", END)

    return graph.compile()

tax_workflow = build_workflow()
```

### Snippet 3: Enhanced UI Dropdown (Cell 19 modification)
```python
# Add after query textbox:
with gr.Accordion("🔧 Advanced Query Options", open=False):
    tax_area_dd = gr.Dropdown(
        ["Auto", "VAT", "Corporate Tax", "Excise", "Transfer Pricing"],
        value="Auto", label="Tax Area"
    )
    taxpayer_dd = gr.Dropdown(
        ["Any", "Individual", "SME", "Corporate", "Government"],
        value="Any", label="Taxpayer Type"
    )
    web_toggle = gr.Checkbox(True, label="Enable Web Search (Tavily)")
```

---

## Appendix: File Structure After Enhancement

```
Capstone/
├── Problem_statement_1_AI_Assisted_Tax_Policy_&_Research_final.ipynb
├── ENHANCEMENT_PLAN.md
├── README.md
├── requirements.txt
├── .env.example
├── tax_demo_docs/
│   ├── VAT_Law_Demo.pdf
│   ├── VAT_Regulation_Demo.pdf
│   ├── VAT_Guidance_Demo.pdf
│   └── tax_demo_docs.zip
└── exports/
    └── (generated PDFs and markdown files)
```

---

*End of Enhancement Plan*
