# Tax Audit Copilot - Capstone Project

**By: Abdulla Ahmed Alaydaroos**

This document contains all the code cells for my capstone project. Copy each cell into Google Colab in order.

---

## CELL 1 - Install Required Libraries

```python
!pip -q install --upgrade \
  openai==1.66.3 \
  langchain>=1.0.0 \
  langchain-core>=1.0.0 \
  langchain-openai>=0.3.0 \
  langchain-community>=0.3.0 \
  langgraph>=0.2.0 \
  tavily-python \
  faiss-cpu \
  sentence-transformers \
  pypdf \
  python-docx \
  gradio \
  pandas \
  reportlab
```

---

## CELL 2 - Load API Keys

```python
from google.colab import userdata

# OpenAI API
API_KEY = userdata.get("OPEN_AI_API")
assert API_KEY, "Missing Colab Secret: OPEN_AI_API"
BASE_URL = "https://aibe.mygreatlearning.com/openai/v1"
print("✅ OpenAI key loaded and gateway set:", BASE_URL)

# Tavily API (optional - works without it too)
try:
    TAVILY_API_KEY = userdata.get("TAVILY_API_KEY")
    if TAVILY_API_KEY:
        print("✅ Tavily API key loaded - web search enabled")
    else:
        TAVILY_API_KEY = None
        print("⚠️ TAVILY_API_KEY not set - web search disabled (local docs only)")
except Exception:
    TAVILY_API_KEY = None
    print("⚠️ Tavily API key not found - web search disabled (local docs only)")
```

---

## CELL 3 - Import Libraries

```python
import os, io, re, json, traceback
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from enum import Enum

from pypdf import PdfReader
import docx
```

---

## CELL 4 - Define Data Structures

```python
class TaxArea(str, Enum):
    VAT = "VAT"
    CORPORATE_TAX = "Corporate Tax"
    EXCISE_TAX = "Excise Tax"
    TRANSFER_PRICING = "Transfer Pricing"
    CUSTOMS = "Customs"
    GENERAL = "General"
    UNKNOWN = "Unknown"

class RequestType(str, Enum):
    PENALTY_INQUIRY = "penalty_inquiry"
    THRESHOLD_CHECK = "threshold_check"
    COMPLIANCE_CHECKLIST = "compliance_checklist"
    EXEMPTION_CHECK = "exemption_check"
    FILING_DEADLINE = "filing_deadline"
    GENERAL_RESEARCH = "general_research"

@dataclass
class QueryState:
    """Holds the user query and related options."""
    query_text: str
    tax_area: Optional[str] = None
    request_type: Optional[str] = None
    detail_level: str = "standard"
    taxpayer_type: Optional[str] = None
    sector: Optional[str] = None
    as_of_date: str = field(default_factory=lambda: date.today().isoformat())
    constraints: List[str] = field(default_factory=list)
    enable_web_search: bool = True
    previous_query_id: Optional[str] = None

    def to_prompt_context(self) -> str:
        parts = [f"Query: {self.query_text}"]
        if self.tax_area and self.tax_area != "Auto-detect":
            parts.append(f"Tax Area: {self.tax_area}")
        if self.taxpayer_type and self.taxpayer_type != "Not specified":
            parts.append(f"Taxpayer Type: {self.taxpayer_type}")
        if self.sector and self.sector != "Not specified":
            parts.append(f"Sector: {self.sector}")
        if self.constraints:
            parts.append(f"Focus Constraints: {', '.join(self.constraints)}")
        if self.detail_level != "standard":
            parts.append(f"Detail Level: {self.detail_level}")
        return "\n".join(parts)

@dataclass
class ContextAnalysis:
    """Stores the classification results."""
    tax_area: str
    request_type: str
    detail_level: str
    key_entities: List[str]
    search_keywords: List[str]
    confidence: float = 0.0

@dataclass
class UnifiedSource:
    """Represents a source from either local docs or web."""
    provenance: str  # "local" or "web"
    source_id: str
    chunk_id: str
    content: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_citation(self) -> str:
        return f"[{self.provenance} | {self.source_id} | {self.chunk_id}]"

print("✅ State objects defined: QueryState, ContextAnalysis, UnifiedSource")
```

---

## CELL 5 - File Reading Functions

```python
def read_file_bytes(filename: str, file_bytes: bytes) -> str:
    name = filename.lower()

    if name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(file_bytes))
        pages = []
        for i, page in enumerate(reader.pages):
            txt = page.extract_text() or ""
            pages.append(f"[PAGE {i+1}] {txt}")
        text = "\n".join(pages)

    elif name.endswith(".docx"):
        d = docx.Document(io.BytesIO(file_bytes))
        text = "\n".join(p.text for p in d.paragraphs)

    else:
        text = file_bytes.decode("utf-8", errors="ignore")

    text = re.sub(r"\s+", " ", text).strip()
    return text
```

---

## CELL 6 - Create Sample Documents

```python
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from pathlib import Path
import textwrap, zipfile

out_dir = Path("tax_demo_docs")
out_dir.mkdir(exist_ok=True)

def make_pdf(path, title, sections):
    c = canvas.Canvas(str(path), pagesize=letter)
    w, h = letter
    x, y = 0.75*inch, h - 0.9*inch

    c.setFont("Helvetica-Bold", 16)
    c.drawString(x, y, title)
    y -= 0.4*inch

    c.setFont("Helvetica", 9)
    c.drawString(x, y, f"SYNTHETIC DEMO DOCUMENT - {date.today()} (NOT REAL LAW)")
    y -= 0.3*inch

    for header, body in sections:
        if y < 1.2*inch:
            c.showPage()
            y = h - 0.9*inch

        c.setFont("Helvetica-Bold", 12)
        c.drawString(x, y, header)
        y -= 0.25*inch

        c.setFont("Helvetica", 11)
        for line in textwrap.wrap(body, 95):
            if y < 1.0*inch:
                c.showPage()
                y = h - 0.9*inch
            c.drawString(x, y, line)
            y -= 0.18*inch
        y -= 0.15*inch

    c.save()

# Sample VAT Law
make_pdf(
    out_dir / "VAT_Law_Demo.pdf",
    "Synthetic VAT Law (Demo)",
    [
        ("Article 12 - Registration Threshold",
         "Mandatory VAT registration applies if taxable supplies exceed AED 375,000 "
         "in the preceding 12 months. Voluntary registration applies from AED 187,500."),
        ("Article 22 - Filing Deadline",
         "VAT returns must be submitted no later than the 28th day following the end "
         "of the tax period."),
        ("Article 59 - Late Filing Penalty",
         "AED 1,000 for the first late return. AED 2,000 for repeated late returns "
         "within 24 months."),
        ("Article 60 - Late Payment Penalty",
         "2% immediately after the due date, 4% after 7 days, plus 1% daily thereafter.")
    ]
)

# Sample Regulation
make_pdf(
    out_dir / "VAT_Regulation_Demo.pdf",
    "Synthetic VAT Executive Regulation (Demo)",
    [
        ("Regulation 7 - Small Business Supplies",
         "Persons below the mandatory registration threshold are not required to "
         "charge VAT unless voluntarily registered."),
        ("Regulation 52 - Penalty Mitigation",
         "Penalties may be reduced if a justified excuse is accepted by the authority.")
    ]
)

# Sample Guidance
make_pdf(
    out_dir / "VAT_Guidance_Demo.pdf",
    "Synthetic Tax Authority Guidance (Demo)",
    [
        ("GN-07 - Late Filing Review",
         "Auditors should verify submission timestamps and assigned tax periods."),
        ("GN-07 - Audit Checklist",
         "Check filing history, payment confirmations, turnover evidence, and "
         "mitigation requests.")
    ]
)

# Create ZIP
zip_path = out_dir / "tax_demo_docs.zip"
with zipfile.ZipFile(zip_path, "w") as z:
    for f in out_dir.glob("*.pdf"):
        z.write(f, f.name)

print("✅ Demo files created:")
for f in out_dir.iterdir():
    print(" -", f)
```

---

## CELL 7 - Set Up Vector Store

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=900,
    chunk_overlap=150,
    separators=["\n\n", "\n", "Article ", "Section ", ". ", " "]
)

VECTORSTORE = None

def build_index(files: List[Dict[str, Any]]) -> str:
    global VECTORSTORE

    docs = []
    for f in files:
        text = read_file_bytes(f["name"], f["bytes"])
        if len(text) < 50:
            print(f"⚠️ Low text extracted from {f['name']} (len={len(text)})")
        docs.append(Document(page_content=text, metadata={"source": f["name"]}))

    chunks = splitter.split_documents(docs)
    VECTORSTORE = FAISS.from_documents(chunks, embeddings)

    return f"✅ Indexed {len(files)} file(s) into {len(chunks)} chunks."

print("✅ Vector store ready")
```

---

## CELL 8 - Initialize LLM

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.1,
    api_key=API_KEY,
    base_url=BASE_URL
)

# Faster model for quick tasks
llm_fast = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.0,
    api_key=API_KEY,
    base_url=BASE_URL
)

print("✅ LLM clients initialized")
```

---

## CELL 9 - Query Classification

```python
CONTEXT_ANALYSIS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a tax query classifier for UAE tax regulations. Analyze the user's tax query and return a JSON object.

Return ONLY valid JSON with these exact keys:
- tax_area: one of ["VAT", "Corporate Tax", "Excise Tax", "Transfer Pricing", "Customs", "General", "Unknown"]
- request_type: one of ["penalty_inquiry", "threshold_check", "compliance_checklist", "exemption_check", "filing_deadline", "general_research"]
- detail_level: one of ["brief", "standard", "comprehensive"] based on query complexity
- key_entities: array of specific items mentioned (amounts, dates, company types, etc.)
- search_keywords: array of 3-5 optimal search terms for retrieval (include "UAE" context)
- confidence: float 0.0-1.0 indicating classification confidence

Return ONLY the JSON object. No markdown, no explanation."""),
    ("human", "{query}")
])

def analyze_context(query_text: str, user_tax_area: str = None) -> ContextAnalysis:
    """Classify the query before retrieval."""
    try:
        msg = CONTEXT_ANALYSIS_PROMPT.format_messages(query=query_text)
        response = llm_fast.invoke(msg).content

        data = json.loads(response.strip())

        if user_tax_area and user_tax_area not in ["Auto-detect", "Auto", None, ""]:
            data["tax_area"] = user_tax_area

        return ContextAnalysis(
            tax_area=data.get("tax_area", "Unknown"),
            request_type=data.get("request_type", "general_research"),
            detail_level=data.get("detail_level", "standard"),
            key_entities=data.get("key_entities", []),
            search_keywords=data.get("search_keywords", [query_text]),
            confidence=data.get("confidence", 0.5)
        )
    except Exception as e:
        print(f"⚠️ Context analysis fallback: {e}")
        return ContextAnalysis(
            tax_area=user_tax_area if user_tax_area and user_tax_area != "Auto-detect" else "Unknown",
            request_type="general_research",
            detail_level="standard",
            key_entities=[],
            search_keywords=[query_text],
            confidence=0.0
        )

print("✅ Context understanding ready")
```

---

## CELL 10 - Web Search with Tavily

```python
# UAE official tax source domains
TRUSTED_DOMAINS = [
    "tax.gov.ae",           # Federal Tax Authority
    "mof.gov.ae",           # Ministry of Finance
    "economy.ae",           # Ministry of Economy
    "government.ae",        # UAE Government Portal
    "u.ae",                 # Official UAE portal
    "gcc-sg.org",           # GCC Secretariat
]

EXTENDED_TRUSTED = TRUSTED_DOMAINS + [
    "pwc.com",
    "ey.com",
    "kpmg.com",
    "deloitte.com",
]

@dataclass
class WebSearchResult:
    url: str
    domain: str
    title: str
    snippet: str
    score: float
    published_date: Optional[str] = None
    is_official: bool = False

def search_web_tavily(query: str, max_results: int = 5, context: ContextAnalysis = None) -> List[WebSearchResult]:
    """Search the web for UAE tax info using Tavily."""
    if not TAVILY_API_KEY:
        return []

    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=TAVILY_API_KEY)

        enhanced_query = f"UAE {query}"
        if context and context.tax_area not in ["Unknown", "General"]:
            enhanced_query = f"UAE {context.tax_area} {query}"

        response = client.search(
            query=enhanced_query,
            search_depth="advanced",
            max_results=max_results * 2,
            include_answer=False,
            include_raw_content=False,
        )

        results = []
        for item in response.get("results", []):
            url = item.get("url", "")
            domain = ""
            try:
                from urllib.parse import urlparse
                domain = urlparse(url).netloc.replace("www.", "")
            except:
                domain = url.split("/")[2] if len(url.split("/")) > 2 else url

            is_official = any(d in domain for d in TRUSTED_DOMAINS)
            is_trusted = any(d in domain for d in EXTENDED_TRUSTED)

            base_score = item.get("score", 0.5)
            if is_official:
                adjusted_score = min(base_score * 1.4, 1.0)
            elif is_trusted:
                adjusted_score = min(base_score * 1.2, 1.0)
            else:
                adjusted_score = base_score * 0.8

            results.append(WebSearchResult(
                url=url,
                domain=domain,
                title=item.get("title", ""),
                snippet=item.get("content", "")[:800],
                score=adjusted_score,
                published_date=item.get("published_date"),
                is_official=is_official
            ))

        results.sort(key=lambda x: (x.is_official, x.score), reverse=True)
        return results[:max_results]

    except Exception as e:
        print(f"⚠️ Web search error: {e}")
        return []

print(f"✅ Web search configured. Tavily enabled: {TAVILY_API_KEY is not None}")
```

---

## CELL 11 - Hybrid Retrieval

```python
def retrieve_local(query: str, k: int = 6) -> List[UnifiedSource]:
    """Get results from local FAISS index."""
    global VECTORSTORE
    if VECTORSTORE is None:
        return []

    results = VECTORSTORE.similarity_search_with_score(query, k=k)
    sources = []
    for i, (doc, score) in enumerate(results, start=1):
        sources.append(UnifiedSource(
            provenance="local",
            source_id=doc.metadata.get("source", "unknown"),
            chunk_id=f"L{i}",
            content=doc.page_content[:1200],
            score=float(score),
            metadata={"filename": doc.metadata.get("source")}
        ))
    return sources

def retrieve_web(query: str, k: int = 3, context: ContextAnalysis = None) -> List[UnifiedSource]:
    """Get results from web search."""
    web_results = search_web_tavily(query, max_results=k, context=context)
    sources = []
    for i, ws in enumerate(web_results, start=1):
        sources.append(UnifiedSource(
            provenance="web",
            source_id=ws.domain,
            chunk_id=f"W{i}",
            content=ws.snippet,
            score=ws.score,
            metadata={
                "url": ws.url,
                "title": ws.title,
                "published_date": ws.published_date,
                "is_official": ws.is_official
            }
        ))
    return sources

def retrieve_hybrid(
    query: str,
    context: ContextAnalysis,
    k_local: int = 4,
    k_web: int = 3,
    enable_web: bool = True
) -> List[UnifiedSource]:
    """Combine local and web search results."""
    all_sources = []

    local_sources = retrieve_local(query, k=k_local)
    all_sources.extend(local_sources)

    if enable_web and TAVILY_API_KEY:
        search_query = query
        if context and context.search_keywords:
            search_query = " ".join(context.search_keywords[:3])

        web_sources = retrieve_web(search_query, k=k_web, context=context)
        all_sources.extend(web_sources)

    return all_sources

def format_sources_for_prompt(sources: List[UnifiedSource]) -> str:
    """Format sources for the LLM prompt."""
    blocks = []
    for src in sources:
        header = src.to_citation()
        extra = ""
        if src.provenance == "web":
            url = src.metadata.get("url", "")
            extra = f" | URL: {url}" if url else ""
        blocks.append(f"SOURCE {src.chunk_id} {header}{extra} (score={src.score:.3f}):\n{src.content}")
    return "\n\n---\n\n".join(blocks)

print("✅ Hybrid retrieval ready")
```

---

## CELL 12 - Relevance Filter

```python
RELEVANCE_FILTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a relevance filter for UAE tax research. Given a query and a source snippet, determine if the source is relevant.

Return ONLY valid JSON with these keys:
- relevant: boolean (true ONLY if source DIRECTLY addresses the query topic)
- confidence: float 0.0-1.0
- reason: brief explanation (max 20 words)
- outdated: boolean (true if content appears outdated based on dates/references to old laws)

Be STRICT: only mark relevant if the source directly addresses the query. Generic tax info is NOT relevant.
Return ONLY the JSON object."""),
    ("human", "Query: {query}\n\nSource [{source_id}]:\n{content}")
])

def filter_relevance(
    query: str,
    sources: List[UnifiedSource],
    threshold: float = 0.5,
    max_to_filter: int = 8
) -> List[UnifiedSource]:
    """Filter out irrelevant sources using LLM."""
    if not sources:
        return []

    sources_to_filter = sources[:max_to_filter]
    filtered = []

    for src in sources_to_filter:
        try:
            msg = RELEVANCE_FILTER_PROMPT.format_messages(
                query=query,
                source_id=src.chunk_id,
                content=src.content[:500]
            )
            response = llm_fast.invoke(msg).content
            result = json.loads(response.strip())

            is_relevant = result.get("relevant", False)
            confidence = result.get("confidence", 0)
            is_outdated = result.get("outdated", False)

            if is_relevant and confidence >= threshold and not is_outdated:
                src.metadata["relevance_confidence"] = confidence
                src.metadata["relevance_reason"] = result.get("reason", "")
                filtered.append(src)

        except Exception as e:
            # If filter fails, keep the source
            filtered.append(src)

    return filtered

print("✅ Relevance filtering ready")
```

---

## CELL 13 - Answer Generation

```python
MAIN_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an AI-assisted tax research and audit support assistant for UAE tax regulations.

Rules (STRICT):
- Use ONLY the provided SOURCES. Do not use outside knowledge.
- Every factual statement MUST have a citation like [local | filename | chunk_id] or [web | domain | chunk_id].
- If sources are insufficient, say "INSUFFICIENT EVIDENCE" and list what is missing.
- Prefer official sources (tax.gov.ae, mof.gov.ae) over commentary sources.
- If web sources conflict with local documents, note the discrepancy.

Return ONLY valid JSON with these keys:
- tax_area: string (the primary tax area addressed)
- summary: string (plain-language summary of findings)
- relevant_laws: array of objects {{ "law": string, "citation": string }}
- key_provisions: array of objects {{ "point": string, "citation": string }}
- obligations: array of objects {{ "item": string, "citation": string }}
- exemptions: array of objects {{ "item": string, "citation": string }}
- penalties: array of objects {{ "item": string, "citation": string }}
- audit_checklist: array of strings
- assumptions: array of strings
- sources_used: array of strings (list all source citations used)
- web_references: array of objects {{ "title": string, "url": string }} (for web sources only)"""),

    ("human", """AUDIT DATE (as-of): {as_of_date}

QUERY CONTEXT:
{query_context}

SOURCES:
{sources}

Return ONLY JSON. No markdown. No commentary.""")
])

def _safe_json_parse(text: str) -> Tuple[Dict[str, Any], str]:
    raw = (text or "").strip()

    try:
        return json.loads(raw), raw
    except Exception:
        pass

    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = raw[start:end+1]
        try:
            return json.loads(candidate), raw
        except Exception:
            pass

    return {"error": "Model returned non-JSON or invalid JSON.", "raw_output": raw[:6000]}, raw

def generate_answer(
    query_state: QueryState,
    context: ContextAnalysis,
    sources: List[UnifiedSource]
) -> Dict[str, Any]:
    """Generate the final answer from sources."""

    if not sources:
        return {
            "error": "INSUFFICIENT EVIDENCE",
            "summary": "No relevant sources found. Please upload relevant documents or try a different query.",
            "tax_area": context.tax_area,
            "sources_used": []
        }

    sources_text = format_sources_for_prompt(sources)
    query_context = query_state.to_prompt_context()

    try:
        msg = MAIN_PROMPT.format_messages(
            query_context=query_context,
            as_of_date=query_state.as_of_date,
            sources=sources_text
        )
        response = llm.invoke(msg).content
        parsed, raw = _safe_json_parse(response)

        parsed["_context_analysis"] = asdict(context)
        parsed["_retrieved_sources"] = [
            {
                "rank": i,
                "provenance": s.provenance,
                "source": s.source_id,
                "chunk_id": s.chunk_id,
                "score": s.score,
                "preview": s.content[:350].replace("\n", " "),
                "url": s.metadata.get("url", ""),
                "is_official": s.metadata.get("is_official", False)
            }
            for i, s in enumerate(sources, start=1)
        ]

        return parsed

    except Exception as e:
        return {"error": f"LLM call failed: {type(e).__name__}: {e}"}

print("✅ Answer generation ready")
```

---

## CELL 14 - LangGraph Workflow

```python
try:
    from langgraph.graph import StateGraph, END
    from typing import TypedDict
    LANGGRAPH_AVAILABLE = True
    print("✅ LangGraph imported")
except ImportError:
    LANGGRAPH_AVAILABLE = False
    print("⚠️ LangGraph not available, using fallback workflow")

if LANGGRAPH_AVAILABLE:
    class WorkflowState(TypedDict):
        query_state: Dict
        enable_filtering: bool
        context_analysis: Optional[Dict]
        raw_sources: List[Dict]
        filtered_sources: List[Dict]
        response: Dict
        error: Optional[str]

    def node_context_understanding(state: WorkflowState) -> WorkflowState:
        if state.get("error"):
            return state
        qs = QueryState(**state["query_state"])
        analysis = analyze_context(qs.query_text, qs.tax_area)
        state["context_analysis"] = asdict(analysis)
        return state

    def node_retrieval(state: WorkflowState) -> WorkflowState:
        if state.get("error"):
            return state
        qs = QueryState(**state["query_state"])
        context = ContextAnalysis(**state["context_analysis"])
        sources = retrieve_hybrid(qs.query_text, context, enable_web=qs.enable_web_search)
        state["raw_sources"] = [asdict(s) for s in sources]
        return state

    def node_filtering(state: WorkflowState) -> WorkflowState:
        if state.get("error"):
            return state
        qs = QueryState(**state["query_state"])
        sources = [UnifiedSource(**s) for s in state["raw_sources"]]
        if state.get("enable_filtering", True) and sources:
            filtered = filter_relevance(qs.query_text, sources)
        else:
            filtered = sources
        state["filtered_sources"] = [asdict(s) for s in filtered]
        return state

    def node_summary(state: WorkflowState) -> WorkflowState:
        if state.get("error"):
            return state
        qs = QueryState(**state["query_state"])
        context = ContextAnalysis(**state["context_analysis"])
        sources = [UnifiedSource(**s) for s in state["filtered_sources"]]
        response = generate_answer(qs, context, sources)
        state["response"] = response
        return state

    workflow = StateGraph(WorkflowState)
    workflow.add_node("context_understanding", node_context_understanding)
    workflow.add_node("retrieval", node_retrieval)
    workflow.add_node("filtering", node_filtering)
    workflow.add_node("summary", node_summary)

    workflow.set_entry_point("context_understanding")
    workflow.add_edge("context_understanding", "retrieval")
    workflow.add_edge("retrieval", "filtering")
    workflow.add_edge("filtering", "summary")
    workflow.add_edge("summary", END)

    tax_copilot_graph = workflow.compile()
    print("✅ LangGraph workflow compiled")

def run_workflow(query_state: QueryState, enable_filtering: bool = True) -> Dict[str, Any]:
    """Run the complete workflow."""

    if LANGGRAPH_AVAILABLE:
        initial_state = {
            "query_state": asdict(query_state),
            "enable_filtering": enable_filtering,
            "context_analysis": None,
            "raw_sources": [],
            "filtered_sources": [],
            "response": {},
            "error": None
        }
        final_state = tax_copilot_graph.invoke(initial_state)
        return final_state.get("response", {"error": "Workflow failed"})
    else:
        # Fallback if LangGraph not available
        context = analyze_context(query_state.query_text, query_state.tax_area)
        sources = retrieve_hybrid(query_state.query_text, context, enable_web=query_state.enable_web_search)
        if enable_filtering and sources:
            sources = filter_relevance(query_state.query_text, sources)
        return generate_answer(query_state, context, sources)

print("✅ Workflow runner ready")
```

---

## CELL 15 - Format Output as Memo

```python
def json_to_memo_md(result: dict) -> str:
    if not isinstance(result, dict):
        return "## Error\n\nUnexpected result type."

    if "error" in result:
        tb = result.get("traceback", "")
        raw = result.get("raw_output", "")
        return (
            "## Error\n\n"
            f"**{result['error']}**\n\n"
            + (f"### Traceback\n```text\n{tb}\n```\n" if tb else "")
            + (f"### Raw Output\n```text\n{raw}\n```\n" if raw else "")
        )

    def bullets(items, key="item"):
        if not items:
            return "_None found in provided sources._"
        out = []
        for x in items:
            if isinstance(x, dict):
                text = x.get(key) or x.get("point") or x.get("law") or ""
                cit = x.get("citation", "")
                out.append(f"- {text} **{cit}**" if cit else f"- {text}")
            else:
                out.append(f"- {x}")
        return "\n".join(out)

    md = []
    md.append("# Audit Research Memo")
    md.append("")

    tax_area = result.get("tax_area", "Not specified")
    md.append(f"**Tax Area:** {tax_area}")
    md.append("")

    md.append("## Summary")
    md.append(result.get("summary", "_No summary returned._"))
    md.append("")

    if result.get("relevant_laws"):
        md.append("## Relevant Laws & Regulations")
        md.append(bullets(result.get("relevant_laws", []), key="law"))
        md.append("")

    md.append("## Key Provisions")
    md.append(bullets(result.get("key_provisions", []), key="point"))
    md.append("")

    md.append("## Obligations")
    md.append(bullets(result.get("obligations", []), key="item"))
    md.append("")

    md.append("## Exemptions / Thresholds")
    md.append(bullets(result.get("exemptions", []), key="item"))
    md.append("")

    md.append("## Penalties")
    md.append(bullets(result.get("penalties", []), key="item"))
    md.append("")

    md.append("## Audit Checklist")
    checklist = result.get("audit_checklist", [])
    md.append("\n".join([f"- [ ] {x}" for x in checklist]) if checklist else "_None._")
    md.append("")

    md.append("## Assumptions")
    assumptions = result.get("assumptions", [])
    md.append("\n".join([f"- {x}" for x in assumptions]) if assumptions else "_None._")
    md.append("")

    md.append("## Sources Used")
    srcs = result.get("sources_used", [])
    md.append("\n".join([f"- {x}" for x in srcs]) if srcs else "_See Retrieved Sources panel._")

    web_refs = result.get("web_references", [])
    if web_refs:
        md.append("")
        md.append("## Web References")
        for ref in web_refs:
            title = ref.get("title", "Link")
            url = ref.get("url", "")
            if url:
                md.append(f"- [{title}]({url})")
            else:
                md.append(f"- {title}")

    return "\n".join(md)

print("✅ Memo formatting ready")
```

---

## CELL 16 - Load Demo Files

```python
out_dir = Path("tax_demo_docs")

initial_uploaded_files_store = []
if out_dir.exists():
    for f_path in out_dir.glob("*.pdf"):
        with open(f_path, "rb") as f:
            initial_uploaded_files_store.append({"name": f_path.name, "bytes": f.read()})

print(f"✅ Demo files found: {len(initial_uploaded_files_store)}")
```

---

## CELL 17 - File Upload Handlers

```python
uploaded_files_store = []

def ui_upload(files) -> str:
    global uploaded_files_store
    try:
        uploaded_files_store = []

        if not files:
            return "No new files selected. Use demo files by clicking 'Build Index', or upload new ones."

        for f in files:
            if isinstance(f, str) or hasattr(f, "__fspath__"):
                file_path = os.fspath(f)
                file_name = os.path.basename(file_path)
                with open(file_path, "rb") as fp:
                    file_bytes = fp.read()
            elif isinstance(f, dict) and "name" in f and "data" in f:
                file_name = f["name"]
                file_bytes = f["data"]
            elif hasattr(f, "name"):
                file_path = getattr(f, "name")
                file_name = os.path.basename(file_path)
                with open(file_path, "rb") as fp:
                    file_bytes = fp.read()
            else:
                return f"Unsupported file object type: {type(f)}"

            uploaded_files_store.append({"name": file_name, "bytes": file_bytes})

        return f"✅ Uploaded {len(uploaded_files_store)} file(s). Now click 'Build Index'."

    except Exception as e:
        return f"❌ Upload failed: {type(e).__name__}: {e}\n{traceback.format_exc()}"

def ui_build_index() -> str:
    global uploaded_files_store

    if not uploaded_files_store and initial_uploaded_files_store:
        uploaded_files_store = initial_uploaded_files_store

    if not uploaded_files_store:
        return "🛑 No files to index. Upload PDFs or ensure demo PDFs exist."

    try:
        return build_index(uploaded_files_store)
    except Exception as e:
        return f"❌ Indexing failed: {type(e).__name__}: {e}\n{traceback.format_exc()}"
```

---

## CELL 18 - UI Helper Functions

```python
import gradio as gr
import pandas as pd
import uuid

# PDF Export
def _memo_md_to_plain_lines(memo_md: str):
    lines = []
    for raw in (memo_md or "").splitlines():
        s = raw.strip()
        if not s:
            lines.append("")
            continue
        if s.startswith("### "):
            lines.append(s.replace("### ", "").upper())
            continue
        if s.startswith("## "):
            lines.append(s.replace("## ", "").upper())
            continue
        if s.startswith("# "):
            lines.append(s.replace("# ", "").upper())
            continue
        if s.startswith("- "):
            lines.append("* " + s[2:])
            continue
        s = s.replace("**", "")
        lines.append(s)
    return lines

def export_memo_pdf(memo_md: str, filename_prefix: str = "audit_memo") -> str:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import inch

    export_dir = "exports"
    os.makedirs(export_dir, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    uid = uuid.uuid4().hex[:8]
    pdf_path = os.path.join(export_dir, f"{filename_prefix}_{ts}_{uid}.pdf")

    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter
    x = 0.75 * inch
    y = height - 0.9 * inch

    c.setFont("Helvetica-Bold", 16)
    c.drawString(x, y, "Audit Findings Memorandum")
    y -= 0.35 * inch

    c.setFont("Helvetica", 9)
    c.drawString(x, y, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    y -= 0.35 * inch

    c.setFont("Helvetica", 11)
    lines = _memo_md_to_plain_lines(memo_md)

    def wrap_line(line, max_chars=100):
        if len(line) <= max_chars:
            return [line]
        chunks = []
        words = line.split(" ")
        cur = ""
        for w in words:
            if len(cur) + len(w) + 1 <= max_chars:
                cur = (cur + " " + w).strip()
            else:
                chunks.append(cur)
                cur = w
        if cur:
            chunks.append(cur)
        return chunks

    for line in lines:
        if y < 0.8 * inch:
            c.showPage()
            y = height - 0.9 * inch
            c.setFont("Helvetica", 11)

        for wl in wrap_line(line, max_chars=100):
            if y < 0.8 * inch:
                c.showPage()
                y = height - 0.9 * inch
                c.setFont("Helvetica", 11)
            c.drawString(x, y, wl)
            y -= 0.18 * inch

        if line == "":
            y -= 0.05 * inch

    c.save()
    return pdf_path

def export_memo_markdown(memo_md: str) -> str:
    export_dir = "exports"
    os.makedirs(export_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    md_path = os.path.join(export_dir, f"audit_memo_{ts}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(memo_md)
    return md_path

def evidence_badge(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "### Evidence strength: 🔴 Low\nNo source evidence retrieved."

    unique_sources = df["source"].nunique()
    total_chunks = len(df)
    web_count = len(df[df["provenance"] == "web"]) if "provenance" in df.columns else 0
    local_count = total_chunks - web_count

    if unique_sources >= 3:
        strength = "🟢 High"
    elif unique_sources == 2:
        strength = "🟡 Medium"
    else:
        strength = "🔴 Low"

    return (
        f"### Evidence strength: {strength}\n"
        f"Evidence from {unique_sources} source(s): {local_count} local, {web_count} web "
        f"({total_chunks} total excerpts)."
    )

def ui_ask(
    query: str,
    as_of_date: str,
    top_k: int,
    tax_area: str,
    taxpayer_type: str,
    sector: str,
    detail_level: str,
    constraints: List[str],
    enable_web: bool,
    enable_filtering: bool,
    refine_mode: bool,
    refinement_text: str,
    last_state: dict
):
    try:
        actual_query = query
        if refine_mode and refinement_text and last_state:
            actual_query = f"{query}\n\nREFINEMENT: {refinement_text}"

        query_state = QueryState(
            query_text=actual_query,
            tax_area=tax_area if tax_area != "Auto-detect" else None,
            taxpayer_type=taxpayer_type if taxpayer_type != "Not specified" else None,
            sector=sector if sector != "Not specified" else None,
            detail_level=detail_level.lower(),
            as_of_date=as_of_date or date.today().isoformat(),
            constraints=list(constraints) if constraints else [],
            enable_web_search=enable_web
        )

        result = run_workflow(query_state, enable_filtering=enable_filtering)

        sources = result.get("_retrieved_sources", [])
        df = pd.DataFrame(sources) if sources else pd.DataFrame(
            columns=["rank", "provenance", "source", "chunk_id", "score", "preview", "url"]
        )

        memo = json_to_memo_md(result)
        badge = evidence_badge(df)

        ctx = result.get("_context_analysis", {})
        ctx_info = f"Tax Area: {ctx.get('tax_area', 'N/A')} | Type: {ctx.get('request_type', 'N/A')} | Confidence: {ctx.get('confidence', 0):.0%}"

        return (
            json.dumps(result, indent=2, ensure_ascii=False),
            df,
            memo,
            badge,
            ctx_info,
            result
        )

    except Exception as e:
        err = {
            "error": f"{type(e).__name__}: {str(e)}",
            "traceback": traceback.format_exc()[:6000]
        }
        df = pd.DataFrame(columns=["rank", "provenance", "source", "chunk_id", "score", "preview"])
        memo = json_to_memo_md(err)
        badge = "### Evidence strength: 🔴 Low\nAn error occurred."
        return json.dumps(err, indent=2), df, memo, badge, "Error", err

def show_selected(df: pd.DataFrame, evt: gr.SelectData):
    if df is None or df.empty:
        return "No sources to preview."
    row = df.iloc[evt.index[0]].to_dict()
    url_info = f"\nURL: {row.get('url')}" if row.get('url') else ""
    return (
        f"Source: {row.get('source')}\n"
        f"Chunk: {row.get('chunk_id')}\n"
        f"Provenance: {row.get('provenance', 'local')}\n"
        f"Score: {row.get('score'):.4f}"
        f"{url_info}\n\n"
        f"{row.get('preview')}"
    )

def fill_q1():
    return "Late filing VAT return: what are the penalties and what small business thresholds or exemptions apply?"

def fill_q2():
    return "Provide an audit checklist to verify late filing and late payment, including what evidence to request."

def fill_q3():
    return "What is the corporate income tax rate and what penalties apply?"

def ui_download_pdf(state_result: dict):
    if not isinstance(state_result, dict) or not state_result:
        raise gr.Error("No memo available yet. Please run a query first.")
    memo_md = json_to_memo_md(state_result)
    return export_memo_pdf(memo_md)

def ui_download_md(state_result: dict):
    if not isinstance(state_result, dict) or not state_result:
        raise gr.Error("No memo available yet. Please run a query first.")
    memo_md = json_to_memo_md(state_result)
    return export_memo_markdown(memo_md)

def process_batch(queries_text: str, enable_web: bool, enable_filtering: bool):
    if not queries_text.strip():
        return pd.DataFrame(columns=["query", "tax_area", "summary", "sources_count"])

    queries = [q.strip() for q in queries_text.strip().split("\n") if q.strip()]
    results = []

    for q in queries:
        query_state = QueryState(query_text=q, enable_web_search=enable_web)
        result = run_workflow(query_state, enable_filtering=enable_filtering)
        results.append({
            "query": q[:100],
            "tax_area": result.get("tax_area", result.get("_context_analysis", {}).get("tax_area", "")),
            "summary": result.get("summary", "")[:200],
            "sources_count": len(result.get("_retrieved_sources", []))
        })

    return pd.DataFrame(results)

print("✅ UI functions ready")
```

---

## CELL 19 - Launch the App

```python
with gr.Blocks(title="Tax Audit Copilot") as demo:
    gr.Markdown("""
# Tax Audit Research & Decision Support
**Workflow:** Upload PDFs → Build Index → Ask scenario → Review memo + evidence → Download

**By: Abdulla Ahmed Alaydaroos**
""")

    state_result = gr.State({})

    with gr.Tab("1) Upload & Index"):
        uploader = gr.File(file_count="multiple", label="Upload PDFs/DOCX/TXT")
        upload_status = gr.Textbox(label="Upload status", interactive=False)
        uploader.change(fn=ui_upload, inputs=uploader, outputs=upload_status)

        build_btn = gr.Button("Build Index", variant="primary")
        build_status = gr.Textbox(label="Index status", interactive=False)
        build_btn.click(fn=ui_build_index, inputs=None, outputs=build_status)

        gr.Markdown(f"""
---
**Web Search Status:** {'✅ Enabled (Tavily API key loaded)' if TAVILY_API_KEY else '⚠️ Disabled (add TAVILY_API_KEY to Colab Secrets)'}
""")

    with gr.Tab("2) Ask"):
        with gr.Row():
            with gr.Column(scale=2):
                query = gr.Textbox(
                    label="Audit question / scenario",
                    lines=4,
                    placeholder="Example: Late filing VAT return - penalties, thresholds, exemptions..."
                )

                with gr.Row():
                    demo_q1 = gr.Button("Demo: Late filing + thresholds")
                    demo_q2 = gr.Button("Demo: Audit checklist")
                    demo_q3 = gr.Button("Demo: Refusal test")

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
                            choices=["Not specified", "Real Estate", "Financial Services", "Manufacturing", "Retail", "Technology", "Healthcare", "Oil & Gas"],
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

                with gr.Row():
                    as_of = gr.Textbox(label="As-of date", value="Today", scale=1)
                    topk = gr.Slider(3, 10, value=6, step=1, label="Top-K excerpts", scale=1)

                with gr.Row():
                    enable_web = gr.Checkbox(
                        value=TAVILY_API_KEY is not None,
                        label="Enable Web Search",
                        interactive=TAVILY_API_KEY is not None
                    )
                    enable_filtering = gr.Checkbox(
                        value=True,
                        label="Enable Relevance Filtering"
                    )

                with gr.Accordion("Refine Previous Answer", open=False):
                    refine_mode = gr.Checkbox(value=False, label="Refine previous answer")
                    refinement_text = gr.Textbox(
                        label="Refinement instruction",
                        placeholder="e.g., Focus on penalties only, Exclude VAT, Add more detail...",
                        lines=2
                    )

                ask_btn = gr.Button("Generate Audit Summary", variant="primary")

                context_info = gr.Textbox(label="Context Analysis", interactive=False)
                badge_md = gr.Markdown()

                with gr.Row():
                    download_pdf_btn = gr.Button("Download PDF")
                    download_md_btn = gr.Button("Download Markdown")
                download_file = gr.File(label="Download", interactive=False)

            with gr.Column(scale=3):
                out_memo = gr.Markdown(label="Audit Findings Memorandum")

        gr.Markdown("---")

        with gr.Row():
            with gr.Column(scale=2):
                out_sources = gr.Dataframe(
                    label="Source Evidence (click row to preview)",
                    interactive=False,
                    wrap=True
                )
                selected_preview = gr.Textbox(label="Selected Evidence Preview", lines=8, interactive=False)

            with gr.Column(scale=2):
                out_json = gr.Code(label="Structured Output (JSON)", language="json")

        # Events
        demo_q1.click(fill_q1, outputs=query)
        demo_q2.click(fill_q2, outputs=query)
        demo_q3.click(fill_q3, outputs=query)

        ask_btn.click(
            fn=ui_ask,
            inputs=[
                query, as_of, topk,
                tax_area, taxpayer_type, sector, detail_level, constraints,
                enable_web, enable_filtering,
                refine_mode, refinement_text, state_result
            ],
            outputs=[out_json, out_sources, out_memo, badge_md, context_info, state_result]
        )

        out_sources.select(fn=show_selected, inputs=out_sources, outputs=selected_preview)

        download_pdf_btn.click(fn=ui_download_pdf, inputs=state_result, outputs=download_file)
        download_md_btn.click(fn=ui_download_md, inputs=state_result, outputs=download_file)

    with gr.Tab("3) Batch Analysis"):
        gr.Markdown("""
### Batch Query Processing
Enter multiple queries (one per line) to process them all at once.
""")

        batch_input = gr.Textbox(
            label="Queries (one per line)",
            lines=6,
            placeholder="Query 1: What are VAT penalties?\nQuery 2: Corporate tax rate in UAE?\nQuery 3: Excise tax on tobacco?"
        )

        with gr.Row():
            batch_web = gr.Checkbox(value=TAVILY_API_KEY is not None, label="Enable Web Search", interactive=TAVILY_API_KEY is not None)
            batch_filter = gr.Checkbox(value=True, label="Enable Filtering")

        batch_btn = gr.Button("Process Batch", variant="primary")
        batch_output = gr.Dataframe(label="Results", wrap=True)

        batch_btn.click(
            fn=process_batch,
            inputs=[batch_input, batch_web, batch_filter],
            outputs=batch_output
        )

demo.launch(share=False)
```

---

## How to Use

1. Open a new Google Colab notebook
2. Copy each cell above in order (Cell 1 to Cell 19)
3. Add your API keys to Colab Secrets (key icon on left sidebar):
   - `OPEN_AI_API` - Required
   - `TAVILY_API_KEY` - Optional (for web search)
4. Run all cells
5. Use the Gradio interface

---

## System Overview

```
┌─────────┐    ┌──────────────────┐    ┌───────────┐    ┌───────────┐    ┌─────────┐
│  Input  │───▶│ Query Classification│───▶│ Retrieval │───▶│ Filtering │───▶│ Summary │
└─────────┘    └──────────────────┘    └───────────┘    └───────────┘    └─────────┘
                                              │
                                        ┌─────┴─────┐
                                        │           │
                                   ┌────┴────┐ ┌────┴────┐
                                   │  Local  │ │   Web   │
                                   │ (FAISS) │ │(Tavily) │
                                   └─────────┘ └─────────┘
```

---

**Capstone Project - Abdulla Ahmed Alaydaroos**
