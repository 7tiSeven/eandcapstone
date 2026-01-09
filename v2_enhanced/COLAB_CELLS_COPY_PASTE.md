# Audit Readiness Assistant - Colab Cells

**Capstone Project by: Abdulla Ahmed Alaydaroos**

This document contains all code cells for the AI-Powered Audit Readiness Assistant. Copy each cell into Google Colab in order.

**Before starting:** Add your API keys in Colab Secrets:
- `OPEN_AI_API` (required)
- `TAVILY_API_KEY` (optional - enables web search)

---

## Cell 1 - Install Dependencies

```python
# Cell 1 - Install Dependencies
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

print("All packages installed successfully.")
```

---

## Cell 2 - Load API Credentials

```python
# Cell 2 - Load API Credentials
from google.colab import userdata

API_KEY = userdata.get("OPEN_AI_API")
assert API_KEY, "Missing Colab Secret: OPEN_AI_API"
BASE_URL = "https://aibe.mygreatlearning.com/openai/v1"
print("API key loaded successfully.")
print(f"Using endpoint: {BASE_URL}")

try:
    TAVILY_API_KEY = userdata.get("TAVILY_API_KEY")
    if TAVILY_API_KEY:
        print("Tavily API key loaded - web search is enabled")
    else:
        TAVILY_API_KEY = None
        print("Tavily API key not set - web search is disabled")
except Exception:
    TAVILY_API_KEY = None
    print("Tavily API key not found - web search is disabled")
```

---

## Cell 3 - Core Imports

```python
# Cell 3 - Core Imports
import os
import io
import re
import json
import traceback
import uuid
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from enum import Enum
from pathlib import Path

from pypdf import PdfReader
import docx
import pandas as pd

print("Core imports completed.")
```

---

## Cell 4 - Data Structures

```python
# Cell 4 - Data Structures

class EntityType(str, Enum):
    GOVERNMENT = "Government Entity"
    SEMI_GOVERNMENT = "Semi-Government Entity"
    PRIVATE = "Private Sector"
    NON_PROFIT = "Non-Profit Organization"

class EntitySize(str, Enum):
    SMALL = "Small (< 50 employees)"
    MEDIUM = "Medium (50-250 employees)"
    LARGE = "Large (> 250 employees)"

class ComplianceArea(str, Enum):
    FINANCIAL_REPORTING = "Financial Reporting"
    INTERNAL_CONTROLS = "Internal Controls"
    ASSET_MANAGEMENT = "Asset Management"
    PROCUREMENT = "Procurement & Contracts"
    HR_PAYROLL = "HR & Payroll"
    IT_SYSTEMS = "IT Systems & Security"
    REGULATORY = "Regulatory Compliance"
    GOVERNANCE = "Governance & Oversight"

class AssessmentStatus(str, Enum):
    COMPLIANT = "Compliant"
    PARTIAL = "Partially Compliant"
    NON_COMPLIANT = "Non-Compliant"
    NOT_ASSESSED = "Not Yet Assessed"
    NOT_APPLICABLE = "Not Applicable"

class RiskLevel(str, Enum):
    CRITICAL = "Critical"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"

@dataclass
class EntityProfile:
    entity_name: str
    entity_type: str
    sector: str
    size_category: str
    reporting_framework: str
    fiscal_year_end: str
    years_in_operation: int = 0
    total_employees: int = 0
    annual_budget: str = ""
    prior_audit_rating: Optional[str] = None
    notes: str = ""

@dataclass
class ComplianceIndicator:
    area: str
    status: str
    has_documentation: bool = False
    has_policies: bool = False
    last_review_date: Optional[str] = None
    notes: str = ""

@dataclass
class PriorFinding:
    finding_id: str
    category: str
    severity: str
    status: str
    description: str
    year_identified: int = 0
    remediation_plan: str = ""
    target_date: str = ""

@dataclass
class IdentifiedGap:
    gap_id: str
    area: str
    description: str
    risk_level: str
    requirement_reference: str
    recommendation: str
    evidence_needed: List[str] = field(default_factory=list)

@dataclass
class AuditReadinessState:
    entity: Optional[EntityProfile] = None
    compliance_indicators: List[ComplianceIndicator] = field(default_factory=list)
    prior_findings: List[PriorFinding] = field(default_factory=list)
    identified_gaps: List[IdentifiedGap] = field(default_factory=list)
    overall_risk_score: float = 0.0
    readiness_level: str = "Not Assessed"
    assessment_date: str = field(default_factory=lambda: date.today().isoformat())

@dataclass
class UnifiedSource:
    provenance: str
    source_id: str
    chunk_id: str
    content: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_citation(self) -> str:
        return f"[{self.provenance} | {self.source_id}]"

print("Data structures defined successfully.")
print(f"Compliance areas: {[a.value for a in ComplianceArea]}")
```

---

## Cell 5 - File Reading Utilities

```python
# Cell 5 - File Reading Utilities

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

print("File reading utilities ready.")
```

---

## Cell 6 - Demo Standards Documents

```python
# Cell 6 - Demo Standards Documents
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
import textwrap
import zipfile

out_dir = Path("standards_docs")
out_dir.mkdir(exist_ok=True)

def make_pdf(path, title, sections):
    c = canvas.Canvas(str(path), pagesize=letter)
    w, h = letter
    x, y = 0.75*inch, h - 0.9*inch

    c.setFont("Helvetica-Bold", 16)
    c.drawString(x, y, title)
    y -= 0.4*inch

    c.setFont("Helvetica", 9)
    c.drawString(x, y, f"DEMO DOCUMENT - {date.today()} (For Educational Purposes)")
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

# IFRS Standards
make_pdf(
    out_dir / "IFRS_Standards_Summary.pdf",
    "IFRS Financial Reporting Standards (Summary)",
    [
        ("IFRS 1 - First-time Adoption",
         "Entities adopting IFRS for the first time must prepare an opening IFRS statement of "
         "financial position. Full retrospective application is required with limited exemptions. "
         "Comparative information for at least one prior period must be presented."),
        ("IFRS 15 - Revenue Recognition",
         "Revenue is recognized when control of goods or services transfers to the customer. "
         "The five-step model requires: (1) identify contract, (2) identify performance obligations, "
         "(3) determine transaction price, (4) allocate price, (5) recognize revenue."),
        ("IFRS 16 - Leases",
         "Lessees must recognize a right-of-use asset and lease liability for most leases. "
         "Short-term leases (under 12 months) and low-value assets may be exempt. "
         "Disclosure of lease obligations and maturity analysis is required."),
        ("IAS 1 - Presentation of Financial Statements",
         "Financial statements must include: statement of financial position, statement of profit or loss, "
         "statement of changes in equity, statement of cash flows, and notes. "
         "Fair presentation and compliance with IFRS must be explicitly stated.")
    ]
)

# Internal Control Framework
make_pdf(
    out_dir / "Internal_Control_Framework.pdf",
    "Internal Control Framework Requirements",
    [
        ("Control Environment",
         "The organization must establish a control environment that demonstrates commitment to integrity "
         "and ethical values. Board oversight must be independent. Organizational structure must define "
         "clear reporting lines and responsibilities."),
        ("Risk Assessment",
         "Management must identify and assess risks to achieving objectives. Risk assessment must consider "
         "likelihood and impact. Fraud risk must be explicitly considered. Changes that could affect "
         "internal control must be identified."),
        ("Control Activities",
         "Control activities must be designed to mitigate identified risks. Segregation of duties is required "
         "for key processes. Authorization controls must be documented. IT general controls must protect "
         "systems and data integrity."),
        ("Information and Communication",
         "Relevant information must be captured and communicated timely. Internal communication must "
         "support internal control. External communication must be appropriate and controlled."),
        ("Monitoring Activities",
         "Ongoing monitoring must evaluate control effectiveness. Internal audit function should be "
         "independent. Control deficiencies must be reported to appropriate levels. "
         "Corrective actions must be tracked to completion.")
    ]
)

# ADAA Requirements
make_pdf(
    out_dir / "ADAA_Audit_Requirements.pdf",
    "ADAA Audit Requirements and Guidelines",
    [
        ("Documentation Requirements",
         "Entities must maintain complete and accurate records of all financial transactions. "
         "Supporting documentation must be retained for minimum 7 years. "
         "Electronic records must have appropriate backup and recovery procedures."),
        ("Financial Reporting Deadlines",
         "Annual financial statements must be prepared within 3 months of fiscal year end. "
         "Quarterly reports are required for government entities. "
         "Audit reports must be submitted within 6 months of year end."),
        ("Governance Requirements",
         "Audit committee must meet at least quarterly. Internal audit function must report "
         "directly to audit committee. Conflict of interest policies must be documented and enforced. "
         "Whistleblower mechanisms must be established."),
        ("Asset Management",
         "Fixed asset register must be maintained and reconciled annually. Physical verification "
         "of assets must be performed. Disposal procedures must be documented with proper approvals. "
         "Impairment must be assessed annually."),
        ("Procurement Standards",
         "Procurement must follow competitive bidding for amounts exceeding thresholds. "
         "Vendor evaluation criteria must be documented. Contract management procedures must be "
         "in place. Purchase orders must precede goods receipt.")
    ]
)

# Audit Readiness Checklist
make_pdf(
    out_dir / "Audit_Readiness_Checklist.pdf",
    "Audit Readiness Checklist",
    [
        ("Financial Reporting Readiness",
         "Checklist: (1) Chart of accounts aligned with reporting framework, (2) Month-end close "
         "procedures documented, (3) Journal entry approval process in place, (4) Reconciliations "
         "performed and reviewed monthly, (5) Financial statement preparation timeline established."),
        ("Documentation Completeness",
         "Required documents: (1) Board meeting minutes, (2) Policy and procedure manuals, "
         "(3) Organizational charts, (4) Delegation of authority matrix, (5) Risk register, "
         "(6) Internal audit reports, (7) Prior audit reports and management responses."),
        ("Control Evidence",
         "Evidence to prepare: (1) Bank reconciliations with sign-off, (2) Accounts receivable aging, "
         "(3) Inventory count documentation, (4) Fixed asset verification, (5) Payroll reconciliations, "
         "(6) Access control reviews, (7) IT change management logs."),
        ("Common Deficiencies",
         "Watch for: (1) Missing segregation of duties, (2) Incomplete supporting documentation, "
         "(3) Untimely reconciliations, (4) Lack of formal policies, (5) Inadequate IT controls, "
         "(6) Unremediated prior findings, (7) Insufficient audit trail.")
    ]
)

# HR and Payroll Standards
make_pdf(
    out_dir / "HR_Payroll_Standards.pdf",
    "HR and Payroll Compliance Standards",
    [
        ("Payroll Processing Controls",
         "Payroll changes must be authorized by HR and approved by department head. "
         "Segregation required between payroll preparation, approval, and payment. "
         "Payroll reconciliation to general ledger must be performed monthly."),
        ("Employee Records",
         "Personnel files must contain: employment contract, identification documents, "
         "qualifications verification, performance evaluations, and disciplinary records. "
         "Records must be secured with restricted access."),
        ("Leave and Benefits",
         "Leave balances must be tracked and reconciled. End-of-service benefits must be "
         "calculated according to labor law. Accruals must be recorded in financial statements.")
    ]
)

# IT Controls
make_pdf(
    out_dir / "IT_Control_Standards.pdf",
    "IT General Controls Standards",
    [
        ("Access Controls",
         "User access must follow least-privilege principle. Access reviews must be performed "
         "quarterly. Terminated employee access must be revoked within 24 hours. "
         "Privileged access must be logged and monitored."),
        ("Change Management",
         "All system changes must be authorized, tested, and approved before implementation. "
         "Segregation required between development and production environments. "
         "Emergency changes must be documented and ratified."),
        ("Backup and Recovery",
         "Backups must be performed daily for critical systems. Backup restoration must be "
         "tested quarterly. Offsite backup storage is required. "
         "Business continuity plan must be documented and tested annually.")
    ]
)

zip_path = out_dir / "standards_docs.zip"
with zipfile.ZipFile(zip_path, "w") as z:
    for f in out_dir.glob("*.pdf"):
        z.write(f, f.name)

print("Demo standards documents created:")
for f in sorted(out_dir.glob("*.pdf")):
    print(f"  - {f.name}")
```

---

## Cell 7 - Vector Store Setup

```python
# Cell 7 - Vector Store Setup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=900,
    chunk_overlap=150,
    separators=["\n\n", "\n", "Section ", "Article ", ". ", " "]
)

VECTORSTORE = None

def build_index(files: List[Dict[str, Any]]) -> str:
    global VECTORSTORE

    docs = []
    for f in files:
        text = read_file_bytes(f["name"], f["bytes"])
        if len(text) < 50:
            print(f"Warning: Low text extracted from {f['name']}")
        docs.append(Document(page_content=text, metadata={"source": f["name"]}))

    chunks = splitter.split_documents(docs)
    VECTORSTORE = FAISS.from_documents(chunks, embeddings)

    return f"Indexed {len(files)} document(s) into {len(chunks)} searchable chunks."

print("Vector store configured.")
```

---

## Cell 8 - LLM Setup

```python
# Cell 8 - LLM Setup
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.1,
    api_key=API_KEY,
    base_url=BASE_URL
)

llm_fast = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.0,
    api_key=API_KEY,
    base_url=BASE_URL
)

print("Language model configured.")
```

---

## Cell 9 - Web Search Integration

```python
# Cell 9 - Web Search Integration
TRUSTED_DOMAINS = [
    "adaa.gov.ae",
    "government.ae",
    "mof.gov.ae",
    "ifrs.org",
    "iasb.org",
]

EXTENDED_TRUSTED = TRUSTED_DOMAINS + [
    "pwc.com",
    "ey.com",
    "kpmg.com",
    "deloitte.com",
    "aicpa.org",
]

@dataclass
class WebSearchResult:
    url: str
    domain: str
    title: str
    snippet: str
    score: float
    is_official: bool = False

def search_web_tavily(query: str, max_results: int = 5) -> List[WebSearchResult]:
    if not TAVILY_API_KEY:
        return []

    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=TAVILY_API_KEY)

        enhanced_query = f"audit compliance {query}"

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
                is_official=is_official
            ))

        results.sort(key=lambda x: (x.is_official, x.score), reverse=True)
        return results[:max_results]

    except Exception as e:
        print(f"Web search error: {e}")
        return []

print(f"Web search configured. Tavily enabled: {TAVILY_API_KEY is not None}")
```

---

## Cell 10 - Retrieval Functions

```python
# Cell 10 - Retrieval Functions

def retrieve_local(query: str, k: int = 6) -> List[UnifiedSource]:
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

def retrieve_web(query: str, k: int = 3) -> List[UnifiedSource]:
    web_results = search_web_tavily(query, max_results=k)
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
                "is_official": ws.is_official
            }
        ))
    return sources

def retrieve_standards(query: str, enable_web: bool = True) -> List[UnifiedSource]:
    all_sources = []

    local_sources = retrieve_local(query, k=5)
    all_sources.extend(local_sources)

    if enable_web and TAVILY_API_KEY:
        web_sources = retrieve_web(query, k=3)
        all_sources.extend(web_sources)

    return all_sources

def format_sources_for_prompt(sources: List[UnifiedSource]) -> str:
    blocks = []
    for src in sources:
        header = src.to_citation()
        url = src.metadata.get("url", "")
        extra = f" | URL: {url}" if url else ""
        blocks.append(f"SOURCE {src.chunk_id} {header}{extra}:\n{src.content}")
    return "\n\n---\n\n".join(blocks)

print("Retrieval functions ready.")
```

---

## Cell 11 - Gap Analysis Prompts

```python
# Cell 11 - Gap Analysis Prompts

GAP_ANALYSIS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an audit readiness assessment expert. Your task is to identify compliance gaps based on:
1. The organization's profile and context
2. Their self-assessment of compliance areas
3. Any prior audit findings
4. Relevant regulatory standards and requirements

Analyze the inputs and identify specific compliance gaps. For each gap:
- Describe the gap clearly
- Reference the specific requirement not being met
- Assess the risk level (Critical, High, Medium, Low)
- Provide actionable recommendations
- List evidence that should be prepared

Return ONLY valid JSON with these keys:
- gaps: array of gap objects, each with:
  - gap_id: string (e.g., "GAP-001")
  - area: string (compliance area)
  - description: string (clear description of the gap)
  - risk_level: string (Critical, High, Medium, or Low)
  - requirement_reference: string (which standard/requirement is not met)
  - recommendation: string (what to do to address it)
  - evidence_needed: array of strings (documents/evidence to prepare)
- overall_risk_score: float 0.0-10.0 (10 being highest risk)
- readiness_level: string ("Ready", "Partially Ready", "Not Ready", "Critical Gaps")
- summary: string (brief overall assessment)
- priority_actions: array of strings (top 3-5 actions to take)

Be thorough but practical. Focus on material gaps that would concern auditors."""),
    ("human", """ENTITY PROFILE:
{entity_profile}

COMPLIANCE SELF-ASSESSMENT:
{compliance_indicators}

PRIOR AUDIT FINDINGS:
{prior_findings}

RELEVANT STANDARDS AND REQUIREMENTS:
{standards}

Analyze the above and identify all compliance gaps. Return ONLY JSON.""")
])

print("Gap analysis prompts defined.")
```

---

## Cell 12 - Gap Analysis Functions

```python
# Cell 12 - Gap Analysis Functions

def format_entity_profile(entity: EntityProfile) -> str:
    if entity is None:
        return "No entity profile provided."

    return f"""- Entity Name: {entity.entity_name}
- Entity Type: {entity.entity_type}
- Sector: {entity.sector}
- Size: {entity.size_category}
- Reporting Framework: {entity.reporting_framework}
- Fiscal Year End: {entity.fiscal_year_end}
- Years in Operation: {entity.years_in_operation}
- Total Employees: {entity.total_employees}
- Annual Budget: {entity.annual_budget}
- Prior Audit Rating: {entity.prior_audit_rating or 'Not available'}
- Notes: {entity.notes or 'None'}"""

def format_compliance_indicators(indicators: List[ComplianceIndicator]) -> str:
    if not indicators:
        return "No compliance self-assessment provided."

    lines = []
    for ind in indicators:
        lines.append(f"""- {ind.area}:
  Status: {ind.status}
  Documentation: {'Yes' if ind.has_documentation else 'No'}
  Policies: {'Yes' if ind.has_policies else 'No'}
  Last Review: {ind.last_review_date or 'Not specified'}
  Notes: {ind.notes or 'None'}""")
    return "\n".join(lines)

def format_prior_findings(findings: List[PriorFinding]) -> str:
    if not findings:
        return "No prior audit findings recorded."

    lines = []
    for f in findings:
        lines.append(f"""- {f.finding_id} ({f.category}):
  Severity: {f.severity}
  Status: {f.status}
  Description: {f.description}
  Year Identified: {f.year_identified or 'Not specified'}
  Remediation Plan: {f.remediation_plan or 'None documented'}""")
    return "\n".join(lines)

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

    return {"error": "Failed to parse LLM response", "raw_output": raw[:3000]}, raw

def analyze_gaps(
    entity: EntityProfile,
    indicators: List[ComplianceIndicator],
    findings: List[PriorFinding],
    enable_web: bool = True
) -> Dict[str, Any]:

    search_queries = []
    if entity:
        search_queries.append(f"{entity.reporting_framework} requirements {entity.entity_type}")

    for ind in indicators:
        if ind.status in ["Partially Compliant", "Non-Compliant", "Not Yet Assessed"]:
            search_queries.append(f"{ind.area} compliance requirements")

    all_sources = []
    for query in search_queries[:5]:
        sources = retrieve_standards(query, enable_web=enable_web)
        all_sources.extend(sources)

    seen = set()
    unique_sources = []
    for s in all_sources:
        key = (s.source_id, s.content[:100])
        if key not in seen:
            seen.add(key)
            unique_sources.append(s)

    standards_text = format_sources_for_prompt(unique_sources[:10])

    if not standards_text:
        standards_text = "No specific standards retrieved. Use general audit best practices."

    try:
        msg = GAP_ANALYSIS_PROMPT.format_messages(
            entity_profile=format_entity_profile(entity),
            compliance_indicators=format_compliance_indicators(indicators),
            prior_findings=format_prior_findings(findings),
            standards=standards_text
        )
        response = llm.invoke(msg).content
        result, raw = _safe_json_parse(response)

        result["_sources"] = [
            {
                "provenance": s.provenance,
                "source": s.source_id,
                "preview": s.content[:200],
                "url": s.metadata.get("url", "")
            }
            for s in unique_sources[:10]
        ]

        return result

    except Exception as e:
        return {
            "error": f"Gap analysis failed: {type(e).__name__}: {e}",
            "traceback": traceback.format_exc()[:2000]
        }

print("Gap analysis functions ready.")
```

---

## Cell 13 - LangGraph Workflow

```python
# Cell 13 - LangGraph Workflow

try:
    from langgraph.graph import StateGraph, END
    from typing import TypedDict
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    print("LangGraph not available, using fallback workflow")

if LANGGRAPH_AVAILABLE:
    class WorkflowState(TypedDict):
        entity: Optional[Dict]
        indicators: List[Dict]
        findings: List[Dict]
        enable_web: bool
        retrieved_standards: List[Dict]
        gap_analysis: Dict
        error: Optional[str]

    def node_retrieve_standards(state: WorkflowState) -> WorkflowState:
        if state.get("error"):
            return state

        entity = state.get("entity") or {}
        indicators = state.get("indicators") or []

        queries = []
        if entity.get("reporting_framework"):
            queries.append(f"{entity['reporting_framework']} financial reporting requirements")
        if entity.get("entity_type"):
            queries.append(f"{entity['entity_type']} audit requirements")

        for ind in indicators:
            if ind.get("status") in ["Partially Compliant", "Non-Compliant"]:
                queries.append(f"{ind['area']} compliance standards")

        all_sources = []
        for q in queries[:5]:
            sources = retrieve_standards(q, enable_web=state.get("enable_web", True))
            all_sources.extend([asdict(s) for s in sources])

        state["retrieved_standards"] = all_sources[:15]
        return state

    def node_analyze_gaps(state: WorkflowState) -> WorkflowState:
        if state.get("error"):
            return state

        entity = EntityProfile(**state["entity"]) if state.get("entity") else None
        indicators = [ComplianceIndicator(**i) for i in state.get("indicators", [])]
        findings = [PriorFinding(**f) for f in state.get("findings", [])]

        result = analyze_gaps(entity, indicators, findings, enable_web=state.get("enable_web", True))
        state["gap_analysis"] = result
        return state

    workflow = StateGraph(WorkflowState)

    workflow.add_node("retrieve_standards", node_retrieve_standards)
    workflow.add_node("analyze_gaps", node_analyze_gaps)

    workflow.set_entry_point("retrieve_standards")
    workflow.add_edge("retrieve_standards", "analyze_gaps")
    workflow.add_edge("analyze_gaps", END)

    audit_workflow = workflow.compile()
    print("LangGraph workflow compiled.")

def run_assessment(
    entity: EntityProfile,
    indicators: List[ComplianceIndicator],
    findings: List[PriorFinding],
    enable_web: bool = True
) -> Dict[str, Any]:

    if LANGGRAPH_AVAILABLE:
        initial_state = {
            "entity": asdict(entity) if entity else None,
            "indicators": [asdict(i) for i in indicators],
            "findings": [asdict(f) for f in findings],
            "enable_web": enable_web,
            "retrieved_standards": [],
            "gap_analysis": {},
            "error": None
        }

        final_state = audit_workflow.invoke(initial_state)
        return final_state.get("gap_analysis", {"error": "Workflow failed"})
    else:
        return analyze_gaps(entity, indicators, findings, enable_web)

print("Assessment workflow ready.")
```

---

## Cell 14 - Report Formatting

```python
# Cell 14 - Report Formatting

def format_readiness_report(result: Dict[str, Any], entity: EntityProfile = None) -> str:

    if not isinstance(result, dict):
        return "## Error\n\nUnexpected result format."

    if "error" in result:
        return f"""## Error

**{result['error']}**

{result.get('traceback', '')}"""

    md = []

    md.append("# Audit Readiness Assessment Report")
    md.append("")
    md.append(f"**Assessment Date:** {date.today().isoformat()}")
    if entity:
        md.append(f"**Entity:** {entity.entity_name}")
        md.append(f"**Type:** {entity.entity_type} | **Sector:** {entity.sector}")
    md.append("")

    md.append("## Overall Assessment")
    md.append("")

    readiness = result.get("readiness_level", "Not Assessed")
    risk_score = result.get("overall_risk_score", 0)

    if readiness == "Ready":
        badge = "LOW RISK - Ready for Audit"
    elif readiness == "Partially Ready":
        badge = "MEDIUM RISK - Some Gaps to Address"
    elif readiness == "Not Ready":
        badge = "HIGH RISK - Significant Gaps"
    else:
        badge = "CRITICAL RISK - Major Issues"

    md.append(f"**Readiness Level:** {readiness}")
    md.append(f"**Overall Risk Score:** {risk_score:.1f} / 10")
    md.append(f"**Assessment:** {badge}")
    md.append("")

    if result.get("summary"):
        md.append("### Summary")
        md.append(result["summary"])
        md.append("")

    actions = result.get("priority_actions", [])
    if actions:
        md.append("## Priority Actions")
        md.append("")
        for i, action in enumerate(actions, 1):
            md.append(f"{i}. {action}")
        md.append("")

    gaps = result.get("gaps", [])
    if gaps:
        md.append("## Identified Compliance Gaps")
        md.append("")

        for risk in ["Critical", "High", "Medium", "Low"]:
            risk_gaps = [g for g in gaps if g.get("risk_level") == risk]
            if risk_gaps:
                md.append(f"### {risk} Risk Gaps")
                md.append("")
                for gap in risk_gaps:
                    md.append(f"**{gap.get('gap_id', 'GAP')}** - {gap.get('area', 'General')}")
                    md.append(f"")
                    md.append(f"- **Description:** {gap.get('description', 'N/A')}")
                    md.append(f"- **Requirement:** {gap.get('requirement_reference', 'N/A')}")
                    md.append(f"- **Recommendation:** {gap.get('recommendation', 'N/A')}")
                    evidence = gap.get("evidence_needed", [])
                    if evidence:
                        md.append(f"- **Evidence Needed:**")
                        for e in evidence:
                            md.append(f"  - {e}")
                    md.append("")
    else:
        md.append("## Identified Compliance Gaps")
        md.append("")
        md.append("_No specific gaps identified based on the provided information._")
        md.append("")

    sources = result.get("_sources", [])
    if sources:
        md.append("## Reference Sources")
        md.append("")
        for src in sources[:5]:
            url = src.get("url", "")
            if url:
                md.append(f"- [{src['source']}]({url})")
            else:
                md.append(f"- {src['source']}")
        md.append("")

    return "\n".join(md)

def export_report_pdf(report_md: str, entity_name: str = "entity") -> str:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import inch

    export_dir = Path("exports")
    export_dir.mkdir(exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = re.sub(r'[^a-zA-Z0-9]', '_', entity_name)[:30]
    pdf_path = export_dir / f"audit_readiness_{safe_name}_{ts}.pdf"

    c = canvas.Canvas(str(pdf_path), pagesize=letter)
    width, height = letter
    x = 0.75 * inch
    y = height - 0.9 * inch

    c.setFont("Helvetica-Bold", 16)
    c.drawString(x, y, "Audit Readiness Assessment Report")
    y -= 0.35 * inch

    c.setFont("Helvetica", 9)
    c.drawString(x, y, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    y -= 0.35 * inch

    c.setFont("Helvetica", 10)

    for line in report_md.split("\n"):
        if y < 0.8 * inch:
            c.showPage()
            y = height - 0.9 * inch
            c.setFont("Helvetica", 10)

        line = line.strip()
        if line.startswith("# "):
            c.setFont("Helvetica-Bold", 14)
            line = line[2:]
        elif line.startswith("## "):
            c.setFont("Helvetica-Bold", 12)
            line = line[3:]
        elif line.startswith("### "):
            c.setFont("Helvetica-Bold", 11)
            line = line[4:]
        else:
            c.setFont("Helvetica", 10)

        line = line.replace("**", "").replace("_", "")

        for wrapped in textwrap.wrap(line, 90) or [""]:
            if y < 0.8 * inch:
                c.showPage()
                y = height - 0.9 * inch
            c.drawString(x, y, wrapped)
            y -= 0.18 * inch

    c.save()
    return str(pdf_path)

print("Report formatting ready.")
```

---

## Cell 15 - Load Demo Documents

```python
# Cell 15 - Load Demo Documents

out_dir = Path("standards_docs")

initial_files = []
if out_dir.exists():
    for f_path in out_dir.glob("*.pdf"):
        with open(f_path, "rb") as f:
            initial_files.append({"name": f_path.name, "bytes": f.read()})

print(f"Found {len(initial_files)} demo standards documents.")

uploaded_files_store = []
```

---

## Cell 16 - UI Helper Functions

```python
# Cell 16 - UI Helper Functions

def ui_upload(files) -> str:
    global uploaded_files_store
    try:
        uploaded_files_store = []

        if not files:
            return "No files selected. You can use the demo documents."

        for f in files:
            if isinstance(f, str) or hasattr(f, "__fspath__"):
                file_path = os.fspath(f)
                file_name = os.path.basename(file_path)
                with open(file_path, "rb") as fp:
                    file_bytes = fp.read()
            elif hasattr(f, "name"):
                file_path = getattr(f, "name")
                file_name = os.path.basename(file_path)
                with open(file_path, "rb") as fp:
                    file_bytes = fp.read()
            else:
                return f"Unsupported file type: {type(f)}"

            uploaded_files_store.append({"name": file_name, "bytes": file_bytes})

        return f"Uploaded {len(uploaded_files_store)} file(s). Click 'Build Index' to process."

    except Exception as e:
        return f"Upload failed: {e}"

def ui_build_index() -> str:
    global uploaded_files_store

    if not uploaded_files_store and initial_files:
        uploaded_files_store = initial_files

    if not uploaded_files_store:
        return "No documents to index. Please upload files or use demo documents."

    try:
        return build_index(uploaded_files_store)
    except Exception as e:
        return f"Indexing failed: {e}"

print("UI helper functions ready.")
```

---

## Cell 17 - Gradio User Interface

```python
# Cell 17 - Gradio User Interface

import gradio as gr

current_entity = None
current_indicators = []
current_findings = []
current_result = {}

def run_full_assessment(
    entity_name, entity_type, sector, size, framework, fiscal_year,
    years_operation, employees, budget, prior_rating, entity_notes,
    indicators_json,
    findings_json,
    enable_web
):
    global current_entity, current_indicators, current_findings, current_result

    try:
        entity = EntityProfile(
            entity_name=entity_name or "Unnamed Entity",
            entity_type=entity_type,
            sector=sector,
            size_category=size,
            reporting_framework=framework,
            fiscal_year_end=fiscal_year,
            years_in_operation=int(years_operation) if years_operation else 0,
            total_employees=int(employees) if employees else 0,
            annual_budget=budget or "",
            prior_audit_rating=prior_rating if prior_rating != "Not Available" else None,
            notes=entity_notes or ""
        )
        current_entity = entity

        indicators = []
        if indicators_json:
            try:
                ind_list = json.loads(indicators_json)
                indicators = [ComplianceIndicator(**i) for i in ind_list]
            except:
                pass
        current_indicators = indicators

        findings = []
        if findings_json:
            try:
                find_list = json.loads(findings_json)
                findings = [PriorFinding(**f) for f in find_list]
            except:
                pass
        current_findings = findings

        result = run_assessment(entity, indicators, findings, enable_web=enable_web)
        current_result = result

        report = format_readiness_report(result, entity)

        gaps = result.get("gaps", [])
        critical = len([g for g in gaps if g.get("risk_level") == "Critical"])
        high = len([g for g in gaps if g.get("risk_level") == "High"])
        medium = len([g for g in gaps if g.get("risk_level") == "Medium"])
        low = len([g for g in gaps if g.get("risk_level") == "Low"])

        summary = f"""### Assessment Complete

**Readiness Level:** {result.get('readiness_level', 'N/A')}
**Risk Score:** {result.get('overall_risk_score', 0):.1f} / 10

**Gaps Found:**
- Critical: {critical}
- High: {high}
- Medium: {medium}
- Low: {low}
"""

        return (
            report,
            summary,
            json.dumps(result, indent=2, ensure_ascii=False),
            result
        )

    except Exception as e:
        error_msg = f"Assessment failed: {type(e).__name__}: {e}"
        return (
            f"## Error\n\n{error_msg}",
            f"### Error\n\n{error_msg}",
            json.dumps({"error": error_msg}),
            {}
        )

def add_compliance_indicator(area, status, has_docs, has_policies, last_review, notes, current_json):
    try:
        indicators = json.loads(current_json) if current_json else []
    except:
        indicators = []

    new_indicator = {
        "area": area,
        "status": status,
        "has_documentation": has_docs,
        "has_policies": has_policies,
        "last_review_date": last_review or None,
        "notes": notes or ""
    }
    indicators.append(new_indicator)

    display = "\n".join([f"- {i['area']}: {i['status']}" for i in indicators])

    return json.dumps(indicators), display

def clear_indicators():
    return "", "_No indicators added yet_"

def add_prior_finding(category, severity, status, description, year, remediation, current_json):
    try:
        findings = json.loads(current_json) if current_json else []
    except:
        findings = []

    finding_id = f"F-{len(findings)+1:03d}"
    new_finding = {
        "finding_id": finding_id,
        "category": category,
        "severity": severity,
        "status": status,
        "description": description,
        "year_identified": int(year) if year else 0,
        "remediation_plan": remediation or "",
        "target_date": ""
    }
    findings.append(new_finding)

    display = "\n".join([f"- {f['finding_id']}: {f['category']} ({f['severity']}) - {f['status']}" for f in findings])

    return json.dumps(findings), display

def clear_findings():
    return "", "_No findings added yet_"

def download_report(result_state):
    global current_entity, current_result

    if not current_result:
        raise gr.Error("No assessment results. Please run an assessment first.")

    report = format_readiness_report(current_result, current_entity)
    entity_name = current_entity.entity_name if current_entity else "entity"
    return export_report_pdf(report, entity_name)

print("UI functions defined.")
```

---

## Cell 18 - Launch Interface

```python
# Cell 18 - Launch Interface

with gr.Blocks(title="Audit Readiness Assistant", theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
# AI-Powered Audit Readiness Assistant

This system helps assess whether an organization is ready for audit by:
1. Collecting organization information and compliance self-assessment
2. Reviewing inputs against regulatory requirements
3. Identifying compliance gaps and high-risk areas
4. Generating an audit readiness report with recommendations

**Instructions:** Complete each tab in order, then run the assessment.
""")

    indicators_state = gr.State("")
    findings_state = gr.State("")
    result_state = gr.State({})

    with gr.Tab("1. Document Setup"):
        gr.Markdown("""
### Upload Standards Documents

Upload regulatory documents, standards, or guidelines that the assessment should reference.
Demo documents are provided for testing.
""")

        uploader = gr.File(file_count="multiple", label="Upload Documents (PDF, DOCX, TXT)")
        upload_status = gr.Textbox(label="Upload Status", interactive=False)
        uploader.change(fn=ui_upload, inputs=uploader, outputs=upload_status)

        build_btn = gr.Button("Build Index", variant="primary")
        build_status = gr.Textbox(label="Index Status", interactive=False)
        build_btn.click(fn=ui_build_index, outputs=build_status)

        gr.Markdown(f"""
---
**Web Search:** {'Enabled' if TAVILY_API_KEY else 'Disabled (add TAVILY_API_KEY to enable)'}

**Demo Documents Available:** {len(initial_files)} standards documents
""")

    with gr.Tab("2. Organization Profile"):
        gr.Markdown("""
### Enter Organization Details

Provide basic information about the organization being assessed.
""")

        with gr.Row():
            entity_name = gr.Textbox(label="Organization Name", placeholder="e.g., ABC Government Department")
            entity_type = gr.Dropdown(
                label="Entity Type",
                choices=["Government Entity", "Semi-Government Entity", "Private Sector", "Non-Profit Organization"],
                value="Government Entity"
            )

        with gr.Row():
            sector = gr.Dropdown(
                label="Sector",
                choices=["Public Administration", "Healthcare", "Education", "Finance", "Infrastructure", "Technology", "Other"],
                value="Public Administration"
            )
            size = gr.Dropdown(
                label="Organization Size",
                choices=["Small (< 50 employees)", "Medium (50-250 employees)", "Large (> 250 employees)"],
                value="Medium (50-250 employees)"
            )

        with gr.Row():
            framework = gr.Dropdown(
                label="Reporting Framework",
                choices=["IFRS", "Local GAAP", "IPSAS", "US GAAP", "Other"],
                value="IFRS"
            )
            fiscal_year = gr.Textbox(label="Fiscal Year End", placeholder="e.g., December 31", value="December 31")

        with gr.Row():
            years_operation = gr.Number(label="Years in Operation", value=10)
            employees = gr.Number(label="Total Employees", value=150)

        with gr.Row():
            budget = gr.Textbox(label="Annual Budget", placeholder="e.g., AED 50 million")
            prior_rating = gr.Dropdown(
                label="Prior Audit Rating",
                choices=["Not Available", "Unqualified", "Qualified", "Adverse", "Disclaimer"],
                value="Not Available"
            )

        entity_notes = gr.Textbox(label="Additional Notes", lines=2, placeholder="Any relevant context...")

    with gr.Tab("3. Compliance Self-Assessment"):
        gr.Markdown("""
### Compliance Self-Assessment

For each compliance area, indicate your current status. Add all relevant areas.
""")

        with gr.Row():
            with gr.Column(scale=2):
                comp_area = gr.Dropdown(
                    label="Compliance Area",
                    choices=[
                        "Financial Reporting", "Internal Controls", "Asset Management",
                        "Procurement & Contracts", "HR & Payroll", "IT Systems & Security",
                        "Regulatory Compliance", "Governance & Oversight"
                    ],
                    value="Financial Reporting"
                )
                comp_status = gr.Dropdown(
                    label="Self-Assessment Status",
                    choices=["Compliant", "Partially Compliant", "Non-Compliant", "Not Yet Assessed", "Not Applicable"],
                    value="Partially Compliant"
                )
                with gr.Row():
                    has_docs = gr.Checkbox(label="Documentation Available", value=True)
                    has_policies = gr.Checkbox(label="Policies Documented", value=True)
                last_review = gr.Textbox(label="Last Review Date", placeholder="e.g., 2024-06-30")
                comp_notes = gr.Textbox(label="Notes", placeholder="Additional context...")

                with gr.Row():
                    add_indicator_btn = gr.Button("Add Indicator", variant="primary")
                    clear_indicators_btn = gr.Button("Clear All")

            with gr.Column(scale=1):
                gr.Markdown("### Added Indicators")
                indicators_display = gr.Markdown("_No indicators added yet_")

        add_indicator_btn.click(
            fn=add_compliance_indicator,
            inputs=[comp_area, comp_status, has_docs, has_policies, last_review, comp_notes, indicators_state],
            outputs=[indicators_state, indicators_display]
        )
        clear_indicators_btn.click(
            fn=clear_indicators,
            outputs=[indicators_state, indicators_display]
        )

    with gr.Tab("4. Prior Audit Findings"):
        gr.Markdown("""
### Prior Audit Findings

Enter any findings from previous audits. This helps identify recurring issues.
""")

        with gr.Row():
            with gr.Column(scale=2):
                find_category = gr.Dropdown(
                    label="Category",
                    choices=[
                        "Financial Reporting", "Internal Controls", "Asset Management",
                        "Procurement", "HR & Payroll", "IT Controls", "Governance", "Other"
                    ],
                    value="Internal Controls"
                )
                with gr.Row():
                    find_severity = gr.Dropdown(
                        label="Severity",
                        choices=["Critical", "High", "Medium", "Low"],
                        value="Medium"
                    )
                    find_status = gr.Dropdown(
                        label="Current Status",
                        choices=["Open", "In Progress", "Remediated", "Recurring"],
                        value="Open"
                    )
                find_description = gr.Textbox(
                    label="Finding Description",
                    lines=2,
                    placeholder="Describe the audit finding..."
                )
                find_year = gr.Number(label="Year Identified", value=2023)
                find_remediation = gr.Textbox(
                    label="Remediation Plan",
                    placeholder="What actions have been or will be taken?"
                )

                with gr.Row():
                    add_finding_btn = gr.Button("Add Finding", variant="primary")
                    clear_findings_btn = gr.Button("Clear All")

            with gr.Column(scale=1):
                gr.Markdown("### Added Findings")
                findings_display = gr.Markdown("_No findings added yet_")

        add_finding_btn.click(
            fn=add_prior_finding,
            inputs=[find_category, find_severity, find_status, find_description, find_year, find_remediation, findings_state],
            outputs=[findings_state, findings_display]
        )
        clear_findings_btn.click(
            fn=clear_findings,
            outputs=[findings_state, findings_display]
        )

    with gr.Tab("5. Run Assessment"):
        gr.Markdown("""
### Generate Audit Readiness Assessment

Click the button below to analyze all inputs and generate the gap analysis report.
""")

        with gr.Row():
            enable_web = gr.Checkbox(
                label="Enable Web Search",
                value=TAVILY_API_KEY is not None,
                interactive=TAVILY_API_KEY is not None
            )

        run_btn = gr.Button("Run Audit Readiness Assessment", variant="primary", size="lg")

        gr.Markdown("---")

        with gr.Row():
            with gr.Column(scale=1):
                summary_output = gr.Markdown("### Results will appear here")
                download_btn = gr.Button("Download Report (PDF)")
                download_file = gr.File(label="Download", interactive=False)

            with gr.Column(scale=2):
                report_output = gr.Markdown(label="Audit Readiness Report")

        gr.Markdown("---")

        with gr.Accordion("Raw JSON Output", open=False):
            json_output = gr.Code(language="json")

        run_btn.click(
            fn=run_full_assessment,
            inputs=[
                entity_name, entity_type, sector, size, framework, fiscal_year,
                years_operation, employees, budget, prior_rating, entity_notes,
                indicators_state, findings_state, enable_web
            ],
            outputs=[report_output, summary_output, json_output, result_state]
        )

        download_btn.click(
            fn=download_report,
            inputs=[result_state],
            outputs=[download_file]
        )

demo.launch(share=False)
```

---

## How to Use

1. Open a new Google Colab notebook
2. Copy each cell above in order (Cell 1 to Cell 18)
3. Add your API keys to Colab Secrets (key icon on left sidebar):
   - `OPEN_AI_API` - Required
   - `TAVILY_API_KEY` - Optional (for web search)
4. Run all cells
5. Use the Gradio interface:
   - Tab 1: Build the document index
   - Tab 2: Enter organization details
   - Tab 3: Add compliance self-assessment indicators
   - Tab 4: Add prior audit findings (if any)
   - Tab 5: Run assessment and view results

---

## System Workflow

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  ENTITY INTAKE  │────>│ STANDARDS LOAD  │────>│  GAP ANALYSIS   │
│                 │     │                 │     │                 │
│ - Org details   │     │ - IFRS rules    │     │ - Compare       │
│ - Sector/size   │     │ - ADAA guides   │     │ - Find gaps     │
│ - Framework     │     │ - Controls      │     │ - Score risk    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         │                                               │
         v                                               v
┌─────────────────┐                             ┌─────────────────┐
│ COMPLIANCE INFO │                             │ READINESS REPORT│
│                 │                             │                 │
│ - Self-assess   │                             │ - Risk summary  │
│ - Prior findings│                             │ - Gap details   │
│ - Evidence      │                             │ - Actions needed│
└─────────────────┘                             └─────────────────┘
```

---

**Capstone Project by: Abdulla Ahmed Alaydaroos**
