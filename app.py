"""
ResearchMind AI -- Intelligent Research Companion
Powered by IBM watsonx.ai Studio & IBM Granite Models
Agentic AI Architecture | Multi-Agent Collaboration | RAG System

Architecture Overview
---------------------
  Master Orchestrator
    |-- Research Retrieval Agent   (Agent 1)
    |-- Literature Review Agent    (Agent 2)
    |-- Gap Analysis Agent         (Agent 3)
    |-- Trend Forecasting Agent    (Agent 4)
    `-- Research Advisor Agent     (Agent 5)

RAG Pipeline
------------
  Upload PDF/TXT -> Extract Text -> Chunk -> TF-IDF Embeddings -> Retrieve -> Granite

IBM Granite Models Used
-----------------------
  Primary:  ibm/granite-3-3-8b-instruct
  Fallback: ibm/granite-13b-chat-v2

Dependencies (install via pip)
------------------------------
  pip install flask ibm-watsonx-ai PyPDF2 scikit-learn numpy python-dotenv
"""

# ─── Standard library ───────────────────────────────────────────────────────
import os
import re
import json
import math
import time
import uuid
import textwrap
from io import BytesIO
from collections import defaultdict

# ─── Load .env file (if python-dotenv is installed) ─────────────────────────
# Copy .env.example → .env and fill in your IBM watsonx.ai credentials.
# If python-dotenv is not installed the app falls back to real env variables
# or DEMO mode — nothing breaks.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed; rely on environment variables directly

# ─── Flask ───────────────────────────────────────────────────────────────────
from flask import (
    Flask, request, jsonify, render_template, session
)

# ─── IBM watsonx.ai SDK ──────────────────────────────────────────────────────
try:
    from ibm_watsonx_ai import Credentials
    from ibm_watsonx_ai.foundation_models import ModelInference
    from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
    WATSONX_AVAILABLE = True
except ImportError:
    WATSONX_AVAILABLE = False
    print("[WARN] ibm-watsonx-ai not installed. Running in DEMO mode.")

# ─── PDF extraction ──────────────────────────────────────────────────────────
try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False

# ─── Lightweight RAG (TF-IDF) ────────────────────────────────────────────────
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("[WARN] scikit-learn not installed. RAG retrieval disabled.")

# ─────────────────────────────────────────────────────────────────────────────
#  Flask App Initialisation
# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)
# Use SECRET_KEY from .env if set, otherwise fall back to a random value
app.secret_key = os.getenv("FLASK_SECRET_KEY") or os.urandom(24)

# ─────────────────────────────────────────────────────────────────────────────
#  IBM watsonx.ai Configuration
#  Set these environment variables before running:
#    export WATSONX_API_KEY="your-api-key"
#    export WATSONX_PROJECT_ID="your-project-id"
#    export WATSONX_URL="https://us-south.ml.cloud.ibm.com"
# ─────────────────────────────────────────────────────────────────────────────
WATSONX_API_KEY    = os.getenv("WATSONX_API_KEY",    "your-api-key-here")
WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID", "your-project-id-here")
WATSONX_URL        = os.getenv("WATSONX_URL",        "https://us-south.ml.cloud.ibm.com")

# IBM Granite model selection – overridable via .env PRIMARY_MODEL / FALLBACK_MODEL
PRIMARY_MODEL  = os.getenv("PRIMARY_MODEL",  "ibm/granite-3-3-8b-instruct")
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "ibm/granite-13b-chat-v2")

# In-memory knowledge base (replaces a database)
# Structure: { session_id: { "chunks": [...], "vectorizer": ..., "matrix": ... } }
KNOWLEDGE_BASE = {}

# In-memory agent results store
AGENT_RESULTS = {}

# =============================================================================
#  SECTION 1: IBM watsonx.ai Helper – generate_response()
#  All five agents call this single function; it hides SDK details.
# =============================================================================

def get_watsonx_model():
    """
    Instantiate and return an IBM Granite ModelInference client.
    Falls back to DEMO mode if credentials are missing or the SDK is absent.
    """
    if not WATSONX_AVAILABLE:
        return None
    if WATSONX_API_KEY == "your-api-key-here":
        return None   # Demo mode – credentials not configured

    try:
        credentials = Credentials(  # type: ignore[misc]
            url=WATSONX_URL,
            api_key=WATSONX_API_KEY
        )
        # ── IBM watsonx.ai integration point ──────────────────────────────
        # ModelInference wraps the Granite REST endpoint.
        # GenParams controls generation behaviour.
        model = ModelInference(  # type: ignore[misc]
            model_id=PRIMARY_MODEL,
            credentials=credentials,
            project_id=WATSONX_PROJECT_ID,
            params={
                GenParams.MAX_NEW_TOKENS: 1024,  # type: ignore[union-attr]
                GenParams.MIN_NEW_TOKENS: 50,  # type: ignore[union-attr]
                GenParams.TEMPERATURE:    0.7,  # type: ignore[union-attr]
                GenParams.TOP_P:          0.95,  # type: ignore[union-attr]
                GenParams.TOP_K:          50,  # type: ignore[union-attr]
                GenParams.REPETITION_PENALTY: 1.1,  # type: ignore[union-attr]
            }
        )
        return model
    except Exception as exc:
        print(f"[ERROR] watsonx.ai model init failed: {exc}")
        return None


def generate_response(prompt: str, context: str = "", agent_name: str = "Agent") -> str:
    """
    Core IBM Granite inference function used by every agent.

    Args:
        prompt  : The instruction / question for the Granite model.
        context : Optional RAG-retrieved passages to ground the answer.
        agent_name: Label used in demo-mode responses.

    Returns:
        A string response from IBM Granite (or a demo placeholder).
    """
    model = get_watsonx_model()

    # Build the full prompt with optional RAG context
    full_prompt = ""
    if context:
        full_prompt += f"[Research Context from Knowledge Base]\n{context}\n\n"
    full_prompt += f"[Task for {agent_name}]\n{prompt}\n\nResponse:"

    if model:
        try:
            # ── IBM watsonx.ai integration point ──────────────────────────
            result = model.generate_text(prompt=full_prompt)
            return result.strip() if isinstance(result, str) and result else _demo_response(agent_name, prompt)
        except Exception as exc:
            print(f"[ERROR] Granite generation failed: {exc}")
            return _demo_response(agent_name, prompt)
    else:
        # Demo mode – returns structured placeholder output so the UI works
        return _demo_response(agent_name, prompt)


def _demo_response(agent_name: str, prompt: str) -> str:
    """
    Returns realistic-looking demo output when watsonx.ai credentials
    are not configured.  The UI is fully functional in this mode.
    """
    demos = {
        "Research Retrieval Agent": (
            "**Research Summary**\n"
            "The uploaded research material covers recent advances in the specified domain. "
            "Key themes include methodology innovation, empirical validation, and cross-disciplinary applications.\n\n"
            "**Key Findings**\n"
            "1. Significant performance improvements over prior baselines (~18 % average gain).\n"
            "2. Scalability demonstrated on datasets exceeding 1M samples.\n"
            "3. Transfer-learning approaches reduce labelled-data requirements by 40 %.\n\n"
            "**Important References**\n"
            "- Smith et al. (2023) – Foundational framework\n"
            "- Johnson & Lee (2024) – Benchmark evaluation\n"
            "- Kumar et al. (2024) – Applied extensions\n\n"
            "*(Demo mode – connect IBM watsonx.ai credentials for live Granite inference)*"
        ),
        "Literature Review Agent": (
            "**Literature Review**\n"
            "The body of work in this area spans approximately 15 years of active research. "
            "Early studies (2010–2018) focused on theoretical foundations, while more recent work "
            "emphasises practical deployment and ethical considerations.\n\n"
            "**Key Contributions**\n"
            "• Establishment of standardised benchmarks (2019)\n"
            "• Introduction of transformer-based architectures (2020)\n"
            "• Multimodal integration techniques (2022–2024)\n\n"
            "**Research Landscape Overview**\n"
            "Consensus exists around core evaluation metrics, but methodological disagreements "
            "remain regarding data augmentation strategies and fairness constraints.\n\n"
            "*(Demo mode – connect IBM watsonx.ai credentials for live Granite inference)*"
        ),
        "Research Gap Analysis Agent": (
            "**Identified Research Gaps**\n"
            "1. Lack of longitudinal studies examining long-term model behaviour.\n"
            "2. Underrepresentation of low-resource languages and edge-case demographics.\n"
            "3. Insufficient adversarial robustness testing under real-world conditions.\n\n"
            "**Novel Research Opportunities**\n"
            "• Privacy-preserving federated learning for sensitive domains\n"
            "• Causal inference integration to move beyond correlation\n"
            "• Green-AI optimisation to reduce carbon footprint\n\n"
            "**Improvement Suggestions**\n"
            "Adopt open-source reproducible pipelines, diversify benchmark datasets, "
            "and establish cross-institution validation protocols.\n\n"
            "*(Demo mode – connect IBM watsonx.ai credentials for live Granite inference)*"
        ),
        "Trend Forecasting Agent": (
            "**Future Research Areas (2025–2030)**\n"
            "1. Neuro-symbolic AI – blending neural networks with symbolic reasoning\n"
            "2. AI-powered climate modelling and carbon-capture optimisation\n"
            "3. Quantum machine learning for drug discovery\n\n"
            "**Emerging Trends**\n"
            "• Multimodal foundation models (text + vision + audio)\n"
            "• On-device edge AI with sub-1B parameter models\n"
            "• Explainable AI mandated by regulatory frameworks\n\n"
            "**Growth Potential**\n"
            "Estimated 34 % CAGR in AI research publications through 2028. "
            "Healthcare, climate science, and materials science show the highest growth vectors.\n\n"
            "*(Demo mode – connect IBM watsonx.ai credentials for live Granite inference)*"
        ),
        "Research Advisor Agent": (
            "**Suggested Research Questions**\n"
            "1. How can we quantify epistemic uncertainty in large language models?\n"
            "2. What governance frameworks best balance innovation and safety?\n"
            "3. Can transfer learning eliminate the need for domain-specific labelled data?\n\n"
            "**Potential Methodologies**\n"
            "• Systematic literature review + meta-analysis\n"
            "• Mixed-methods: quantitative benchmarking + qualitative expert interviews\n"
            "• Randomised controlled trials for A/B evaluation\n\n"
            "**Dataset Recommendations**\n"
            "• HuggingFace Datasets Hub (open-access)\n"
            "• UCI Machine Learning Repository\n"
            "• IEEE DataPort\n\n"
            "**Publication Venues**\n"
            "NeurIPS, ICML, ACL, ICLR, Nature Machine Intelligence\n\n"
            "**Actionable Research Plan**\n"
            "Phase 1 (Months 1–3): Literature survey & gap validation\n"
            "Phase 2 (Months 4–8): Prototype development & baseline experiments\n"
            "Phase 3 (Months 9–12): Full evaluation, ablation studies, paper writing\n\n"
            "*(Demo mode – connect IBM watsonx.ai credentials for live Granite inference)*"
        ),
    }
    return demos.get(agent_name, f"*(Demo response from {agent_name} – connect IBM watsonx.ai for live Granite inference)*")


# =============================================================================
#  SECTION 2: Lightweight RAG System
#  Upload → Extract → Chunk → TF-IDF index → Retrieve top-k passages
# =============================================================================

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract plain text from a PDF byte stream using PyPDF2."""
    if not PYPDF2_AVAILABLE:
        return ""
    try:
        reader = PyPDF2.PdfReader(BytesIO(file_bytes))  # type: ignore[union-attr]
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)
    except Exception as exc:
        print(f"[ERROR] PDF extraction: {exc}")
        return ""


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> list:
    """
    Split text into overlapping chunks for granular retrieval.

    Args:
        text       : Full document text.
        chunk_size : Approximate words per chunk.
        overlap    : Words shared between consecutive chunks.

    Returns:
        List of text chunk strings.
    """
    words  = text.split()
    chunks = []
    start  = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return [c for c in chunks if len(c.strip()) > 30]


def build_rag_index(session_id: str, new_chunks: list):
    """
    Add chunks to the in-memory TF-IDF index for a session.
    Rebuilds the vectoriser after each upload.
    """
    if not SKLEARN_AVAILABLE:
        return

    kb = KNOWLEDGE_BASE.setdefault(session_id, {"chunks": []})
    kb["chunks"].extend(new_chunks)

    if len(kb["chunks"]) < 2:
        return  # Need at least 2 documents for TF-IDF

    # ── RAG integration point ─────────────────────────────────────────────
    # TfidfVectorizer converts chunks to sparse TF-IDF vectors.
    # cosine_similarity ranks chunks against a query at retrieval time.
    vectorizer = TfidfVectorizer(  # type: ignore[misc]
        max_features=5000,
        stop_words="english",
        ngram_range=(1, 2)
    )
    matrix = vectorizer.fit_transform(kb["chunks"])
    kb["vectorizer"] = vectorizer
    kb["matrix"]     = matrix


def retrieve_relevant_chunks(session_id: str, query: str, top_k: int = 5) -> str:
    """
    Retrieve the top-k most relevant passages from the knowledge base
    using TF-IDF cosine similarity.

    Returns a single string of concatenated passages (the RAG context).
    """
    if not SKLEARN_AVAILABLE:
        return ""

    kb = KNOWLEDGE_BASE.get(session_id, {})
    if not kb.get("vectorizer"):
        return ""

    try:
        query_vec   = kb["vectorizer"].transform([query])
        similarities = cosine_similarity(query_vec, kb["matrix"]).flatten()  # type: ignore[misc]
        top_indices = similarities.argsort()[-top_k:][::-1]
        passages    = [kb["chunks"][i] for i in top_indices if similarities[i] > 0.01]
        return "\n\n---\n\n".join(passages[:top_k])
    except Exception as exc:
        print(f"[ERROR] RAG retrieval: {exc}")
        return ""


# =============================================================================
#  SECTION 3: Five Specialised AI Agents
#  Each agent has a focused prompt strategy and calls generate_response().
# =============================================================================

# ── Agent 1: Research Retrieval Agent ────────────────────────────────────────
def retrieval_agent(query: str, session_id: str) -> dict:
    """
    Agent 1 – Research Retrieval Agent
    ───────────────────────────────────
    Retrieves relevant knowledge from the RAG index and produces:
      • Research Summary
      • Key Findings
      • Important References

    Uses IBM Granite for summarisation and context extraction.
    """
    # Step 1: Pull relevant passages from the RAG knowledge base
    context = retrieve_relevant_chunks(session_id, query, top_k=5)

    # Step 2: Build a specialised prompt for the retrieval task
    prompt = textwrap.dedent(f"""
        You are a Research Retrieval Agent specialised in academic literature analysis.
        Your task is to synthesise the available research material and answer the query below.

        Research Query: {query}

        Please provide:
        1. **Research Summary** – A concise overview of what the research material covers.
        2. **Key Findings** – The 3–5 most important empirical or theoretical findings.
        3. **Important References** – Notable authors, papers, or datasets mentioned.

        Be precise, academic in tone, and cite evidence from the context where possible.
    """).strip()

    # Step 3: Call IBM Granite via generate_response()
    response = generate_response(prompt, context, "Research Retrieval Agent")

    return {
        "agent":    "Research Retrieval Agent",
        "icon":     "🔍",
        "color":    "#3b82f6",
        "query":    query,
        "context_used": bool(context),
        "output":   response,
        "timestamp": time.strftime("%H:%M:%S"),
        "why_activated": "Activated to retrieve and organise relevant research knowledge from uploaded materials and the knowledge base.",
    }


# ── Agent 2: Literature Review Agent ─────────────────────────────────────────
def literature_review_agent(query: str, retrieval_output: str, session_id: str) -> dict:
    """
    Agent 2 – Literature Review Agent
    ───────────────────────────────────
    Builds a structured literature review from Agent 1's output and the
    RAG knowledge base. Produces:
      • Literature Review narrative
      • Key Contributions
      • Research Landscape Overview

    Uses IBM Granite for comparative synthesis.
    """
    context = retrieve_relevant_chunks(session_id, query, top_k=5)

    prompt = textwrap.dedent(f"""
        You are a Literature Review Agent with expertise in academic synthesis.
        Using the research retrieval output and additional context, produce a comprehensive literature review.

        Research Topic: {query}

        Research Retrieval Summary:
        {retrieval_output[:800]}

        Please provide:
        1. **Literature Review** – A flowing narrative comparing key works, identifying common themes,
           agreements, and disagreements across the literature.
        2. **Key Contributions** – Bullet list of the most impactful contributions in this field.
        3. **Research Landscape Overview** – How the field has evolved and its current state.

        Write in an academic, structured style suitable for a journal submission.
    """).strip()

    response = generate_response(prompt, context, "Literature Review Agent")

    return {
        "agent":    "Literature Review Agent",
        "icon":     "📚",
        "color":    "#8b5cf6",
        "query":    query,
        "context_used": bool(context),
        "output":   response,
        "timestamp": time.strftime("%H:%M:%S"),
        "why_activated": "Activated to synthesise multiple research sources into a structured, comparative literature review.",
    }


# ── Agent 3: Research Gap Analysis Agent ─────────────────────────────────────
def gap_analysis_agent(query: str, lit_review_output: str, session_id: str) -> dict:
    """
    Agent 3 – Research Gap Analysis Agent
    ──────────────────────────────────────
    Identifies missing opportunities and limitations in existing research.
    Produces:
      • Research Gaps
      • Novel Research Opportunities
      • Improvement Suggestions

    Uses IBM Granite for critical gap detection.
    """
    context = retrieve_relevant_chunks(session_id, query, top_k=4)

    prompt = textwrap.dedent(f"""
        You are a Research Gap Analysis Agent.  Your role is to critically examine existing research
        and identify what is missing, underexplored, or methodologically weak.

        Research Topic: {query}

        Literature Review Summary:
        {lit_review_output[:800]}

        Please provide:
        1. **Research Gaps** – Specific unanswered questions or missing investigations (list 4–6 items).
        2. **Novel Research Opportunities** – Concrete new directions enabled by closing these gaps.
        3. **Improvement Suggestions** – How existing methodologies or datasets could be strengthened.

        Be critical, specific, and constructive. Prioritise gaps that are both important and feasible.
    """).strip()

    response = generate_response(prompt, context, "Research Gap Analysis Agent")

    return {
        "agent":    "Research Gap Analysis Agent",
        "icon":     "🔬",
        "color":    "#ef4444",
        "query":    query,
        "context_used": bool(context),
        "output":   response,
        "timestamp": time.strftime("%H:%M:%S"),
        "why_activated": "Activated to detect unanswered questions, limitations, and unexplored areas in current research.",
    }


# ── Agent 4: Trend Forecasting Agent ─────────────────────────────────────────
def trend_forecasting_agent(query: str, gap_output: str) -> dict:
    """
    Agent 4 – Trend Forecasting Agent
    ──────────────────────────────────
    Predicts future research directions based on gap analysis and
    broader technology trends. Produces:
      • Future Research Areas
      • Emerging Trends
      • Growth Potential

    Uses IBM Granite for forward-looking synthesis.
    """
    prompt = textwrap.dedent(f"""
        You are a Trend Forecasting Agent specialised in research and technology foresight.
        Based on identified research gaps and the current state of the field, forecast future directions.

        Research Topic: {query}

        Identified Gaps & Opportunities:
        {gap_output[:700]}

        Please provide:
        1. **Future Research Areas (next 3–5 years)** – List the top emerging research directions
           with brief justification for each.
        2. **Emerging Trends** – Technological, methodological, or societal trends that will shape
           this field (e.g. AI-powered diagnostics, quantum ML, sustainable optimisation).
        3. **Growth Potential** – Estimate qualitative growth trajectory and which sub-fields
           will attract the most attention and funding.

        Be forward-looking, evidence-based, and bold in your forecasts.
    """).strip()

    response = generate_response(prompt, "", "Trend Forecasting Agent")

    return {
        "agent":    "Trend Forecasting Agent",
        "icon":     "📈",
        "color":    "#f59e0b",
        "query":    query,
        "context_used": False,
        "output":   response,
        "timestamp": time.strftime("%H:%M:%S"),
        "why_activated": "Activated to analyse emerging trends, publication trajectories, and predict future research directions.",
    }


# ── Agent 5: Research Advisor Agent ──────────────────────────────────────────
def research_advisor_agent(query: str, all_prior_outputs: str) -> dict:
    """
    Agent 5 – Research Advisor Agent
    ─────────────────────────────────
    Synthesises all prior agent outputs into a concrete, actionable
    research plan. Produces:
      • Suggested Research Questions
      • Potential Methodologies
      • Dataset Recommendations
      • Publication Recommendations
      • Thesis / Project Ideas
      • Actionable Research Plan

    Uses IBM Granite for strategic synthesis.
    """
    prompt = textwrap.dedent(f"""
        You are a Research Advisor Agent – a senior academic mentor providing strategic guidance.
        Based on the full research analysis below, create a concrete, actionable research plan
        for a researcher working in this area.

        Research Topic: {query}

        Full Research Analysis Summary:
        {all_prior_outputs[:1000]}

        Please provide:
        1. **Suggested Research Questions** – 3 specific, novel, answerable research questions.
        2. **Potential Methodologies** – Best-fit research designs and methods with justification.
        3. **Dataset Recommendations** – Named open-access datasets suitable for this research.
        4. **Publication Recommendations** – Top venues (conferences/journals) to target.
        5. **Thesis / Project Ideas** – 2 concrete thesis titles or project concepts.
        6. **Actionable Research Plan** – A phased timeline (Phase 1 / 2 / 3) with deliverables.

        Write as a helpful, encouraging senior mentor. Be specific and practical.
    """).strip()

    response = generate_response(prompt, "", "Research Advisor Agent")

    return {
        "agent":    "Research Advisor Agent",
        "icon":     "🎯",
        "color":    "#10b981",
        "query":    query,
        "context_used": False,
        "output":   response,
        "timestamp": time.strftime("%H:%M:%S"),
        "why_activated": "Activated to synthesise all findings into a strategic, actionable research plan for the researcher.",
    }


# =============================================================================
#  SECTION 4: Master Orchestrator Agent
#  The brain of the system – coordinates all five agents in sequence.
# =============================================================================

def orchestrator_agent(query: str, session_id: str) -> dict:
    """
    Master Orchestrator Agent
    ─────────────────────────
    Workflow:
      1. Research Retrieval Agent   → extract knowledge
      2. Literature Review Agent    → synthesise literature
      3. Gap Analysis Agent         → identify gaps
      4. Trend Forecasting Agent    → predict future directions
      5. Research Advisor Agent     → build actionable plan
      6. Final Report generation    → compile everything

    All inter-agent communication is explicit; each agent receives the
    output of its predecessor(s) to maintain a coherent reasoning chain.
    """
    results = {
        "session_id":    session_id,
        "query":         query,
        "agents":        [],
        "final_report":  "",
        "knowledge_graph": {},
        "timestamp":     time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # ── Step 1: Research Retrieval ───────────────────────────────────────
    r1 = retrieval_agent(query, session_id)
    results["agents"].append(r1)

    # ── Step 2: Literature Review (uses Agent 1 output) ──────────────────
    r2 = literature_review_agent(query, r1["output"], session_id)
    results["agents"].append(r2)

    # ── Step 3: Gap Analysis (uses Agent 2 output) ───────────────────────
    r3 = gap_analysis_agent(query, r2["output"], session_id)
    results["agents"].append(r3)

    # ── Step 4: Trend Forecasting (uses Agent 3 output) ──────────────────
    r4 = trend_forecasting_agent(query, r3["output"])
    results["agents"].append(r4)

    # ── Step 5: Research Advisor (uses all prior outputs) ─────────────────
    all_prior = "\n\n".join([
        f"[{r['agent']}]\n{r['output']}"
        for r in results["agents"]
    ])
    r5 = research_advisor_agent(query, all_prior)
    results["agents"].append(r5)

    # ── Step 6: Final Report compilation ─────────────────────────────────
    final_prompt = textwrap.dedent(f"""
        You are the Master Orchestrator of a multi-agent research intelligence system.
        Synthesise the outputs of all five specialised agents into a single, coherent,
        executive-level Research Intelligence Report.

        Research Topic: {query}

        Agent Outputs:
        {all_prior[:2000]}

        Generate a Final Research Intelligence Report with these sections:
        ## Executive Summary
        ## Key Research Insights
        ## Critical Gaps & Opportunities
        ## Future Outlook
        ## Recommended Next Steps

        Write clearly, concisely, and with authority. Suitable for a research director or funding body.
    """).strip()

    results["final_report"] = generate_response(final_prompt, "", "Master Orchestrator")

    # ── Step 7: Build Knowledge Graph data (topics, concepts, relations) ──
    results["knowledge_graph"] = _build_knowledge_graph(query, results["agents"])

    # Cache results for the session
    AGENT_RESULTS[session_id] = results
    return results


def _build_knowledge_graph(query: str, agent_results: list) -> dict:
    """
    Build a simple knowledge graph from agent outputs.
    Returns nodes and edges suitable for front-end D3/SVG rendering.
    """
    nodes = [{"id": "query", "label": query[:40], "type": "query", "size": 30}]
    edges = []

    agent_ids = []
    for i, agent in enumerate(agent_results):
        aid = f"agent_{i}"
        agent_ids.append(aid)
        nodes.append({
            "id":    aid,
            "label": agent["agent"].replace(" Agent", ""),
            "type":  "agent",
            "color": agent["color"],
            "size":  22,
        })
        edges.append({"source": "query", "target": aid, "label": "activates"})

    # Extract key concept nodes from outputs (simple keyword heuristic)
    all_text = " ".join(a["output"] for a in agent_results)
    concept_words = _extract_concepts(all_text, n=8)
    for j, concept in enumerate(concept_words):
        cid = f"concept_{j}"
        nodes.append({"id": cid, "label": concept, "type": "concept", "size": 14})
        # Connect to the most relevant agent
        target_agent = agent_ids[j % len(agent_ids)]
        edges.append({"source": target_agent, "target": cid, "label": "generates"})

    return {"nodes": nodes, "edges": edges}


def _extract_concepts(text: str, n: int = 8) -> list:
    """
    Extract the top-n meaningful noun phrases / keywords from text.
    Uses a simple frequency heuristic (no NLP library required).
    """
    stopwords = {
        "the","a","an","is","are","was","were","be","been","being","have","has",
        "had","do","does","did","will","would","could","should","may","might",
        "must","can","this","that","these","those","of","in","on","at","to","for",
        "with","by","from","as","or","and","but","not","it","its","their","our",
        "your","my","we","they","he","she","i","you","research","study","paper",
        "results","data","analysis","based","using","used","also","more","such",
        "both","each","other","new","use","work","provide","provide","include",
        "model","models","approach","method","methods","system","systems","agent",
        "agents","output","outputs","into","within","between","across","through",
    }
    words = re.findall(r'\b[A-Za-z][a-z]{3,}\b', text)
    freq  = defaultdict(int)
    for w in words:
        if w.lower() not in stopwords:
            freq[w.lower()] += 1
    top = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [w.title() for w, _ in top[:n]]

# =============================================================================
#  SECTION 5: Flask Routes
# =============================================================================

@app.route("/")
def index():
    """Serve the single-page Research Intelligence Dashboard."""
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    return render_template("index.html")


@app.route("/api/upload", methods=["POST"])
def upload_document():
    """
    Handle document uploads (PDF / TXT).
    Extracts text, chunks it, and builds the RAG TF-IDF index for the session.
    """
    sid = session.get("session_id", str(uuid.uuid4()))

    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file provided"}), 400

    file = request.files["file"]
    filename = file.filename or "unknown"
    file_bytes = file.read()

    # ── Extract text ──────────────────────────────────────────────────────
    if filename.lower().endswith(".pdf"):
        text = extract_text_from_pdf(file_bytes)
    elif filename.lower().endswith(".txt"):
        text = file_bytes.decode("utf-8", errors="ignore")
    else:
        return jsonify({"success": False, "error": "Unsupported file type. Use PDF or TXT."}), 400

    if not text.strip():
        return jsonify({"success": False, "error": "Could not extract text from the document."}), 400

    # ── Chunk and index ───────────────────────────────────────────────────
    chunks = chunk_text(text)
    build_rag_index(sid, chunks)

    kb = KNOWLEDGE_BASE.get(sid, {})
    return jsonify({
        "success":      True,
        "filename":     filename,
        "chunks_added": len(chunks),
        "total_chunks": len(kb.get("chunks", [])),
        "preview":      text[:300] + "…" if len(text) > 300 else text,
        "rag_enabled":  SKLEARN_AVAILABLE,
    })


@app.route("/api/research", methods=["POST"])
def run_research():
    """
    Main research endpoint – triggers the Master Orchestrator.
    Accepts a JSON body: { "query": "..." }
    Returns full agent outputs, final report, and knowledge graph data.
    """
    sid = session.get("session_id", str(uuid.uuid4()))
    data = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()

    if not query:
        return jsonify({"success": False, "error": "Please provide a research query."}), 400

    try:
        results = orchestrator_agent(query, sid)
        return jsonify({"success": True, "results": results})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/status", methods=["GET"])
def system_status():
    """Return current system configuration and capability flags."""
    sid = session.get("session_id", "")
    kb  = KNOWLEDGE_BASE.get(sid, {})
    return jsonify({
        "watsonx_available":   WATSONX_AVAILABLE,
        "watsonx_configured":  WATSONX_API_KEY != "your-api-key-here",
        "rag_available":       SKLEARN_AVAILABLE,
        "pdf_available":       PYPDF2_AVAILABLE,
        "primary_model":       PRIMARY_MODEL,
        "fallback_model":      FALLBACK_MODEL,
        "knowledge_base_chunks": len(kb.get("chunks", [])),
        "mode":                "LIVE (IBM Granite)" if (WATSONX_AVAILABLE and WATSONX_API_KEY != "your-api-key-here") else "DEMO",
    })


@app.route("/api/clear", methods=["POST"])
def clear_session():
    """Clear the knowledge base for the current session."""
    sid = session.get("session_id", "")
    KNOWLEDGE_BASE.pop(sid, None)
    AGENT_RESULTS.pop(sid, None)
    session["session_id"] = str(uuid.uuid4())
    return jsonify({"success": True, "message": "Session cleared."})


# =============================================================================
#  SECTION 6: HTML Template
#  Moved to templates/index.html  (served via render_template)
#  Styles moved to static/style.css
# =============================================================================

# HTML_TEMPLATE removed — see templates/index.html

_SENTINEL = r"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>ResearchMind AI – Intelligent Research Companion</title>
  <!-- Bootstrap 5 CDN -->
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet"/>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css" rel="stylesheet"/>
  <style>
    /* ── Base ─────────────────────────────────────────────────────────── */
    :root {
      --ibm-blue:    #0f62fe;
      --ibm-dark:    #161616;
      --ibm-gray:    #393939;
      --ibm-light:   #f4f4f4;
      --ibm-border:  #e0e0e0;
      --accent-blue: #3b82f6;
      --accent-purple:#8b5cf6;
      --accent-red:  #ef4444;
      --accent-amber:#f59e0b;
      --accent-green:#10b981;
    }
    * { box-sizing: border-box; }
    body {
      background: #0a0a14;
      color: #e2e8f0;
      font-family: 'IBM Plex Sans', 'Segoe UI', system-ui, sans-serif;
      min-height: 100vh;
    }

    /* ── Navbar ───────────────────────────────────────────────────────── */
    .navbar-brand-title { font-size: 1.25rem; font-weight: 700; color: #fff; }
    .navbar-brand-sub   { font-size: 0.7rem;  color: #a0aec0; letter-spacing: .08em; text-transform: uppercase; }
    .ibm-badge {
      background: var(--ibm-blue);
      color: #fff;
      font-size: 0.65rem;
      padding: 2px 8px;
      border-radius: 3px;
      font-weight: 600;
      letter-spacing: .05em;
    }

    /* ── Cards ────────────────────────────────────────────────────────── */
    .card-dark {
      background: #12121f;
      border: 1px solid #2d2d4e;
      border-radius: 12px;
    }
    .card-dark .card-header {
      background: transparent;
      border-bottom: 1px solid #2d2d4e;
      padding: 1rem 1.25rem;
    }

    /* ── Sidebar ──────────────────────────────────────────────────────── */
    .sidebar { position: sticky; top: 80px; }

    /* ── Status badge ─────────────────────────────────────────────────── */
    .status-dot {
      width: 8px; height: 8px;
      border-radius: 50%;
      display: inline-block;
      margin-right: 6px;
    }
    .status-live { background: #10b981; box-shadow: 0 0 6px #10b981; }
    .status-demo { background: #f59e0b; box-shadow: 0 0 6px #f59e0b; }

    /* ── Agent cards ──────────────────────────────────────────────────── */
    .agent-card {
      border-left: 3px solid var(--accent-blue);
      background: #0d0d1a;
      border-radius: 8px;
      padding: 1rem;
      margin-bottom: 1rem;
      transition: transform .2s, box-shadow .2s;
    }
    .agent-card:hover { transform: translateX(4px); box-shadow: 0 0 18px rgba(59,130,246,.15); }
    .agent-card.agent-1 { border-color: #3b82f6; }
    .agent-card.agent-2 { border-color: #8b5cf6; }
    .agent-card.agent-3 { border-color: #ef4444; }
    .agent-card.agent-4 { border-color: #f59e0b; }
    .agent-card.agent-5 { border-color: #10b981; }

    .agent-icon {
      width: 40px; height: 40px;
      border-radius: 10px;
      display: flex; align-items: center; justify-content: center;
      font-size: 1.3rem;
      flex-shrink: 0;
    }

    .agent-output-body {
      background: #0a0a14;
      border: 1px solid #2d2d4e;
      border-radius: 6px;
      padding: .75rem 1rem;
      font-size: .85rem;
      line-height: 1.65;
      white-space: pre-wrap;
      max-height: 320px;
      overflow-y: auto;
      color: #cbd5e1;
    }

    /* ── Progress flow ────────────────────────────────────────────────── */
    .flow-step {
      display: flex;
      align-items: center;
      padding: .5rem .75rem;
      border-radius: 8px;
      margin-bottom: .5rem;
      background: #0d0d1a;
      border: 1px solid #2d2d4e;
      font-size: .8rem;
      gap: .6rem;
      opacity: .45;
      transition: opacity .4s, border-color .4s;
    }
    .flow-step.active  { opacity: 1; border-color: var(--ibm-blue); }
    .flow-step.done    { opacity: 1; border-color: #10b981; }
    .flow-step .step-num {
      width: 22px; height: 22px;
      border-radius: 50%;
      background: #1e1e3a;
      display: flex; align-items: center; justify-content: center;
      font-size: .7rem; font-weight: 700; flex-shrink: 0;
    }

    /* ── Knowledge graph canvas ───────────────────────────────────────── */
    #kgCanvas {
      width: 100%; height: 380px;
      background: #08080f;
      border-radius: 8px;
      border: 1px solid #2d2d4e;
    }

    /* ── Upload zone ──────────────────────────────────────────────────── */
    .upload-zone {
      border: 2px dashed #2d2d4e;
      border-radius: 10px;
      padding: 1.5rem;
      text-align: center;
      cursor: pointer;
      transition: border-color .2s, background .2s;
    }
    .upload-zone:hover, .upload-zone.drag-over {
      border-color: var(--ibm-blue);
      background: rgba(15,98,254,.06);
    }

    /* ── Query input ──────────────────────────────────────────────────── */
    .query-input {
      background: #0d0d1a;
      border: 1px solid #2d2d4e;
      border-radius: 8px;
      color: #e2e8f0;
      padding: .75rem 1rem;
      font-size: .95rem;
      resize: vertical;
      width: 100%;
      transition: border-color .2s;
    }
    .query-input:focus {
      outline: none;
      border-color: var(--ibm-blue);
      box-shadow: 0 0 0 3px rgba(15,98,254,.2);
    }

    /* ── Buttons ──────────────────────────────────────────────────────── */
    .btn-research {
      background: linear-gradient(135deg, #0f62fe, #6929c4);
      border: none;
      color: #fff;
      font-weight: 600;
      padding: .65rem 1.5rem;
      border-radius: 8px;
      letter-spacing: .03em;
      transition: opacity .2s, transform .1s;
    }
    .btn-research:hover   { opacity: .9; transform: translateY(-1px); color: #fff; }
    .btn-research:active  { transform: translateY(0); }
    .btn-research:disabled { opacity: .5; cursor: not-allowed; }

    /* ── Final report ─────────────────────────────────────────────────── */
    .final-report {
      background: linear-gradient(135deg, #0d0d2b 0%, #0a1628 100%);
      border: 1px solid #1e3a5f;
      border-radius: 10px;
      padding: 1.5rem;
      white-space: pre-wrap;
      font-size: .88rem;
      line-height: 1.7;
      color: #cbd5e1;
    }

    /* ── Section headers ──────────────────────────────────────────────── */
    .section-header {
      font-size: .65rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: .12em;
      color: #64748b;
      margin-bottom: .75rem;
    }

    /* ── Spinner ──────────────────────────────────────────────────────── */
    .spinner-pulse {
      animation: pulse 1.5s ease-in-out infinite;
    }
    @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: .3; } }

    /* ── Scrollbar ────────────────────────────────────────────────────── */
    ::-webkit-scrollbar        { width: 5px; height: 5px; }
    ::-webkit-scrollbar-track { background: #12121f; }
    ::-webkit-scrollbar-thumb { background: #2d2d4e; border-radius: 3px; }

    /* ── Toast ────────────────────────────────────────────────────────── */
    .toast-container { position: fixed; bottom: 1.5rem; right: 1.5rem; z-index: 9999; }

    /* ── Responsive tweaks ────────────────────────────────────────────── */
    @media (max-width: 768px) {
      .sidebar { position: static; }
    }
  </style>
</head>
<body>

<!-- ═══════════════════════════ NAVBAR ══════════════════════════════════════ -->
<nav class="navbar navbar-expand-lg navbar-dark" style="background:#0d0d1a; border-bottom:1px solid #2d2d4e; position:sticky; top:0; z-index:1000;">
  <div class="container-fluid px-4">
    <a class="navbar-brand d-flex align-items-center gap-3" href="#">
      <div style="width:38px;height:38px;background:linear-gradient(135deg,#0f62fe,#6929c4);border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:1.3rem;">🧠</div>
      <div>
        <div class="navbar-brand-title">ResearchMind AI</div>
        <div class="navbar-brand-sub">Intelligent Research Companion</div>
      </div>
    </a>
    <div class="d-flex align-items-center gap-3 ms-auto">
      <span class="ibm-badge">IBM watsonx.ai</span>
      <span class="ibm-badge" style="background:#6929c4;">Granite Models</span>
      <span id="modeLabel" class="badge bg-warning text-dark" style="font-size:.7rem;">DEMO MODE</span>
      <button class="btn btn-sm btn-outline-secondary" onclick="clearSession()">
        <i class="bi bi-trash3"></i> Clear
      </button>
    </div>
  </div>
</nav>

<!-- ═══════════════════════════ HERO BANNER ════════════════════════════════ -->
<div style="background:linear-gradient(135deg,#0a0a1e 0%,#0d1635 50%,#0a0a1e 100%);border-bottom:1px solid #1e2a4a;padding:2rem 0;">
  <div class="container-fluid px-4">
    <div class="row align-items-center">
      <div class="col-lg-7">
        <h1 class="fw-bold mb-2" style="font-size:1.9rem;">
          Agentic AI Research Intelligence
          <span style="color:#0f62fe;">Powered by IBM Granite</span>
        </h1>
        <p class="text-secondary mb-3" style="font-size:.95rem;">
          Five specialised AI agents collaborate to retrieve literature, synthesise reviews,
          identify research gaps, forecast trends, and generate your personalised research plan —
          all orchestrated by a central IBM Granite reasoning engine.
        </p>
        <div class="d-flex flex-wrap gap-2">
          <span class="badge" style="background:#1e3a5f;color:#93c5fd;font-size:.75rem;"><i class="bi bi-cpu me-1"></i>IBM Granite 3.3 8B Instruct</span>
          <span class="badge" style="background:#2d1b4e;color:#c4b5fd;font-size:.75rem;"><i class="bi bi-diagram-3 me-1"></i>5-Agent Agentic Architecture</span>
          <span class="badge" style="background:#1a2e1a;color:#86efac;font-size:.75rem;"><i class="bi bi-database me-1"></i>Lightweight RAG System</span>
          <span class="badge" style="background:#2e1a1a;color:#fca5a5;font-size:.75rem;"><i class="bi bi-file-earmark-pdf me-1"></i>PDF / TXT Upload</span>
        </div>
      </div>
      <div class="col-lg-5 mt-4 mt-lg-0">
        <!-- System status card -->
        <div class="card-dark p-3" id="statusCard">
          <div class="section-header mb-2">System Status</div>
          <div id="statusContent" class="text-secondary" style="font-size:.82rem;">Loading…</div>
        </div>
      </div>
    </div>
  </div>
</div>

<!-- ═══════════════════════════ MAIN LAYOUT ════════════════════════════════ -->
<div class="container-fluid px-4 py-4">
  <div class="row g-4">

    <!-- ── LEFT SIDEBAR ─────────────────────────────────────────────────── -->
    <div class="col-lg-3">
      <div class="sidebar">

        <!-- Upload Panel -->
        <div class="card-dark mb-4">
          <div class="card-header">
            <span class="fw-semibold"><i class="bi bi-cloud-upload me-2"></i>Knowledge Upload</span>
          </div>
          <div class="p-3">
            <div class="upload-zone" id="uploadZone" onclick="document.getElementById('fileInput').click()">
              <i class="bi bi-file-earmark-richtext" style="font-size:2rem;color:#4a5568;"></i>
              <p class="mt-2 mb-1 text-secondary" style="font-size:.85rem;">Drop PDF or TXT here</p>
              <p style="font-size:.75rem;color:#4a5568;">Supports research papers, journals, whitepapers</p>
            </div>
            <input type="file" id="fileInput" accept=".pdf,.txt" style="display:none" onchange="uploadFile(this)"/>
            <div id="uploadStatus" class="mt-2" style="font-size:.8rem;"></div>
            <div id="kbStats" class="mt-2 text-secondary" style="font-size:.78rem;"></div>
          </div>
        </div>

        <!-- Agent Workflow Panel -->
        <div class="card-dark mb-4">
          <div class="card-header">
            <span class="fw-semibold"><i class="bi bi-diagram-3 me-2"></i>Agent Workflow</span>
          </div>
          <div class="p-3" id="workflowPanel">
            <div class="flow-step" id="step-orchestrator">
              <div class="step-num">★</div>
              <div>
                <div class="fw-semibold" style="font-size:.8rem;">Master Orchestrator</div>
                <div style="font-size:.7rem;color:#64748b;">Coordinates all agents</div>
              </div>
            </div>
            <div class="flow-step" id="step-0">
              <div class="step-num" style="color:#3b82f6;">1</div>
              <div>
                <div class="fw-semibold" style="font-size:.8rem;">🔍 Research Retrieval</div>
                <div style="font-size:.7rem;color:#64748b;">Extract & organise knowledge</div>
              </div>
            </div>
            <div class="flow-step" id="step-1">
              <div class="step-num" style="color:#8b5cf6;">2</div>
              <div>
                <div class="fw-semibold" style="font-size:.8rem;">📚 Literature Review</div>
                <div style="font-size:.7rem;color:#64748b;">Synthesise & compare</div>
              </div>
            </div>
            <div class="flow-step" id="step-2">
              <div class="step-num" style="color:#ef4444;">3</div>
              <div>
                <div class="fw-semibold" style="font-size:.8rem;">🔬 Gap Analysis</div>
                <div style="font-size:.7rem;color:#64748b;">Identify missing areas</div>
              </div>
            </div>
            <div class="flow-step" id="step-3">
              <div class="step-num" style="color:#f59e0b;">4</div>
              <div>
                <div class="fw-semibold" style="font-size:.8rem;">📈 Trend Forecasting</div>
                <div style="font-size:.7rem;color:#64748b;">Predict future directions</div>
              </div>
            </div>
            <div class="flow-step" id="step-4">
              <div class="step-num" style="color:#10b981;">5</div>
              <div>
                <div class="fw-semibold" style="font-size:.8rem;">🎯 Research Advisor</div>
                <div style="font-size:.7rem;color:#64748b;">Strategic guidance</div>
              </div>
            </div>
          </div>
        </div>

        <!-- Quick examples -->
        <div class="card-dark">
          <div class="card-header">
            <span class="fw-semibold"><i class="bi bi-lightbulb me-2"></i>Example Queries</span>
          </div>
          <div class="p-3">
            <div class="d-flex flex-column gap-2">
              <button class="btn btn-sm text-start" style="background:#0d0d2b;border:1px solid #2d2d4e;color:#93c5fd;font-size:.78rem;border-radius:6px;" onclick="setQuery('Transformer models for natural language processing')">
                Transformer models for NLP
              </button>
              <button class="btn btn-sm text-start" style="background:#0d0d2b;border:1px solid #2d2d4e;color:#c4b5fd;font-size:.78rem;border-radius:6px;" onclick="setQuery('Federated learning for privacy-preserving AI')">
                Federated learning & privacy
              </button>
              <button class="btn btn-sm text-start" style="background:#0d0d2b;border:1px solid #2d2d4e;color:#86efac;font-size:.78rem;border-radius:6px;" onclick="setQuery('AI-powered drug discovery and healthcare diagnostics')">
                AI in drug discovery
              </button>
              <button class="btn btn-sm text-start" style="background:#0d0d2b;border:1px solid #2d2d4e;color:#fcd34d;font-size:.78rem;border-radius:6px;" onclick="setQuery('Quantum machine learning algorithms and applications')">
                Quantum machine learning
              </button>
              <button class="btn btn-sm text-start" style="background:#0d0d2b;border:1px solid #2d2d4e;color:#fca5a5;font-size:.78rem;border-radius:6px;" onclick="setQuery('Climate change modelling using deep learning')">
                Climate change & deep learning
              </button>
            </div>
          </div>
        </div>

      </div>
    </div>

    <!-- ── MAIN CONTENT ──────────────────────────────────────────────────── -->
    <div class="col-lg-9">

      <!-- Query Input Card -->
      <div class="card-dark mb-4">
        <div class="card-header d-flex align-items-center justify-content-between">
          <span class="fw-semibold"><i class="bi bi-search me-2"></i>Research Query</span>
          <span style="font-size:.75rem;color:#64748b;">Powered by IBM Granite · Multi-Agent Orchestration</span>
        </div>
        <div class="p-3">
          <textarea id="queryInput" class="query-input mb-3" rows="3"
            placeholder="Enter your research question, topic, or abstract…&#10;e.g. 'What are the latest advances in explainable AI for medical imaging?'"></textarea>
          <div class="d-flex align-items-center gap-3 flex-wrap">
            <button class="btn btn-research" onclick="runResearch()" id="runBtn">
              <i class="bi bi-cpu me-2"></i>Launch Research Agents
            </button>
            <div id="loadingIndicator" style="display:none;" class="d-flex align-items-center gap-2 text-secondary">
              <div class="spinner-border spinner-border-sm text-primary" role="status"></div>
              <span id="loadingText" style="font-size:.85rem;">Initialising agents…</span>
            </div>
          </div>
        </div>
      </div>

      <!-- ── RESULTS AREA (hidden until research runs) ─────────────────── -->
      <div id="resultsArea" style="display:none;">

        <!-- Tabs navigation -->
        <ul class="nav nav-tabs mb-4" style="border-bottom:1px solid #2d2d4e;" id="resultTabs">
          <li class="nav-item">
            <button class="nav-link active" data-tab="agents" onclick="showTab('agents', this)"
              style="background:transparent;color:#93c5fd;border:none;border-bottom:2px solid #0f62fe;padding:.6rem 1rem;font-size:.88rem;">
              <i class="bi bi-diagram-3 me-1"></i>Agent Outputs
            </button>
          </li>
          <li class="nav-item">
            <button class="nav-link" data-tab="report" onclick="showTab('report', this)"
              style="background:transparent;color:#64748b;border:none;border-bottom:2px solid transparent;padding:.6rem 1rem;font-size:.88rem;">
              <i class="bi bi-file-text me-1"></i>Final Report
            </button>
          </li>
          <li class="nav-item">
            <button class="nav-link" data-tab="graph" onclick="showTab('graph', this)"
              style="background:transparent;color:#64748b;border:none;border-bottom:2px solid transparent;padding:.6rem 1rem;font-size:.88rem;">
              <i class="bi bi-share me-1"></i>Knowledge Graph
            </button>
          </li>
          <li class="nav-item">
            <button class="nav-link" data-tab="dashboard" onclick="showTab('dashboard', this)"
              style="background:transparent;color:#64748b;border:none;border-bottom:2px solid transparent;padding:.6rem 1rem;font-size:.88rem;">
              <i class="bi bi-grid-3x3-gap me-1"></i>Dashboard
            </button>
          </li>
        </ul>

        <!-- Tab: Agent Outputs -->
        <div id="tab-agents">
          <div class="section-header">Agent Collaboration Trace — showing why each agent was activated</div>
          <div id="agentOutputs"></div>
        </div>

        <!-- Tab: Final Report -->
        <div id="tab-report" style="display:none;">
          <div class="section-header">Research Intelligence Report — compiled by the Master Orchestrator</div>
          <div class="final-report" id="finalReportContent"></div>
        </div>

        <!-- Tab: Knowledge Graph -->
        <div id="tab-graph" style="display:none;">
          <div class="section-header mb-3">Concept Knowledge Graph — relationships between topics, agents, and concepts</div>
          <svg id="kgCanvas"></svg>
          <div class="mt-2 text-secondary" style="font-size:.75rem;">
            <span style="color:#0f62fe;">●</span> Query &nbsp;
            <span style="color:#6929c4;">●</span> Agents &nbsp;
            <span style="color:#10b981;">●</span> Concepts
          </div>
        </div>

        <!-- Tab: Dashboard -->
        <div id="tab-dashboard" style="display:none;">
          <div class="section-header">Research Intelligence Dashboard</div>
          <div class="row g-3" id="dashboardCards"></div>
        </div>

      </div><!-- /resultsArea -->

    </div><!-- /main col -->
  </div><!-- /row -->
</div><!-- /container -->

<!-- Toast container -->
<div class="toast-container" id="toastContainer"></div>

<!-- Bootstrap JS -->
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>

<script>
// ═══════════════════════════════════════════════════════════════════════════
//  ResearchMind AI – Frontend JavaScript
// ═══════════════════════════════════════════════════════════════════════════

let currentResults = null;

// ── Boot ────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  loadStatus();
  initDragDrop();
});

// ── System Status ────────────────────────────────────────────────────────────
async function loadStatus() {
  try {
    const res  = await fetch('/api/status');
    const data = await res.json();
    const live = data.mode.startsWith('LIVE');
    document.getElementById('modeLabel').textContent = data.mode;
    document.getElementById('modeLabel').className   = live
      ? 'badge bg-success' : 'badge bg-warning text-dark';
    document.getElementById('statusContent').innerHTML = `
      <div class="d-flex flex-column gap-1">
        <div class="d-flex justify-content-between">
          <span>IBM watsonx.ai SDK</span>
          <span class="${data.watsonx_available?'text-success':'text-danger'}">${data.watsonx_available?'✓ Installed':'✗ Missing'}</span>
        </div>
        <div class="d-flex justify-content-between">
          <span>Granite Credentials</span>
          <span class="${data.watsonx_configured?'text-success':'text-warning'}">${data.watsonx_configured?'✓ Configured':'⚠ Not Set'}</span>
        </div>
        <div class="d-flex justify-content-between">
          <span>RAG System (TF-IDF)</span>
          <span class="${data.rag_available?'text-success':'text-warning'}">${data.rag_available?'✓ Ready':'⚠ Disabled'}</span>
        </div>
        <div class="d-flex justify-content-between">
          <span>PDF Extraction</span>
          <span class="${data.pdf_available?'text-success':'text-warning'}">${data.pdf_available?'✓ Ready':'⚠ Disabled'}</span>
        </div>
        <div class="d-flex justify-content-between">
          <span>Primary Model</span>
          <span style="color:#93c5fd;font-size:.75rem;">${data.primary_model}</span>
        </div>
        <div class="d-flex justify-content-between">
          <span>Knowledge Chunks</span>
          <span style="color:#86efac;">${data.knowledge_base_chunks}</span>
        </div>
      </div>`;
  } catch(e) {
    document.getElementById('statusContent').textContent = 'Status unavailable';
  }
}

// ── File Upload ───────────────────────────────────────────────────────────────
function initDragDrop() {
  const zone = document.getElementById('uploadZone');
  zone.addEventListener('dragover',  e => { e.preventDefault(); zone.classList.add('drag-over'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
  zone.addEventListener('drop', e => {
    e.preventDefault();
    zone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file) uploadFileObj(file);
  });
}

function uploadFile(input) {
  if (input.files[0]) uploadFileObj(input.files[0]);
}

async function uploadFileObj(file) {
  const status = document.getElementById('uploadStatus');
  status.innerHTML = `<span class="text-warning spinner-pulse"><i class="bi bi-arrow-repeat me-1"></i>Uploading & indexing…</span>`;
  const fd = new FormData();
  fd.append('file', file);
  try {
    const res  = await fetch('/api/upload', { method: 'POST', body: fd });
    const data = await res.json();
    if (data.success) {
      status.innerHTML = `<span class="text-success"><i class="bi bi-check-circle me-1"></i>${data.filename} — ${data.chunks_added} chunks indexed</span>`;
      document.getElementById('kbStats').innerHTML =
        `<i class="bi bi-database me-1"></i>Knowledge base: <strong style="color:#86efac;">${data.total_chunks}</strong> chunks`;
      loadStatus();
      showToast(`✓ ${data.filename} added to knowledge base (${data.chunks_added} chunks)`, 'success');
    } else {
      status.innerHTML = `<span class="text-danger"><i class="bi bi-x-circle me-1"></i>${data.error}</span>`;
    }
  } catch(e) {
    status.innerHTML = `<span class="text-danger">Upload failed: ${e.message}</span>`;
  }
}

// ── Example Query Setter ──────────────────────────────────────────────────────
function setQuery(q) {
  document.getElementById('queryInput').value = q;
  document.getElementById('queryInput').focus();
}

// ── Run Research (Main Action) ────────────────────────────────────────────────
async function runResearch() {
  const query = document.getElementById('queryInput').value.trim();
  if (!query) { showToast('Please enter a research query.', 'warning'); return; }

  // UI: show loading
  document.getElementById('runBtn').disabled = true;
  document.getElementById('loadingIndicator').style.display = 'flex';
  document.getElementById('resultsArea').style.display = 'none';
  resetWorkflowSteps();
  activateWorkflowStep('orchestrator');

  const loadingTexts = [
    'Initialising Master Orchestrator…',
    'Agent 1: Retrieving research knowledge…',
    'Agent 2: Generating literature review…',
    'Agent 3: Analysing research gaps…',
    'Agent 4: Forecasting emerging trends…',
    'Agent 5: Building research plan…',
    'Compiling final report…',
  ];
  let textIdx = 0;
  const textTimer = setInterval(() => {
    document.getElementById('loadingText').textContent = loadingTexts[Math.min(textIdx++, loadingTexts.length-1)];
    if (textIdx <= 5) activateWorkflowStep(textIdx - 1);
  }, 900);

  try {
    const res  = await fetch('/api/research', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query })
    });
    const data = await res.json();
    clearInterval(textTimer);

    if (data.success) {
      currentResults = data.results;
      renderResults(data.results);
      markAllWorkflowDone();
      showToast('Research complete — all agents finished!', 'success');
    } else {
      showToast('Error: ' + data.error, 'danger');
      resetWorkflowSteps();
    }
  } catch(e) {
    clearInterval(textTimer);
    showToast('Request failed: ' + e.message, 'danger');
    resetWorkflowSteps();
  } finally {
    document.getElementById('runBtn').disabled = false;
    document.getElementById('loadingIndicator').style.display = 'none';
  }
}

// ── Render All Results ────────────────────────────────────────────────────────
function renderResults(results) {
  // Show area & default to agents tab
  document.getElementById('resultsArea').style.display = 'block';
  showTab('agents', document.querySelector('[data-tab="agents"]'));

  // Agent outputs
  const container = document.getElementById('agentOutputs');
  container.innerHTML = '';
  results.agents.forEach((agent, i) => {
    const card = document.createElement('div');
    card.className = `agent-card agent-${i+1}`;
    card.innerHTML = `
      <div class="d-flex align-items-start gap-3 mb-3">
        <div class="agent-icon" style="background:${agent.color}22;">
          <span>${agent.icon}</span>
        </div>
        <div class="flex-grow-1">
          <div class="d-flex align-items-center gap-2 flex-wrap">
            <span class="fw-bold" style="color:${agent.color};">${agent.agent}</span>
            <span class="badge" style="background:${agent.color}33;color:${agent.color};font-size:.65rem;">Agent ${i+1}</span>
            ${agent.context_used ? '<span class="badge bg-success bg-opacity-25 text-success" style="font-size:.65rem;"><i class="bi bi-database me-1"></i>RAG Context Used</span>' : ''}
            <span class="ms-auto text-secondary" style="font-size:.72rem;"><i class="bi bi-clock me-1"></i>${agent.timestamp}</span>
          </div>
          <div class="text-secondary mt-1" style="font-size:.78rem;">
            <i class="bi bi-info-circle me-1"></i>${agent.why_activated}
          </div>
        </div>
      </div>
      <div class="agent-output-body">${escapeAndFormat(agent.output)}</div>`;
    container.appendChild(card);
  });

  // Final report
  document.getElementById('finalReportContent').textContent = results.final_report;

  // Knowledge graph
  renderKnowledgeGraph(results.knowledge_graph);

  // Dashboard
  renderDashboard(results);

  loadStatus();
}

// ── Knowledge Graph SVG Renderer ──────────────────────────────────────────────
function renderKnowledgeGraph(graph) {
  const svg  = document.getElementById('kgCanvas');
  const W    = svg.getBoundingClientRect().width || 800;
  const H    = 380;
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  svg.innerHTML = '';

  if (!graph || !graph.nodes || graph.nodes.length === 0) {
    svg.innerHTML = `<text x="${W/2}" y="${H/2}" text-anchor="middle" fill="#4a5568" font-size="14">No graph data</text>`;
    return;
  }

  const nodes = graph.nodes;
  const edges = graph.edges;

  // Simple force-like layout: place query in centre, agents in ring, concepts in outer ring
  const cx = W / 2, cy = H / 2;
  const nodePositions = {};

  nodes.forEach((node, i) => {
    if (node.type === 'query') {
      nodePositions[node.id] = { x: cx, y: cy };
    } else if (node.type === 'agent') {
      const agentNodes = nodes.filter(n => n.type === 'agent');
      const idx = agentNodes.findIndex(n => n.id === node.id);
      const angle = (2 * Math.PI * idx / agentNodes.length) - Math.PI / 2;
      nodePositions[node.id] = {
        x: cx + 140 * Math.cos(angle),
        y: cy + 110 * Math.sin(angle),
      };
    } else {
      const conceptNodes = nodes.filter(n => n.type === 'concept');
      const idx = conceptNodes.findIndex(n => n.id === node.id);
      const angle = (2 * Math.PI * idx / conceptNodes.length) - Math.PI / 4;
      nodePositions[node.id] = {
        x: cx + 230 * Math.cos(angle),
        y: cy + 160 * Math.sin(angle),
      };
    }
  });

  // Draw edges
  edges.forEach(edge => {
    const s = nodePositions[edge.source];
    const t = nodePositions[edge.target];
    if (!s || !t) return;
    const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    line.setAttribute('x1', s.x); line.setAttribute('y1', s.y);
    line.setAttribute('x2', t.x); line.setAttribute('y2', t.y);
    line.setAttribute('stroke', '#2d2d4e');
    line.setAttribute('stroke-width', '1.5');
    line.setAttribute('stroke-dasharray', '4,3');
    svg.appendChild(line);
  });

  // Draw nodes
  nodes.forEach(node => {
    const pos = nodePositions[node.id];
    if (!pos) return;
    const r     = node.size || 14;
    const fill  = node.type === 'query' ? '#0f62fe' : node.type === 'agent' ? (node.color || '#6929c4') : '#10b981';
    const g     = document.createElementNS('http://www.w3.org/2000/svg', 'g');

    const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    circle.setAttribute('cx', pos.x); circle.setAttribute('cy', pos.y);
    circle.setAttribute('r',  r);
    circle.setAttribute('fill', fill + '33');
    circle.setAttribute('stroke', fill);
    circle.setAttribute('stroke-width', '2');
    g.appendChild(circle);

    const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    text.setAttribute('x', pos.x);
    text.setAttribute('y', pos.y + r + 12);
    text.setAttribute('text-anchor', 'middle');
    text.setAttribute('fill', '#94a3b8');
    text.setAttribute('font-size', '10');
    text.textContent = node.label.length > 18 ? node.label.substring(0,16)+'…' : node.label;
    g.appendChild(text);

    svg.appendChild(g);
  });
}

// ── Dashboard Cards ───────────────────────────────────────────────────────────
function renderDashboard(results) {
  const container = document.getElementById('dashboardCards');
  const icons  = ['🔍','📚','🔬','📈','🎯'];
  const colors  = ['#3b82f6','#8b5cf6','#ef4444','#f59e0b','#10b981'];
  const labels = ['Literature Summary','Review Synthesis','Gap Analysis','Trend Forecast','Research Plan'];

  container.innerHTML = results.agents.map((agent, i) => `
    <div class="col-md-6 col-xl-4">
      <div class="card-dark h-100">
        <div class="card-header d-flex align-items-center gap-2">
          <span style="font-size:1.1rem;">${icons[i]}</span>
          <span class="fw-semibold" style="font-size:.85rem;">${labels[i]}</span>
          <span class="ms-auto badge" style="background:${colors[i]}22;color:${colors[i]};font-size:.65rem;">Agent ${i+1}</span>
        </div>
        <div class="p-3">
          <div style="font-size:.8rem;color:#94a3b8;line-height:1.6;max-height:180px;overflow:hidden;text-overflow:ellipsis;">
            ${escapeAndFormat(agent.output.substring(0, 400))}${agent.output.length > 400 ? '…' : ''}
          </div>
        </div>
      </div>
    </div>`).join('');

  // Add final report card
  container.innerHTML += `
    <div class="col-12">
      <div class="card-dark">
        <div class="card-header d-flex align-items-center gap-2">
          <span style="font-size:1.1rem;">🏆</span>
          <span class="fw-semibold">Research Intelligence Report</span>
          <span class="ms-auto badge" style="background:#0f62fe22;color:#93c5fd;font-size:.65rem;">Master Orchestrator</span>
        </div>
        <div class="p-3">
          <div style="font-size:.82rem;color:#94a3b8;line-height:1.7;white-space:pre-wrap;max-height:200px;overflow-y:auto;">
            ${escapeAndFormat(results.final_report.substring(0, 800))}${results.final_report.length > 800 ? '\n\n[See Full Report tab for complete output]' : ''}
          </div>
        </div>
      </div>
    </div>`;
}

// ── Tab Switcher ──────────────────────────────────────────────────────────────
function showTab(tabName, btn) {
  ['agents','report','graph','dashboard'].forEach(t => {
    document.getElementById('tab-' + t).style.display = t === tabName ? 'block' : 'none';
  });
  document.querySelectorAll('#resultTabs .nav-link').forEach(b => {
    b.style.color = '#64748b';
    b.style.borderBottom = '2px solid transparent';
  });
  if (btn) {
    btn.style.color = '#93c5fd';
    btn.style.borderBottom = '2px solid #0f62fe';
  }
  if (tabName === 'graph' && currentResults) {
    // Re-render after layout
    setTimeout(() => renderKnowledgeGraph(currentResults.knowledge_graph), 50);
  }
}

// ── Workflow Step Animations ──────────────────────────────────────────────────
function resetWorkflowSteps() {
  document.querySelectorAll('.flow-step').forEach(el => {
    el.classList.remove('active','done');
  });
}
function activateWorkflowStep(idx) {
  if (idx === 'orchestrator') {
    document.getElementById('step-orchestrator').classList.add('active');
    return;
  }
  const el = document.getElementById('step-' + idx);
  if (el) {
    // Mark previous as done
    if (idx > 0) {
      const prev = document.getElementById('step-' + (idx-1));
      if (prev) { prev.classList.remove('active'); prev.classList.add('done'); }
    }
    el.classList.add('active');
  }
}
function markAllWorkflowDone() {
  document.querySelectorAll('.flow-step').forEach(el => {
    el.classList.remove('active');
    el.classList.add('done');
  });
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function escapeAndFormat(text) {
  return text
    .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
    .replace(/\*\*(.*?)\*\*/g, '<strong style="color:#e2e8f0;">$1</strong>')
    .replace(/\n/g, '<br>');
}

function showToast(message, type = 'info') {
  const colors = { success:'#10b981', danger:'#ef4444', warning:'#f59e0b', info:'#3b82f6' };
  const id = 'toast-' + Date.now();
  const el = document.createElement('div');
  el.id = id;
  el.style.cssText = `background:#12121f;border:1px solid ${colors[type]||'#2d2d4e'};border-radius:8px;padding:.75rem 1.1rem;color:#e2e8f0;font-size:.85rem;margin-top:.5rem;min-width:260px;box-shadow:0 4px 20px rgba(0,0,0,.5);`;
  el.innerHTML = `<span style="color:${colors[type]};">●</span> ${message}`;
  document.getElementById('toastContainer').appendChild(el);
  setTimeout(() => el.remove(), 4000);
}

async function clearSession() {
  await fetch('/api/clear', { method: 'POST' });
  document.getElementById('resultsArea').style.display = 'none';
  document.getElementById('uploadStatus').innerHTML = '';
  document.getElementById('kbStats').innerHTML = '';
  document.getElementById('queryInput').value = '';
  resetWorkflowSteps();
  currentResults = null;
  loadStatus();
  showToast('Session cleared — knowledge base reset.', 'info');
}
</script>
</body>
</html>
"""


# =============================================================================
#  SECTION 7: Application Entry Point
# =============================================================================

if __name__ == "__main__":
    # Force UTF-8 stdout so the banner prints correctly on all platforms
    import sys, io
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

    print("""
+--------------------------------------------------------------------------+
|   ResearchMind AI -- Intelligent Research Companion                      |
|   Powered by IBM watsonx.ai Studio & IBM Granite Models                  |
+--------------------------------------------------------------------------+
|                                                                          |
|   Setup (if not done):                                                   |
|     pip install flask ibm-watsonx-ai PyPDF2 scikit-learn numpy           |
|                                                                          |
|   Credentials loaded from: .env                                          |
|     WATSONX_API_KEY    = (set in .env)                                   |
|     WATSONX_PROJECT_ID = (set in .env)                                   |
|     WATSONX_URL        = https://us-south.ml.cloud.ibm.com               |
|                                                                          |
|   Running in DEMO mode if credentials are not configured.                |
|   Visit: http://localhost:5000                                           |
+--------------------------------------------------------------------------+
""")
    _host  = os.getenv("FLASK_HOST",  "0.0.0.0")
    _port  = int(os.getenv("FLASK_PORT", "5000"))
    _debug = os.getenv("FLASK_DEBUG", "true").lower() == "true"
    app.run(debug=_debug, host=_host, port=_port)