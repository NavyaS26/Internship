import os
import torch
import pandas as pd
import pdfplumber
import spacy
import streamlit as st
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline, logging
import re
import unicodedata

# Suppress warnings
logging.set_verbosity_error()

# ----------------------------
# CONFIGURATION
# ----------------------------
device = "cpu"
model_name = "nlpaueb/legal-bert-base-uncased"
custom_keywords = [
    "agreement", "contract", "obligation", "termination", "confidentiality", "liability", "dispute", "arbitration", "indemnity", "jurisdiction",
    "governing law", "force majeure", "intellectual property", "payment terms", "breach", "remedy", "warranty", "representation", "assignment",
    "entire agreement", "severability", "amendment", "waiver", "notices", "execution", "compliance", "injunction", "damages", "fees",
    "non-disclosure", "data protection", "privacy policy", "retention", "termination clause", "non-compete", "non-solicitation", "exclusive rights",
    "affiliate", "counterparts", "applicable law", "legal entity", "third party", "disclosure", "revocation",
    "consultant", "client", "employer", "employee", "agent", "representative", "party", "supplier", "vendor", "customer", "subcontractor",
    "service level agreement", "deliverable", "timeline", "milestone", "scope of work", "acceptance criteria", "intellectual property ownership",
    "change order", "conflict of interest", "audit", "termination for convenience", "termination for cause", "financial obligation",
    "invoice", "billing", "due date", "late fee", "tax", "penalty", "reimbursement", "advance", "credit note", "refund", "payment method"
]

CLAUSE_KEYWORDS = {
    "Confidentiality": ["confidential", "non-disclosure", "nda"],
    "Indemnification": ["indemnify", "indemnification", "hold harmless"],
    "Force Majeure": ["force majeure", "acts of god", "unforeseen event"],
    "Governing Law": ["governing law", "jurisdiction", "laws of"],
    "Termination": ["terminate", "termination", "end of contract"],
    "Intellectual Property Ownership": ["intellectual property", "ip ownership", "retain ownership"],
    "Limitation of Liability": ["limitation of liability", "limited liability", "liability cap"],
    "Warranties": ["warranties", "no warranties", "as-is"],
    "Dispute Resolution": ["dispute", "resolution", "settle disputes"],
    "Arbitration": ["arbitration", "arbitrator", "binding arbitration"],
    "Insurance": ["insurance", "coverage", "insured"]
}

def normalize(text):
    text = unicodedata.normalize("NFKD", text).lower().strip()
    return re.sub(r'[^a-z0-9\s]', '', text)

@st.cache_resource
def load_models():
    model = SentenceTransformer(model_name, trust_remote_code=True).to(device)
    summarizer = pipeline("summarization", model="t5-base", tokenizer="t5-base", device=0 if torch.cuda.is_available() else -1)
    nlp = spacy.load("en_core_web_sm")
    return model, summarizer, nlp

model, summarizer, nlp = load_models()

@st.cache_data
def load_clauses(filepath="cuad_clauses.csv"):
    df = pd.read_csv(filepath)
    clauses = df["clause_name"].dropna().str.lower().unique().tolist()
    embeddings = model.encode(clauses, convert_to_tensor=True, device=device)
    return clauses, embeddings

def extract_chunks(pdf_file):
    chunks = []
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                split_chunks = text.split("\n\n")
                for chunk in split_chunks:
                    cleaned = chunk.strip()
                    if len(cleaned.split()) > 10:
                        chunks.append(cleaned)
    return chunks

def match_clauses_semantic(chunks, clause_names, clause_embeddings, threshold=0.82):
    found = {}
    for chunk in chunks:
        embedding = model.encode(chunk, convert_to_tensor=True, device=device)
        scores = util.cos_sim(embedding, clause_embeddings)[0]
        matches = [(i, score.item()) for i, score in enumerate(scores) if score.item() >= threshold]
        matches = sorted(matches, key=lambda x: x[1], reverse=True)
        for i, score in matches:
            clause = clause_names[i]
            if clause not in found:
                found[clause] = score
    return found

def match_clauses_keyword(text):
    found = set()
    lowered = normalize(text)
    for clause, keywords in CLAUSE_KEYWORDS.items():
        for kw in keywords:
            if normalize(kw) in lowered:
                found.add(clause)
                break
    return sorted(found)

def detect_missing_clauses(detected, reference):
    return sorted(set(reference) - set(detected))

def extract_entities(chunks):
    entities = set()
    for chunk in chunks:
        doc = nlp(chunk)
        for ent in doc.ents:
            entities.add((ent.text, ent.label_))
    return sorted(entities)

def summarize_document(chunks, max_chunk_words=400, max_output_tokens=200):
    final_summary = ""
    for chunk in chunks:
        word_count = len(chunk.split())
        if word_count < 30:
            continue
        max_len = min(max_output_tokens, int(word_count * 1.2))
        min_len = max(40, int(max_len * 0.5))
        try:
            result = summarizer(
                "summarize: " + chunk,
                max_length=max_len,
                min_length=min_len,
                do_sample=False
            )[0]["summary_text"]
            final_summary += f"{result} "
        except Exception:
            continue
    return final_summary.strip() if final_summary else "No valid summary could be generated."

def clean_summary_text(text):
    return re.sub(r'[^\x00-\x7F]+', '', text)

st.set_page_config(page_title="AI Legal Assistant", layout="wide")
st.title("📁 AI Legal Clause Analyzer")

uploaded_file = st.file_uploader("Upload a legal PDF document", type=["pdf"])
if uploaded_file:
    clause_names, clause_embeddings = load_clauses()
    chunks = extract_chunks(uploaded_file)
    full_text = " ".join(chunks)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔎 Clause (Exact Match)", "🧠 Semantics (LegalBERT)", "📝 Summary", "🔑 Key Entities", "❌ Missing Clauses"])

    with tab1:
        matched_clauses = match_clauses_keyword(full_text)
        st.subheader("✅ Detected Clauses (Exact Keyword Match)")
        for clause in matched_clauses:
            st.markdown(f"- {clause}")

    with tab2:
        matched_semantic = match_clauses_semantic(chunks, clause_names, clause_embeddings)
        st.subheader("🧠 Detected Clauses (Semantic Match)")
        for clause in sorted(matched_semantic):
            with st.expander(f"🔹 {clause.title()}"):
                st.markdown(f"**Similarity Score:** `{matched_semantic[clause]:.2f}`")

    with tab3:
        summary = summarize_document(chunks)
        st.subheader("📝 Summary")
        st.text_area("Summary", clean_summary_text(summary), height=300)

    with tab4:
        entities = extract_entities(chunks)
        st.subheader("🔑 Key Entities")
        if entities:
            df = pd.DataFrame(entities, columns=["Entity", "Label"])
            st.dataframe(df)
        else:
            st.info("No entities found.")

    with tab5:
        matched_clause_names = [c.lower() for c in matched_semantic.keys()] if matched_semantic else []
        missing_clauses = detect_missing_clauses(matched_clause_names, clause_names)
        st.subheader("❌ Missing Clauses")
        if missing_clauses:
            for clause in missing_clauses:
                st.markdown(f"- {clause.title()}")
        else:
            st.success("No important clauses missing!")

else:
    st.info("Please upload a PDF to begin.")
