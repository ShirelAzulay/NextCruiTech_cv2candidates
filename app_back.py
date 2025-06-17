#streamlit run app.py




import streamlit as st
import pandas as pd
import logging
from src.loader import load_all_texts
from src.filterer import filter_by_prompt
from src.embeddings import compute_embeddings
from src.indexer import build_faiss_index, save_index
from src.matcher import match_resumes_to_jobs

logging.basicConfig(level=logging.ERROR)

st.set_page_config(page_title="CV-to-Job Matcher", layout="wide")
st.title("📋 CV-to-Job Matching Tool")

st.sidebar.header("Configuration")
prompt = st.sidebar.text_area(
    "Filter Prompt", height=100,
    value="Does this candidate have at least 2 years of professional experience?"
)
threshold = st.sidebar.slider("Matching Threshold", 0.0, 1.0, 0.75)
top_k = st.sidebar.number_input("Top K Matches per Job", 1, 20, 5)
run_button = st.sidebar.button("Run Pipeline")

if run_button:
    st.info("🔍 Loading data...")
    resumes = load_all_texts("resumes/")
    jobs = load_all_texts("jobs/")
    st.success(f"Loaded {len(resumes)} resumes and {len(jobs)} job descriptions")

    st.info("📝 Filtering resumes...")
    filtered_resumes = filter_by_prompt(resumes, prompt)
    st.success(f"{len(filtered_resumes)} resumes remain after filtering")

    if not filtered_resumes:
        st.warning("No resumes passed the filter. Adjust your prompt or data and try again.")
    else:
        st.info("🤖 Computing embeddings...")
        resume_vecs = compute_embeddings(filtered_resumes)
        job_vecs = compute_embeddings(jobs)
        st.success("Embeddings computed")

        st.info("📚 Building FAISS index...")
        index = build_faiss_index(resume_vecs)
        save_index(index, "faiss.index")
        st.success("Index built")

        st.info("🔗 Matching resumes to jobs...")
        matches_df = match_resumes_to_jobs(index, job_vecs, top_k=top_k)
        matches_df = matches_df[matches_df["score"] >= threshold]
        st.success(f"Found {len(matches_df)} matches above threshold")

        st.subheader("📊 Matching Results")
        st.dataframe(matches_df)
