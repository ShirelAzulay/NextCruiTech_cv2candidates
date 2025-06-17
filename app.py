import streamlit as st
import pandas as pd
import openai
from openai import OpenAI
import base64
import time
import os
import json

from src.embeddings import update_embeddings, get_processing_summary, scan_folder, load_embeddings
from src.indexer import build_faiss_index
from src.matcher import get_topk_matches, gpt_filter_topk

# ---- Configuration ----
openai.api_key = "sk-proj-"
RESUMES_FOLDER = "resumes/"
JOBS_FOLDER = "jobs/"

# ---- Page Setup ----
st.set_page_config(
    page_title="CV-to-Job Algorithm",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- Simple Dark Theme ----
st.markdown("""
<style>
    /* Basic dark theme */
    .stApp {
        background-color: #1a1a2e;
        color: #ffffff;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0f0f1e;
    }

    /* Headers */
    h1, h2, h3 {
        color: #ffffff !important;
    }

    /* Metrics */
    [data-testid="metric-container"] {
        background-color: #16213e;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e94560;
    }

    /* Logo animation */
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }

    .logo-animation {
        animation: pulse 2s ease-in-out infinite;
    }
</style>
""", unsafe_allow_html=True)


# ---- Helper Functions ----
def load_embeddings():
    """Load existing embeddings from file"""
    if os.path.exists("embeddings.json"):
        with open("embeddings.json", "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def get_base64_of_image(image_path):
    """Convert image to base64 string"""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except:
        return None


def get_detailed_explanation(client, job_text, resume_text, filter_prompt, score):
    """Generate detailed Hebrew explanation for why resume matches job"""
    prompt = f"""
    You are an HR expert. Explain in Hebrew (2-5 sentences) why this resume matches this job.

    Job Description (first 1000 chars): {job_text[:1000]}

    Resume (first 1500 chars): {resume_text[:1500]}

    Match Score: {score:.2%}

    {f"Filter criteria: {filter_prompt}" if filter_prompt else ""}

    Instructions:
    - Write ONLY in Hebrew
    - Be specific about skills, experience, and qualifications that match
    - Mention any red flags or concerns if they exist
    - If filter criteria was provided, explain how the candidate meets it
    - Be honest and balanced
    """

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=200
        )
        return response.choices[0].message.content.strip()
    except:
        return "לא ניתן לייצר הסבר כרגע"


# ---- Initialize session state ----
if 'show_loader' not in st.session_state:
    st.session_state.show_loader = False
if 'results_ready' not in st.session_state:
    st.session_state.results_ready = False

# ---- Header ----
st.markdown("# 🕵️‍♂️ CV-to-Job Matching Algorithm")

# Display logo if exists
logo_base64 = get_base64_of_image("logo.png")
if logo_base64:
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.markdown(
            f'<div style="text-align: center;"><img src="data:image/png;base64,{logo_base64}" width="150" class="logo-animation"></div>',
            unsafe_allow_html=True
        )

# ---- Sidebar ----
with st.sidebar:
    st.header("⚙️ Configuration")

    st.subheader("🔎 Advanced Filters")
    filter_prompt = st.text_area(
        "Filter Prompt (Hebrew/English)",
        placeholder="e.g., מועמד עם ניסיון של מעל 3 שנים בפיתוח",
        help="Describe your ideal candidate",
        height=100
    )

    st.subheader("⚖️ Matching Settings")
    threshold = st.slider("📈 Matching Threshold", 0.0, 1.0, 0.7)
    top_k = st.number_input("🎯 Top K Matches per Job", 1, 30, 20)

    st.subheader("🖥️ Display Settings")
    show_stats = st.checkbox("📊 Show Detailed Statistics", value=True)

    run_button = st.button("🚀 Run Pipeline", type="primary", use_container_width=True)

    # Maintenance section
    st.markdown("---")
    st.subheader("🔧 Maintenance")

    if st.button("🧹 Clean Old Embeddings", help="Remove embeddings for deleted files"):
        with st.spinner("Cleaning..."):
            from src.embeddings import clean_old_embeddings, save_embeddings

            embeddings = load_embeddings()
            original_count = len(embeddings)

            embeddings = clean_old_embeddings(embeddings, RESUMES_FOLDER)
            embeddings = clean_old_embeddings(embeddings, JOBS_FOLDER)

            cleaned_count = original_count - len(embeddings)
            if cleaned_count > 0:
                save_embeddings(embeddings)
                st.success(f"✅ Cleaned {cleaned_count} old embeddings")
            else:
                st.info("No old embeddings to clean")

    # Show embeddings stats
    if os.path.exists("embeddings.json"):
        embeddings = load_embeddings()
        st.markdown(f"**📊 Total embeddings:** {len(embeddings)}")

        # Count by folder
        resume_count = sum(1 for k in embeddings if k.startswith("resumes|"))
        job_count = sum(1 for k in embeddings if k.startswith("jobs|"))
        st.markdown(f"**📄 Resumes:** {resume_count}")
        st.markdown(f"**💼 Jobs:** {job_count}")

# ---- Show loader when processing ----
if st.session_state.show_loader and not st.session_state.results_ready:
    with st.spinner('Processing... This may take a few minutes on first run'):
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Initialize OpenAI client
        client = OpenAI(api_key=openai.api_key)

        # Step 1: Pre-scan and generate embeddings
        status_text.text("📂 Scanning files...")
        progress_bar.progress(5)

        # Load existing embeddings for scanning
        existing_embeddings = load_embeddings()

        # Pre-scan both folders
        resume_status = scan_folder(RESUMES_FOLDER, existing_embeddings)
        job_status = scan_folder(JOBS_FOLDER, existing_embeddings)

        # Display pre-scan results
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📄 Resumes Folder")
            st.success(f"✅ Valid files: {len(resume_status.valid_files)}")
            if resume_status.skipped_files:
                st.info(f"⏭️ Already processed: {len(resume_status.skipped_files)}")
            if resume_status.invalid_files:
                with st.expander(f"❌ Invalid files ({len(resume_status.invalid_files)})", expanded=False):
                    for filename, reason in resume_status.invalid_files:
                        st.text(f"• {filename}: {reason}")

        with col2:
            st.markdown("### 💼 Jobs Folder")
            st.success(f"✅ Valid files: {len(job_status.valid_files)}")
            if job_status.skipped_files:
                st.info(f"⏭️ Already processed: {len(job_status.skipped_files)}")
            if job_status.invalid_files:
                with st.expander(f"❌ Invalid files ({len(job_status.invalid_files)})", expanded=False):
                    for filename, reason in job_status.invalid_files:
                        st.text(f"• {filename}: {reason}")

        # Check if we have any valid files to process
        total_to_process = len(resume_status.valid_files) + len(job_status.valid_files)
        if total_to_process == 0 and len(resume_status.skipped_files) == 0 and len(job_status.skipped_files) == 0:
            st.error("No valid files found to process!")
            st.session_state.show_loader = False
            st.stop()

        # Process embeddings
        status_text.text("⏳ Creating embeddings for resumes...")
        progress_bar.progress(20)

        try:
            # Create a progress callback
            def update_progress(msg):
                status_text.text(f"⏳ {msg}")


            # Process resumes
            start_time = time.time()
            res_embeddings = update_embeddings(RESUMES_FOLDER, client, update_progress)
            resume_time = time.time() - start_time
            progress_bar.progress(50)

            # Process jobs
            status_text.text("⏳ Creating embeddings for jobs...")
            start_time = time.time()
            job_embeddings = update_embeddings(JOBS_FOLDER, client, update_progress)
            job_time = time.time() - start_time
            progress_bar.progress(60)

            # Display processing summary
            st.markdown("### ⏱️ Processing Summary")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Resumes processing time", f"{resume_time:.1f}s")
            with col2:
                st.metric("Jobs processing time", f"{job_time:.1f}s")

            # Show any processing errors
            processing_summary = get_processing_summary()
            if processing_summary:
                with st.expander("📝 Detailed Processing Log", expanded=False):
                    st.text(processing_summary)

        except Exception as e:
            st.error(f"Error creating embeddings: {str(e)}")
            st.session_state.show_loader = False
            st.stop()

        if not res_embeddings or not job_embeddings:
            st.error("No embeddings were created! Check your files and try again.")
            st.session_state.show_loader = False
            st.stop()

        status_text.text(f"✅ Embeddings ready: {len(res_embeddings)} resumes, {len(job_embeddings)} jobs")
        progress_bar.progress(70)

        # Step 2: Build FAISS index
        status_text.text("📊 Building search index...")
        index, resume_ids = build_faiss_index(res_embeddings)
        progress_bar.progress(75)

        # Step 3: Match jobs to resumes
        status_text.text("🔍 Matching resumes to jobs...")
        results = []

        total_jobs = len(job_embeddings)
        matching_start_time = time.time()

        for i, (job_id, job_data) in enumerate(job_embeddings.items()):
            # Update progress
            progress = 75 + (20 * i / total_jobs)
            progress_bar.progress(progress / 100)

            job_name = job_id.split('|')[1] if '|' in job_id else job_id
            status_text.text(f"🔍 Processing job {i + 1}/{total_jobs}: {job_name}")

            job_emb = job_data["embedding"]
            job_text = job_data.get("text", "")

            # Get top K matches
            candidates = get_topk_matches(index, res_embeddings, resume_ids, job_emb, top_k=top_k)

            # Apply GPT filter if provided
            if filter_prompt and filter_prompt.strip():
                candidates = gpt_filter_topk(client, candidates, filter_prompt)

            # Generate explanations for each match
            for item in candidates:
                if item["score"] >= threshold:
                    resume_text = item["resume_text"]
                    explanation = get_detailed_explanation(
                        client, job_text, resume_text, filter_prompt, item["score"]
                    )

                    results.append({
                        "Job": job_id.split('|')[1] if '|' in job_id else job_id,
                        "Resume": item["resume_id"].split('|')[1] if '|' in item["resume_id"] else item["resume_id"],
                        "Score": item["score"],
                        "Explanation": explanation
                    })

        matching_time = time.time() - matching_start_time
        progress_bar.progress(100)
        status_text.text("✅ Matching completed!")

        # Show final summary
        st.markdown("### 🎯 Final Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total matches found", len(results))
        with col2:
            st.metric("Matching time", f"{matching_time:.1f}s")
        with col3:
            st.metric("Avg matches per job", f"{len(results) / total_jobs:.1f}" if total_jobs > 0 else "0")

        # Store results
        if results:
            st.session_state.results_df = pd.DataFrame(results)
            st.session_state.results_ready = True
            st.session_state.show_loader = False
            time.sleep(0.5)
            st.rerun()
        else:
            st.error("No matches found! Try adjusting the threshold or filter criteria.")
            st.session_state.show_loader = False

# ---- Trigger processing ----
if run_button:
    st.session_state.show_loader = True
    st.session_state.results_ready = False
    st.rerun()

# ---- Display Results ----
if st.session_state.results_ready and 'results_df' in st.session_state:
    df = st.session_state.results_df

    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Matches", len(df))
    with col2:
        st.metric("Average Score", f"{df['Score'].mean():.1%}")
    with col3:
        st.metric("Jobs with Matches", df['Job'].nunique())
    with col4:
        st.metric("Matched Candidates", df['Resume'].nunique())

    # Results tabs
    tab1, tab2, tab3 = st.tabs(["📋 Results", "📊 Statistics", "🔍 Detailed View"])

    with tab1:
        # Search functionality
        search_query = st.text_input("🔍 Search results")

        display_df = df.copy()
        if search_query:
            mask = df.apply(lambda row: search_query.lower() in str(row).lower(), axis=1)
            display_df = df[mask]

        # Display results in expandable format
        st.subheader(f"Showing {len(display_df)} matches")

        for idx, row in display_df.iterrows():
            with st.expander(f"**{row['Job']}** ← **{row['Resume']}** (Score: {row['Score']:.1%})"):
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.metric("Match Score", f"{row['Score']:.1%}")
                with col2:
                    st.markdown("**הסבר:**")
                    st.markdown(
                        f"<div style='direction: rtl; text-align: right; background-color: #16213e; padding: 10px; border-radius: 5px;'>{row['Explanation']}</div>",
                        unsafe_allow_html=True)

        # Download button
        csv = df.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            "⬇️ Download Results (CSV)",
            data=csv,
            file_name='cv_job_matches.csv',
            mime='text/csv'
        )

    with tab2:
        if show_stats:
            # Score distribution
            st.subheader("Score Distribution")
            # Filter only scores above threshold (usually 0.7)
            filtered_scores = df[df['Score'] >= 0.7]['Score']
            if len(filtered_scores) > 0:
                score_bins = pd.cut(filtered_scores, bins=[0.7, 0.8, 0.9, 1.0], labels=['70-80%', '80-90%', '90-100%'])
                st.bar_chart(score_bins.value_counts().sort_index())
            else:
                st.info("No matches above 70% threshold")

            # Top jobs by match count
            st.subheader("Top Jobs by Number of Matches")
            job_counts = df['Job'].value_counts().head(10)
            st.bar_chart(job_counts)

            # Top resumes by average score
            st.subheader("Top Candidates by Average Score")
            resume_scores = df.groupby('Resume')['Score'].mean().sort_values(ascending=False).head(10)
            st.bar_chart(resume_scores)

    with tab3:
        # Job selector
        selected_job = st.selectbox("Select a job to view matches", sorted(df['Job'].unique()))

        if selected_job:
            job_matches = df[df['Job'] == selected_job].sort_values('Score', ascending=False)
            st.subheader(f"Matches for: {selected_job}")

            for _, match in job_matches.iterrows():
                st.markdown(f"""
                <div style='background-color: #16213e; padding: 1rem; border-radius: 10px; margin-bottom: 1rem; border-left: 4px solid #e94560;'>
                    <h4>📄 {match['Resume']}</h4>
                    <p><strong>Score:</strong> {match['Score']:.1%}</p>
                    <div style='direction: rtl; text-align: right; margin-top: 10px;'>
                        <strong>הסבר:</strong><br>
                        {match['Explanation']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

# ---- Footer ----
st.markdown(
    "<div style='position: fixed; bottom: 20px; right: 20px; background-color: #16213e; padding: 10px 20px; border-radius: 20px; border: 1px solid #e94560;'>Built with ❤️ by NexCruiTech</div>",
    unsafe_allow_html=True
)