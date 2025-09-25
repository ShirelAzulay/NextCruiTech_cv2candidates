import streamlit as st
import pandas as pd
import openai
from openai import OpenAI
import base64
import time
import os
import json
import re

from src.embeddings import load_embeddings, clean_old_embeddings, save_embeddings
from src.indexer import build_faiss_index
from src.matcher import get_topk_matches, gpt_filter_topk

# ---- Configuration ----
openai.api_key = "sk-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"  # Replace with your OpenAI API key
RESUMES_FOLDER = "resumes/"
JOBS_FOLDER = "jobs/"

# ---- Page Setup ----
st.set_page_config(
    page_title="CV-to-Job Algorithm",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- Enhanced Dark Theme with RTL Support ----
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Rubik:wght@300;400;500;700&display=swap');

    .stApp {
        background-color: #1a1a2e;
        color: #ffffff;
        font-family: 'Rubik', sans-serif;
    }

    /* RTL Support for Hebrew */
    .rtl-content {
        direction: rtl;
        text-align: right;
        font-family: 'Rubik', sans-serif;
        line-height: 1.8;
    }

    /* Better Hebrew Display */
    .hebrew-analysis {
        direction: rtl;
        text-align: right;
        background-color: #16213e;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #2a2a4e;
        margin: 10px 0;
        font-size: 16px;
        line-height: 1.8;
        white-space: pre-wrap;
        font-family: 'Rubik', sans-serif;
    }

    .hebrew-analysis h3, .hebrew-analysis h4 {
        color: #e94560;
        margin: 15px 0 10px 0;
    }

    .hebrew-analysis ul {
        list-style-position: inside;
        padding-right: 20px;
        padding-left: 0;
    }

    .hebrew-analysis li {
        margin: 5px 0;
    }

    /* Emoji support */
    .emoji {
        font-family: "Segoe UI Emoji", "Apple Color Emoji", sans-serif;
    }

    section[data-testid="stSidebar"] {
        background-color: #0f0f1e;
    }

    h1, h2, h3 {
        color: #ffffff !important;
    }

    [data-testid="metric-container"] {
        background-color: #16213e;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e94560;
    }

    /* Button Styles */
    .stButton > button {
        background-color: #e94560;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: 500;
        transition: all 0.3s;
    }

    .stButton > button:hover {
        background-color: #c93652;
        transform: translateY(-2px);
    }

    /* Success button style */
    .success-button > button {
        background-color: #4CAF50 !important;
    }

    .success-button > button:hover {
        background-color: #45a049 !important;
    }

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
@st.cache_data
def load_cached_embeddings():
    """Load embeddings with caching - only reload if file changes"""
    return load_embeddings()


def get_base64_of_image(image_path):
    """Convert image to base64 string"""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except:
        return None


def extract_recommendation(explanation_text):
    """Extract recommendation from Hebrew analysis text"""
    if not explanation_text:
        return "unknown"

    # Positive indicators
    positive_patterns = [
        "כדאי לראיין",
        "מומלץ לראיין",
        "מומלץ",
        "recommended",
        "כן לראיין",
        "בהחלט כדאי",
        "מתאים לתפקיד",
        "החלטה: כדאי"
    ]

    # Negative indicators
    negative_patterns = [
        "לא כדאי",
        "לא מומלץ",
        "לא לראיין",
        "not recommended",
        "פערים קריטיים",
        "החלטה: לא"
    ]

    text_lower = explanation_text.lower()

    # Check positive first
    for pattern in positive_patterns:
        if pattern in text_lower:
            return "recommended"

    # Then check negative
    for pattern in negative_patterns:
        if pattern in text_lower:
            return "not_recommended"

    return "neutral"


def format_hebrew_text(text):
    """Format Hebrew text for better display"""
    if not text:
        return ""

    # Clean up the text
    text = text.strip()

    # Fix common formatting issues
    text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single
    text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)  # Multiple newlines to double

    # Ensure proper RTL marks
    if any('\u0590' <= c <= '\u05FF' for c in text):  # Hebrew characters
        text = '\u202B' + text + '\u202C'  # Add RTL embedding

    return text


def get_detailed_explanation(client, job_text, resume_text, filter_prompt, score):
    """Generate detailed Hebrew analysis - cached per combination"""
    cache_key = f"explanation_{hash(job_text + resume_text + str(filter_prompt))}"

    if cache_key in st.session_state:
        return st.session_state[cache_key]

    prompt = f"""
אתה מומחה משאבי אנוש עם יכולות NLP מתקדמות. נתח לעומק את ההתאמה בין המועמד למשרה.

🛡️ חשוב: שמור על אובייקטיביות מלאה. התמקד אך ורק ביכולות מקצועיות.

הנחיות לניתוח:
1. נתח את איכות ההתאמה (לא רק קיום הדרישות)
2. זהה התאמות חכמות (לדוגמה: GKE = Kubernetes)
3. הדגש פערים משמעותיים וכאלו שניתן לגשר
4. זהה יתרונות ייחודיים שמוסיפים ערך
5. תן המלצה חדה ומנומקת

🚨 בדוק Red Flags:
- פערי זמן לא מוסברים
- החלפת עבודות תכופה
- חוסר התקדמות בקריירה
- ניסיון מנופח או לא עקבי
- אי התאמה בין רמת תפקיד לשנות ניסיון

📌 פורמט התשובה:
==================
🎯 רמת התפקיד: <Junior/Mid/Senior/Lead> | סוג חברה: <Startup/Enterprise/Other>
==================
🚨 Red Flags: <רשימה או "לא זוהו">
==================
✅ התאמות חזקות:
• <נקודות חוזק עיקריות בהתאמה>

❌ פערים עיקריים:
• קריטי (חוסם): <פערים מונעי תפקיד>
• משמעותי (ניתן לגשר): <דורש הכשרה>
• קל (למידה מהירה): <ניתן לסגור במהרה>

🚀 יתרונות ייחודיים:
• <ערכים מוספים מעבר לדרישות>

💡 המלצה סופית:
• החלטה: <כדאי לראיין/לא כדאי/בתנאי>
• רמת ביטחון: <גבוהה/בינונית/נמוכה>
• נקודות לחקירה בראיון: <שאלות מומלצות>
==================

קורות חיים:
{resume_text[:2000]}

דרישות המשרה:
{job_text[:1500]}

{f"קריטריונים נוספים מהמסנן: {filter_prompt}" if filter_prompt and filter_prompt.strip() else ""}

הציון הנוכחי מהמערכת: {score:.1%}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1500
        )
        explanation = response.choices[0].message.content.strip()
        explanation = format_hebrew_text(explanation)
        st.session_state[cache_key] = explanation
        return explanation
    except Exception as e:
        return f"שגיאה בניתוח: {str(e)}\n\nציון מערכת: {score:.1%}"


def filter_embeddings_by_folder(embeddings, folder_name):
    """Filter embeddings by folder prefix"""
    prefix = f"{folder_name}|"
    return {k: v for k, v in embeddings.items() if k.startswith(prefix)}


def check_embeddings_exist():
    """Check if we have sufficient embeddings to run"""
    if not os.path.exists("embeddings.json"):
        return False, "No embeddings file found"

    embeddings = load_cached_embeddings()
    resume_count = sum(1 for k in embeddings if k.startswith("resumes|"))
    job_count = sum(1 for k in embeddings if k.startswith("jobs|"))

    if resume_count == 0:
        return False, f"No resume embeddings found"
    if job_count == 0:
        return False, f"No job embeddings found"

    return True, f"{resume_count} resumes, {job_count} jobs ready"


def get_quick_recommendation(score, threshold=0.7):
    """
    Quick recommendation based on embedding similarity score alone.
    This is INSTANT - no API calls needed!
    """
    if score >= 0.85:
        return "highly_recommended", "✅ מומלץ מאוד - התאמה גבוהה"
    elif score >= 0.75:
        return "recommended", "✅ מומלץ לראיון"
    elif score >= threshold:
        return "maybe", "🔸 שקול לראיון - התאמה בינונית"
    else:
        return "not_recommended", "❌ לא מומלץ - התאמה נמוכה"


def run_matching_process(threshold, top_k, filter_prompt, lazy_explanations=True, auto_analyze=False,
                         use_quick_recommendation=True):
    """
    Run the matching process - separated for reusability
    auto_analyze: if True, generate all explanations immediately
    use_quick_recommendation: if True, use embedding scores for quick recommendations
    """
    client = OpenAI(api_key=openai.api_key)

    # Load embeddings
    all_embeddings = load_cached_embeddings()
    res_embeddings = filter_embeddings_by_folder(all_embeddings, "resumes")
    job_embeddings = filter_embeddings_by_folder(all_embeddings, "jobs")

    # Build FAISS index
    index, resume_ids = build_faiss_index(res_embeddings)

    # Match jobs to resumes
    results = []
    total_jobs = len(job_embeddings)

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, (job_id, job_data) in enumerate(job_embeddings.items()):
        # Update progress
        progress = (i + 1) / total_jobs
        progress_bar.progress(progress)

        job_name = job_id.split('|')[1] if '|' in job_id else job_id
        status_text.text(f"🔍 Processing: {job_name} ({i + 1}/{total_jobs})")

        job_emb = job_data["embedding"]
        job_text = job_data.get("text", "")

        # Get top K matches - THIS IS FAST (milliseconds)
        candidates = get_topk_matches(index, res_embeddings, resume_ids, job_emb, top_k=top_k)

        # Apply GPT filter ONLY if provided and not using quick mode
        if filter_prompt and filter_prompt.strip() and not use_quick_recommendation:
            try:
                candidates = gpt_filter_topk(client, candidates, filter_prompt)
            except Exception as e:
                st.warning(f"Filter failed for {job_name}: {str(e)}")

        # Store matches above threshold
        for item in candidates:
            if item["score"] >= threshold:
                resume_text = item["resume_text"]
                resume_name = item["resume_id"].split('|')[1] if '|' in item["resume_id"] else item["resume_id"]

                # Quick recommendation based on score
                if use_quick_recommendation:
                    recommendation, rec_text = get_quick_recommendation(item["score"], threshold)
                    explanation = f"המלצה מהירה על בסיס התאמת Embeddings: {rec_text}\nציון התאמה: {item['score']:.1%}"
                else:
                    # Generate full explanation only if needed
                    explanation = ""
                    recommendation = "unknown"

                    if not lazy_explanations or auto_analyze:
                        explanation = get_detailed_explanation(
                            client, job_text, resume_text, filter_prompt, item["score"]
                        )
                        recommendation = extract_recommendation(explanation)

                results.append({
                    "Job": job_name,
                    "Resume": resume_name,
                    "Score": item["score"],
                    "Explanation": explanation,
                    "job_text": job_text,
                    "resume_text": resume_text,
                    "recommendation": recommendation,
                    "quick_mode": use_quick_recommendation
                })

    progress_bar.empty()
    status_text.empty()

    return pd.DataFrame(results) if results else None


# ---- Initialize session state ----
if 'results_ready' not in st.session_state:
    st.session_state.results_ready = False
if 'show_only_recommended' not in st.session_state:
    st.session_state.show_only_recommended = False
if 'include_details_in_csv' not in st.session_state:
    st.session_state.include_details_in_csv = True

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

# ---- Check embeddings status ----
embeddings_ok, embeddings_msg = check_embeddings_exist()

if not embeddings_ok:
    st.error(f"❌ {embeddings_msg}")
    st.info("💡 Please run the embedding generation first by processing files in the 'resumes/' and 'jobs/' folders")
    st.stop()

st.success(f"✅ Embeddings ready: {embeddings_msg}")

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

    st.markdown("### 🚀 Speed vs Quality Trade-off")
    analysis_mode = st.radio(
        "Choose analysis mode:",
        ["⚡ Quick Mode (Instant)", "🤖 Full AI Analysis (Slower)"],
        index=0,
        help="Quick Mode uses embedding scores only. Full Analysis uses GPT for detailed insights."
    )

    use_quick_mode = analysis_mode.startswith("⚡")

    if not use_quick_mode:
        lazy_explanations = st.checkbox("⚡ Load explanations on demand", value=True, help="Faster initial results")
        auto_analyze_recommended = st.checkbox("🤖 Auto-analyze all candidates", value=False,
                                               help="Generate analysis for all candidates immediately (slower but shows recommendations)")
    else:
        lazy_explanations = True
        auto_analyze_recommended = False
        st.info(
            "💡 Quick Mode: Instant recommendations based on similarity scores. Click on candidates for detailed AI analysis.")

    st.session_state.include_details_in_csv = st.checkbox("📝 Include detailed analysis in CSV export", value=True)

    run_button = st.button("🚀 Run Matching", type="primary", use_container_width=True)

    # Quick rerun for threshold/filter changes
    if st.session_state.results_ready:
        st.markdown("---")
        st.markdown("### ⚡ Quick Update")
        st.info("Change threshold or filter and click below for instant update (no re-embedding)")
        quick_update_button = st.button("🔄 Update Results", use_container_width=True)

        if quick_update_button:
            with st.spinner('🔍 Updating results...'):
                df = run_matching_process(
                    threshold, top_k, filter_prompt,
                    lazy_explanations and not auto_analyze_recommended,
                    auto_analyze_recommended,
                    use_quick_mode
                )
                if df is not None:
                    st.session_state.results_df = df
                    st.session_state.lazy_mode = lazy_explanations
                    st.rerun()

    # Show embeddings stats
    embeddings = load_cached_embeddings()
    st.markdown("---")
    st.subheader("📊 Data Status")
    resume_count = sum(1 for k in embeddings if k.startswith("resumes|"))
    job_count = sum(1 for k in embeddings if k.startswith("jobs|"))
    st.markdown(f"**📄 Resumes:** {resume_count}")
    st.markdown(f"**💼 Jobs:** {job_count}")

    if st.button("🧹 Clean Old Embeddings"):
        with st.spinner("Cleaning..."):
            original_count = len(embeddings)
            embeddings = clean_old_embeddings(embeddings, RESUMES_FOLDER)
            embeddings = clean_old_embeddings(embeddings, JOBS_FOLDER)
            cleaned_count = original_count - len(embeddings)
            if cleaned_count > 0:
                save_embeddings(embeddings)
                st.success(f"✅ Cleaned {cleaned_count} old embeddings")
                st.rerun()
            else:
                st.info("No old embeddings to clean")

# ---- Run matching process ----
if run_button:
    with st.spinner('🔍 Matching resumes to jobs...'):
        df = run_matching_process(
            threshold, top_k, filter_prompt,
            lazy_explanations and not auto_analyze_recommended,
            auto_analyze_recommended,
            use_quick_mode
        )

        if df is not None:
            st.session_state.results_df = df
            st.session_state.results_ready = True
            st.session_state.lazy_mode = lazy_explanations and not auto_analyze_recommended
            st.session_state.quick_mode = use_quick_mode
            time.sleep(0.5)
            st.rerun()
        else:
            st.error("No matches found! Try adjusting the threshold or filter criteria.")

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
        # Count recommended candidates
        recommended_count = len(df[df['recommendation'] == 'recommended'])
        st.metric("Recommended to Interview", recommended_count)

    # Results tabs
    tab1, tab2, tab3 = st.tabs(["📋 Results", "📊 Statistics", "🔍 Detailed View"])

    with tab1:
        # Filter controls
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            search_query = st.text_input("🔍 Search results")
        with col2:
            if st.button("✅ Show Only Recommended",
                         type="primary" if not st.session_state.show_only_recommended else "secondary",
                         use_container_width=True):
                # First generate all missing analyses
                if not st.session_state.show_only_recommended:
                    with st.spinner('🔍 מנתח את כל המועמדים...'):
                        client = OpenAI(api_key=openai.api_key)
                        progress_bar = st.progress(0)

                        # Count how many need analysis
                        need_analysis = df[df['Explanation'] == ''].index
                        total_to_analyze = len(need_analysis)

                        if total_to_analyze > 0:
                            for i, idx in enumerate(need_analysis):
                                row = df.loc[idx]
                                # Update progress
                                progress_bar.progress((i + 1) / total_to_analyze)

                                # Generate explanation
                                explanation = get_detailed_explanation(
                                    client, row['job_text'], row['resume_text'],
                                    filter_prompt, row['Score']
                                )
                                # Update dataframe
                                st.session_state.results_df.loc[idx, 'Explanation'] = explanation
                                # Update recommendation
                                rec = extract_recommendation(explanation)
                                st.session_state.results_df.loc[idx, 'recommendation'] = rec

                        progress_bar.empty()

                st.session_state.show_only_recommended = not st.session_state.show_only_recommended
                st.rerun()
        with col3:
            if st.session_state.show_only_recommended:
                if st.button("🔄 Show All", use_container_width=True):
                    st.session_state.show_only_recommended = False
                    st.rerun()

        # Apply filters
        display_df = df.copy()

        # Apply recommendation filter
        if st.session_state.show_only_recommended:
            # Filter by recommendation status
            if st.session_state.get('quick_mode', False):
                # In quick mode, filter by score-based recommendations
                recommended_df = display_df[display_df['recommendation'].isin(['highly_recommended', 'recommended'])]
            else:
                # First try to filter by existing recommendations
                recommended_df = display_df[display_df['recommendation'] == 'recommended']

                # If no recommendations found, analyze explanations
                if len(recommended_df) == 0 and len(display_df) > 0:
                    st.info("🔍 Analyzing recommendations...")
                    client = OpenAI(api_key=openai.api_key)
                    for idx, row in display_df.iterrows():
                        if not row['Explanation'] or row['quick_mode']:
                            # Generate full explanation if missing or in quick mode
                            explanation = get_detailed_explanation(
                                client, row['job_text'], row['resume_text'], filter_prompt, row['Score']
                            )
                            display_df.loc[idx, 'Explanation'] = explanation
                            st.session_state.results_df.loc[idx, 'Explanation'] = explanation

                        # Update recommendation status
                        rec_status = extract_recommendation(display_df.loc[idx, 'Explanation'])
                        display_df.loc[idx, 'recommendation'] = rec_status
                        st.session_state.results_df.loc[idx, 'recommendation'] = rec_status

                    recommended_df = display_df[display_df['recommendation'] == 'recommended']

            display_df = recommended_df
            st.success(f"🎯 Showing {len(display_df)} recommended candidates")

        # Apply search filter
        if search_query:
            mask = display_df.apply(lambda row: search_query.lower() in str(row).lower(), axis=1)
            display_df = display_df[mask]

        # Display results
        st.subheader(f"Showing {len(display_df)} matches")

        for idx, row in display_df.iterrows():
            # Determine color based on recommendation
            border_color = "#4CAF50" if row['recommendation'] == 'recommended' else "#e94560"

            with st.expander(f"**{row['Job']}** ← **{row['Resume']}** (Score: {row['Score']:.1%})"):
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.metric("Match Score", f"{row['Score']:.1%}")
                    if row['recommendation'] == 'recommended':
                        st.success("✅ Recommended")
                    elif row['recommendation'] == 'not_recommended':
                        st.error("❌ Not Recommended")

                with col2:
                    # Show explanation or load on demand
                    if row.get('quick_mode', False) and '(Instant)' in analysis_mode:
                        # In quick mode, show quick recommendation
                        st.markdown("**המלצה מהירה:**")
                        st.info(row['Explanation'])
                        if st.button(f"🔍 בקש ניתוח AI מפורט", key=f"analyze_{idx}"):
                            client = OpenAI(api_key=openai.api_key)
                            explanation = get_detailed_explanation(
                                client, row['job_text'], row['resume_text'], filter_prompt, row['Score']
                            )
                            # Update dataframe
                            st.session_state.results_df.loc[idx, 'Explanation'] = explanation
                            st.session_state.results_df.loc[idx, 'quick_mode'] = False
                            # Update recommendation
                            rec = extract_recommendation(explanation)
                            st.session_state.results_df.loc[idx, 'recommendation'] = rec
                            st.rerun()
                    elif st.session_state.get('lazy_mode', False) and not row['Explanation']:
                        if st.button(f"🔍 הצג ניתוח מפורט", key=f"analyze_{idx}"):
                            # Initialize client
                            client = OpenAI(api_key=openai.api_key)
                            explanation = get_detailed_explanation(
                                client, row['job_text'], row['resume_text'], filter_prompt, row['Score']
                            )
                            # Update dataframe
                            st.session_state.results_df.loc[idx, 'Explanation'] = explanation
                            # Update recommendation
                            rec = extract_recommendation(explanation)
                            st.session_state.results_df.loc[idx, 'recommendation'] = rec
                            st.rerun()
                    else:
                        st.markdown("**ניתוח מתקדם:**")
                        explanation_text = row['Explanation'] if row[
                            'Explanation'] else "לחץ על 'הצג ניתוח מפורט' למעלה"
                        st.markdown(
                            f'<div class="hebrew-analysis">{explanation_text}</div>',
                            unsafe_allow_html=True
                        )

        # Download button with options
        st.markdown("---")
        col1, col2 = st.columns([3, 1])
        with col1:
            # Prepare CSV based on settings
            if st.session_state.include_details_in_csv:
                csv_df = df[['Job', 'Resume', 'Score', 'Explanation', 'recommendation']]
            else:
                csv_df = df[['Job', 'Resume', 'Score', 'recommendation']]

            csv = csv_df.to_csv(index=False, encoding='utf-8-sig')

            st.download_button(
                "⬇️ Download Results (CSV)",
                data=csv,
                file_name='cv_job_matches.csv',
                mime='text/csv'
            )
        with col2:
            st.info(f"CSV will {'include' if st.session_state.include_details_in_csv else 'exclude'} detailed analysis")

    with tab2:
        if show_stats:
            # Recommendation distribution
            st.subheader("📊 Recommendation Distribution")
            rec_counts = df['recommendation'].value_counts()

            # Create a cleaner display
            rec_display = {
                'recommended': '✅ Recommended for Interview',
                'not_recommended': '❌ Not Recommended',
                'neutral': '🔸 Neutral/Unclear',
                'unknown': '❓ Not Analyzed Yet'
            }

            rec_data = {}
            for key, label in rec_display.items():
                if key in rec_counts:
                    rec_data[label] = rec_counts[key]

            if rec_data:
                st.bar_chart(pd.Series(rec_data))

            # Score distribution
            st.subheader("📈 Score Distribution")
            filtered_scores = df[df['Score'] >= threshold]['Score']
            if len(filtered_scores) > 0:
                score_bins = pd.cut(filtered_scores, bins=[threshold, 0.8, 0.9, 1.0],
                                    labels=[f'{threshold:.0%}-80%', '80-90%', '90-100%'])
                st.bar_chart(score_bins.value_counts().sort_index())
            else:
                st.info(f"No matches above {threshold:.0%} threshold")

            # Top jobs and candidates
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Top Jobs by Matches")
                job_counts = df['Job'].value_counts().head(10)
                st.bar_chart(job_counts)

            with col2:
                st.subheader("Top Candidates by Avg Score")
                resume_scores = df.groupby('Resume')['Score'].mean().sort_values(ascending=False).head(10)
                st.bar_chart(resume_scores)

    with tab3:
        # Job selector
        selected_job = st.selectbox("Select a job to view matches", sorted(df['Job'].unique()))

        if selected_job:
            job_matches = df[df['Job'] == selected_job].sort_values('Score', ascending=False)
            st.subheader(f"Matches for: {selected_job}")

            for _, match in job_matches.iterrows():
                # Determine styling based on recommendation
                if match['recommendation'] == 'recommended':
                    border_style = "border-left: 4px solid #4CAF50;"
                    rec_badge = '<span style="color: #4CAF50;">✅ מומלץ לראיון</span>'
                elif match['recommendation'] == 'not_recommended':
                    border_style = "border-left: 4px solid #e94560;"
                    rec_badge = '<span style="color: #e94560;">❌ לא מומלץ</span>'
                else:
                    border_style = "border-left: 4px solid #ffa500;"
                    rec_badge = '<span style="color: #ffa500;">🔸 נייטרלי</span>'

                # Display match with proper Hebrew formatting
                explanation_text = match['Explanation'] if match[
                    'Explanation'] else 'לחץ על "הצג ניתוח מפורט" בטאב Results כדי לראות ניתוח'

                st.markdown(f"""
                <div style='background-color: #16213e; padding: 1rem; border-radius: 10px; margin-bottom: 1rem; {border_style}'>
                    <h4>📄 {match['Resume']} {rec_badge}</h4>
                    <p><strong>Score:</strong> {match['Score']:.1%}</p>
                    <div class='hebrew-analysis' style='margin-top: 10px;'>
                        <strong>ניתוח מפורט:</strong><br>
                        {explanation_text}
                    </div>
                </div>
                """, unsafe_allow_html=True)

# ---- Footer ----
st.markdown(
    "<div style='position: fixed; bottom: 20px; right: 20px; background-color: #16213e; padding: 10px 20px; border-radius: 20px; border: 1px solid #e94560;'>Built with ❤️ by NexCruiTech</div>",
    unsafe_allow_html=True
)