"""
Gurulearn QAAgent — Sentiment Intelligence Dashboard
Author: Gurudharsan (Guru)
"""

import streamlit as st
import pandas as pd
import traceback
import os
from gurulearn import QAAgent

# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(
    page_title="Gurulearn • Sentiment Intelligence",
    page_icon="💬",
    layout="wide"
)

# -------------------------------------------------
# Constants
# -------------------------------------------------
DEFAULT_CSV = "TestReviews.csv"

# -------------------------------------------------
# Styles (Custom CSS)
# -------------------------------------------------
st.markdown("""
<style>
    .main-title {
        font-size: 2.6rem;
        font-weight: 700;
        color: #4F46E5;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #6B7280;
        margin-bottom: 1rem;
    }
    .card {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 14px;
        box-shadow: 0px 8px 24px rgba(0,0,0,0.06);
        margin-bottom: 1.5rem;
    }
    .success-box {
        background: #ECFDF5;
        color: #065F46;
        padding: 1rem;
        border-radius: 10px;
        font-weight: 600;
    }
    .warning-box {
        background: #FEF3C7;
        color: #92400E;
        padding: 1rem;
        border-radius: 10px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# Helpers
# -------------------------------------------------
@st.cache_data
def load_csv(uploaded_file):
    if uploaded_file is None:
        if os.path.exists(DEFAULT_CSV):
            return pd.read_csv(DEFAULT_CSV)
        return None
    return pd.read_csv(uploaded_file)

# -------------------------------------------------
# Header
# -------------------------------------------------
st.markdown('<div class="main-title">QAAgent RAG</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Powered by Gurulearn QAAgent • Ask, Analyze & Classify Reviews</div>', unsafe_allow_html=True)

# -------------------------------------------------
# Sidebar — Setup
# -------------------------------------------------
with st.sidebar:
    st.header("⚙ Setup Assistant")

    uploaded = st.file_uploader("📂 Upload CSV (optional)", type=["csv"])
    df = load_csv(uploaded)

    if df is None:
        st.markdown('<div class="warning-box">No dataset loaded</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="success-box">Dataset Loaded Successfully</div>', unsafe_allow_html=True)
        st.metric("Total Records", len(df))

    st.markdown("---")
    st.subheader("🧠 QAAgent Config")

    llm_model = st.text_input("LLM Model", value="llama3.2")

    col_opts = list(df.columns) if df is not None else []
    page_content_fields = st.multiselect(
        "Text Columns",
        col_opts,
        default=[c for c in col_opts if "review" in c.lower()]
    )

    metadata_fields = st.multiselect(
        "Metadata Columns",
        col_opts,
        default=[c for c in col_opts if c.lower() in ("class", "label", "sentiment")]
    )

    system_prompt = st.text_area(
        "System Prompt",
        height=120,
        value="You are a helpful sentiment analysis assistant that classifies reviews and gives actionable insights."
    )

    init_button = st.button("🚀 Initialize QAAgent", use_container_width=True)

# -------------------------------------------------
# Session State
# -------------------------------------------------
if "agent" not in st.session_state:
    st.session_state.agent = None

# -------------------------------------------------
# Initialize Agent
# -------------------------------------------------
if init_button:
    if df is None:
        st.error("❌ Please load a dataset first")
    else:
        try:
            with st.spinner("Initializing QAAgent..."):
                st.session_state.agent = QAAgent(
                    data=df,
                    llm_model=llm_model,
                    page_content_fields=page_content_fields or [df.columns[0]],
                    metadata_fields=metadata_fields or [],
                    system_prompt=system_prompt
                )
            st.success("✅ QAAgent is ready!")
        except Exception:
            st.error("Initialization failed")
            st.code(traceback.format_exc())

# -------------------------------------------------
# Main Content
# -------------------------------------------------
if st.session_state.agent:
    agent = st.session_state.agent

    tab1, tab2, tab3 = st.tabs([
        "💬 Ask Questions",
        "🏷 Classify Review",
        "📊 Dataset Explorer"
    ])

    # -----------------------------
    # TAB 1 — QA Chat
    # -----------------------------
    with tab1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Ask Anything About the Reviews")

        query = st.text_input("Your Question", placeholder="e.g. What are customers complaining about?")
        if st.button("Ask AI", use_container_width=True):
            try:
                with st.spinner("Thinking..."):
                    response = agent.query(query)
                st.markdown("### 🤖 AI Response")
                st.write(response)
            except Exception:
                st.error("Query failed")
                st.code(traceback.format_exc())

        st.markdown('</div>', unsafe_allow_html=True)

    # -----------------------------
    # TAB 2 — Review Classification
    # -----------------------------
    with tab2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Classify a Single Review")

        review_text = st.text_area("Paste customer review here", height=150)
        if st.button("Classify Review", use_container_width=True):
            if not review_text.strip():
                st.warning("Please enter a review")
            else:
                try:
                    with st.spinner("Analyzing sentiment..."):
                        prompt = f"""
                        Classify the following review.
                        Return:
                        - Sentiment label
                        - Short explanation

                        Review:
                        {review_text}
                        """
                        result = agent.query(prompt)
                    st.markdown("### 📌 Result")
                    st.write(result)
                except Exception:
                    st.error("Classification failed")
                    st.code(traceback.format_exc())
        st.markdown('</div>', unsafe_allow_html=True)

    # -----------------------------
    # TAB 3 — Dataset Explorer
    # -----------------------------
    with tab3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Explore Dataset")

        if df is not None:
            st.dataframe(df.sample(min(50, len(df))), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.info("👈 Configure and initialize the QAAgent from the sidebar to begin.")

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown("---")
st.caption("Built with ❤ using Streamlit & Gurulearn QAAgent | © Guru")