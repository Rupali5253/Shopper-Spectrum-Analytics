import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Set page configuration for a modern wide professional layout
st.set_page_config(page_title="E-Commerce Insights Platform", layout="wide", initial_sidebar_state="expanded")

# Load saved binary models and tables with caching to prevent load overheads
@st.cache_resource
def load_project_artifacts():
    with open('kmeans_model.pkl', 'rb') as f:
        kmeans_model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        data_scaler = pickle.load(f)
    with open('item_similarity.pkl', 'rb') as f:
        similarity_matrix = pickle.load(f)
    return kmeans_model, data_scaler, similarity_matrix

kmeans, scaler, item_sim_df = load_project_artifacts()

# Sidebar high-fidelity navigation controls using raw text layouts
st.sidebar.title("Navigation")
st.sidebar.markdown("---")
page_selection = st.sidebar.radio("Go to:", ["Home Dashboard", "Customer Segmentation", "Product Recommender"])
st.sidebar.markdown("---")
st.sidebar.caption("Operational Intelligence System v1.0")

# --- PANEL 1: HOME PLATFORM INTRO ---
if page_selection == "Home Dashboard":
    st.title("🛍️ E-Commerce Customer Analytics Platform")
    st.markdown("Welcome to the executive data command center. This system bridges production operational matrices into predictive client management modules using two foundational pillars:")

    st.info("**🎯 Customer Segmentation Module:** Automates raw client tier categorization based on mathematical Recency, Frequency, and Monetary mathematical limits to separate core high-value capital streams from at-risk accounts.")
    st.success("**📦 Product Recommendation Module:** Deploys high-speed sparse item-based collaborative computing layers to track concurrent product matches and generate top 5 alternate stock targets immediately.")

# --- PANEL 2: CUSTOMER SEGMENTATION ENGINE ---
elif page_selection == "Customer Segmentation":
    st.title("🎯 Customer Segmentation Module")
    st.subheader("Predict Strategic Customer Tiers")
    st.markdown("---")

    # 3-Column horizontal numerical controller layout
    col1, col2, col3 = st.columns(3)
    with col1:
        recency_val = st.number_input("Recency (Days since last purchase)", min_value=0, value=30, step=1)
    with col2:
        frequency_val = st.number_input("Frequency (Total number of purchases)", min_value=1, value=5, step=1)
    with col3:
        monetary_val = st.number_input("Monetary (Total spending value in £)", min_value=0.0, value=100.0, step=10.0)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Predict Segment", use_container_width=True):
        # Format input vector, apply variance standard scaling, and execute cluster prediction
        raw_input = np.array([[recency_val, frequency_val, monetary_val]])
        scaled_input = scaler.transform(raw_input)
        predicted_cluster = kmeans.predict(scaled_input)[0]

        # Hard-coded task configuration label indices mapping
        cluster_map = {2: 'High-Value', 3: 'Regular', 0: 'Occasional', 1: 'At-Risk'}
        assigned_label = cluster_map.get(predicted_cluster, "Undefined")

        st.success(f"Cluster Assignment Group Number: **{predicted_cluster}**")
        st.warning(f"This customer belongs to: **{assigned_label}** Shopper Tier")

# --- PANEL 3: PRODUCT RECOMMENDER ENGINE ---
elif page_selection == "Product Recommender":
    st.title("📦 Product Recommender")
    st.subheader("Item-Based Collaborative Filtering Suggestions")
    st.markdown("---")

    # Auto-populate a safe selectable dropdown block using our matrix indices titles
    all_products = list(item_sim_df.index)
    target_product = st.selectbox("Search or Select an Exact Product Name:", all_products)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Get Recommendations", use_container_width=True):
        if target_product in item_sim_df.columns:
            # Pick indices 1 through 6 to skip the item matching perfectly with itself
            top_matches = item_sim_df[target_product].sort_values(ascending=False).iloc[1:6]

            st.markdown("### **Recommended Products:**")

            # Print entries row-by-row inside high-contrast informational container chips
            for ranking, (item_name, similarity_score) in enumerate(top_matches.items(), 1):
                st.info(f"**{ranking}. {item_name}** — *Match Score: {similarity_score:.2%}*")
        else:
            st.error("The selected title profile contains an index system collision. Please choose an alternate item.")
