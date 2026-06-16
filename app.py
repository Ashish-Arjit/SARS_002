# ==========================================
# Streamlit Application (app.py)
# Purpose: Interactive web interface for the Crop Recommendation System.
# ==========================================

import os
import joblib
import pandas as pd
import streamlit as st

# Configure the page settings (title, icon, layout)
st.set_page_config(
    page_title="AgroSuggest - Smart Crop Recommendation System",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling using CSS
st.markdown("""
    <style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    /* Apply Font Family globally */
    * {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Main container background gradient */
    .stApp {
        background: linear-gradient(135deg, #f4f7f6 0%, #e8efec 100%);
    }
    
    /* Premium Title Banner */
    .banner {
        background: linear-gradient(90deg, #1e3f20 0%, #2e7d32 100%);
        color: white;
        padding: 2.5rem 2rem;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.08);
    }
    
    .banner h1 {
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        color: #ffffff !important;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    .banner p {
        font-size: 1.1rem;
        font-weight: 300;
        opacity: 0.9;
        margin: 0;
    }
    
    /* Sidebar styling override */
    [data-testid="stSidebar"] {
        background-color: #1e3f20;
        color: white;
    }
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Card styling */
    .card {
        background-color: white;
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        border: 1px solid rgba(46, 125, 50, 0.1);
        margin-bottom: 1.5rem;
    }
    
    .card-title {
        color: #1e3f20;
        font-size: 1.4rem;
        font-weight: 600;
        margin-bottom: 1rem;
        border-bottom: 2px solid #2e7d32;
        padding-bottom: 0.5rem;
    }
    
    /* Recommendation Output Cards */
    .success-card {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        color: #1b5e20;
        padding: 2rem;
        border-radius: 16px;
        border-left: 8px solid #2e7d32;
        box-shadow: 0 8px 20px rgba(46, 125, 50, 0.15);
        margin-top: 1.5rem;
        text-align: center;
    }
    
    .success-crop {
        font-size: 3rem;
        font-weight: 800;
        color: #1b5e20;
        margin: 0.5rem 0;
        text-transform: capitalize;
    }
    
    .success-conf {
        font-size: 1.2rem;
        font-weight: 600;
        color: #2e7d32;
    }
    
    /* Tooltip / description style */
    .metric-desc {
        font-size: 0.85rem;
        color: #666;
        margin-top: -8px;
        margin-bottom: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# Helper function to load model
@st.cache_resource
def get_model():
    """
    Loads the saved model and caches it to prevent reloading on every run.
    Uses dynamic path resolution to work locally and on Streamlit Cloud.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "models", "crop_model.pkl")
    
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

# Render custom header
st.markdown("""
    <div class="banner">
        <h1>🌱 Precision Agriculture Crop Recommendation</h1>
        <p>Analyze soil nutrients & climate conditions to recommend the most optimal crop using Machine Learning</p>
    </div>
""", unsafe_allow_html=True)

# Initialize tabs
tab_predict, tab_insights = st.tabs(["🔮 Crop Predictor", "📊 Data Insights & Plots"])

# TAB 1: CROP PREDICTOR
with tab_predict:
    st.markdown("""
        <div class="card">
            <div class="card-title">🔬 Soil & Weather Diagnostics</div>
            <p>Please adjust the sliders or enter the soil chemical composition and climatic parameters below. 
            The Random Forest model will evaluate these inputs and suggest the best-suited crop.</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Create inputs layout (2 columns for soil, 2 columns for weather/climate)
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🪱 Soil Nutrients (NPK)")
        
        n_input = st.number_input(
            "Nitrogen (N) Content in Soil", 
            min_value=0, max_value=200, value=90, step=1,
            help="High nitrogen is essential for leaf growth. Standard range: 0-140"
        )
        st.markdown('<p class="metric-desc">Ratio of Nitrogen content in soil (mg/kg)</p>', unsafe_allow_html=True)
        
        p_input = st.number_input(
            "Phosphorus (P) Content in Soil", 
            min_value=0, max_value=200, value=42, step=1,
            help="Phosphorus helps in root and flower development. Standard range: 0-145"
        )
        st.markdown('<p class="metric-desc">Ratio of Phosphorus content in soil (mg/kg)</p>', unsafe_allow_html=True)
        
        k_input = st.number_input(
            "Potassium (K) Content in Soil", 
            min_value=0, max_value=300, value=43, step=1,
            help="Potassium enhances disease resistance and overall quality. Standard range: 0-205"
        )
        st.markdown('<p class="metric-desc">Ratio of Potassium content in soil (mg/kg)</p>', unsafe_allow_html=True)
        
        ph_input = st.slider(
            "Soil pH Level", 
            min_value=0.0, max_value=14.0, value=6.5, step=0.1,
            help="Measures soil acidity or alkalinity. 7.0 is neutral. Standard range: 3.5-9.9"
        )
        st.markdown('<p class="metric-desc">pH value of the soil (0.0 to 14.0)</p>', unsafe_allow_html=True)

    with col2:
        st.markdown("### ☁️ Climate Parameters")
        
        temp_input = st.slider(
            "Temperature (°C)", 
            min_value=0.0, max_value=50.0, value=20.0, step=0.1,
            help="Air temperature in the region. Standard range: 8.0°C - 44.0°C"
        )
        st.markdown('<p class="metric-desc">Average ambient temperature in Celsius</p>', unsafe_allow_html=True)
        
        humidity_input = st.slider(
            "Relative Humidity (%)", 
            min_value=0.0, max_value=100.0, value=82.0, step=0.1,
            help="Moisture level in the air. Standard range: 14% - 100%"
        )
        st.markdown('<p class="metric-desc">Percentage of moisture in the air</p>', unsafe_allow_html=True)
        
        rainfall_input = st.number_input(
            "Rainfall (mm)", 
            min_value=0.0, max_value=500.0, value=202.0, step=1.0,
            help="Average precipitation level. Standard range: 20mm - 300mm"
        )
        st.markdown('<p class="metric-desc">Annual/seasonal average rainfall (mm)</p>', unsafe_allow_html=True)

    # Perform prediction
    st.markdown("---")
    
    # Centered Recommendation Button
    btn_col_1, btn_col_2, btn_col_3 = st.columns([2, 1, 2])
    
    with btn_col_2:
        predict_button = st.button("🔍 Recommend Crop", use_container_width=True)
        
    if predict_button:
        # Load the model
        model = get_model()
        
        if model is None:
            st.error("Error: Trained model could not be loaded. Please ensure that 'models/crop_model.pkl' exists.")
        else:
            # Prepare inputs DataFrame
            input_df = pd.DataFrame([[
                n_input, p_input, k_input, temp_input, humidity_input, ph_input, rainfall_input
            ]], columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
            
            # Predict crop and confidence
            prediction = model.predict(input_df)[0]
            probabilities = model.predict_proba(input_df)
            crop_classes = list(model.classes_)
            class_idx = crop_classes.index(prediction)
            confidence = probabilities[0][class_idx]
            
            # Display output
            st.markdown(f"""
                <div class="success-card">
                    <h3>🌾 RECOMMENDED CROP</h3>
                    <div class="success-crop">{prediction}</div>
                    <div class="success-conf">Prediction Confidence: {confidence * 100:.1f}%</div>
                </div>
            """, unsafe_allow_html=True)
            
            # Provide explanation/guidelines based on predicted crop
            st.balloons()

# TAB 2: DATA INSIGHTS & PLOTS
with tab_insights:
    st.markdown("""
        <div class="card">
            <div class="card-title">📊 Dataset Statistics & Visualizations</div>
            <p>Below are the graphical reports generated during the training phase using Matplotlib. These graphs help understand the data patterns, correlations, and weather distributions.</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Sub-tabs/cols for images
    img_col1, img_col2 = st.columns(2)
    
    # Path settings
    current_dir = os.path.dirname(os.path.abspath(__file__))
    plots_path = os.path.join(current_dir, "dataset", "plots")
    
    with img_col1:
        st.subheader("1. Crop Counts Distribution")
        crop_count_img = os.path.join(plots_path, "crop_count_distribution.png")
        if os.path.exists(crop_count_img):
            st.image(crop_count_img, caption="Balanced distribution of classes: 100 samples per crop.", use_column_width=True)
        else:
            st.warning("Crop count visualization not found. Please run 'python src/data_visualization.py' to generate it.")
            
        st.subheader("2. Temperature Histogram")
        temp_img = os.path.join(plots_path, "temperature_distribution.png")
        if os.path.exists(temp_img):
            st.image(temp_img, caption="Frequency distribution of the ambient temperature across observations.", use_column_width=True)
        else:
            st.warning("Temperature histogram not found. Please run 'python src/data_visualization.py' to generate it.")

    with img_col2:
        st.subheader("3. Feature Correlation Heatmap")
        heatmap_img = os.path.join(plots_path, "correlation_heatmap.png")
        if os.path.exists(heatmap_img):
            st.image(heatmap_img, caption="Correlation matrix showing relationships between nutrients and weather conditions.", use_column_width=True)
        else:
            st.warning("Correlation heatmap not found. Please run 'python src/data_visualization.py' to generate it.")
            
        st.subheader("4. Rainfall Histogram")
        rainfall_img = os.path.join(plots_path, "rainfall_distribution.png")
        if os.path.exists(rainfall_img):
            st.image(rainfall_img, caption="Frequency distribution of the annual/seasonal rainfall across observations.", use_column_width=True)
        else:
            st.warning("Rainfall histogram not found. Please run 'python src/data_visualization.py' to generate it.")

# Footer content
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray; font-size: 0.8rem;'>Developed as a beginner-friendly precision agriculture demo. Algorithms used: Random Forest Classifier, Decision Tree Classifier.</p>", unsafe_allow_html=True)
