import streamlit as st
import pandas as pd
import joblib

# Load your trained model once
@st.cache_resource
def load_model():
    return joblib.load('voice_gender_classifier_all_features.pkl')

model = load_model()

# --- Page 1: Project Introduction ---
def project_intro():
    st.title("🎯 Project Introduction")

    st.header("Problem Statement")
    st.write("""
    Develop a machine learning-based model to classify and cluster human voice samples based on extracted audio features. 
    The system will preprocess the dataset, apply clustering and classification models, and evaluate their performance. 
    The final application will provide an interface for uploading audio samples and receiving predictions, deployed via Streamlit.
    """)

    st.header("Business Use Cases")
    st.markdown("""
    - **🎙️ Speaker Identification**:  
      Identify individuals based on their voice features.
    
    - **🧑‍🤝‍🧑 Gender Classification**:  
      Classify voices as male or female for applications like call center analytics.
    
    - **📊 Speech Analytics**:  
      Extract insights from audio data for media, security, and customer service industries.
    
    - **♿ Assistive Technologies**:  
      Enhance accessibility by analyzing voice patterns for assistive tools.
    """)

# --- Page 2: Classifier (Voice Gender Prediction) ---
def gender_prediction():
    st.title("🔍 Voice Gender Classifier")

    st.write("Enter the extracted audio features below to predict the speaker's gender.")

    features = [
        "mean_spectral_centroid", "std_spectral_centroid", "mean_spectral_bandwidth", "std_spectral_bandwidth",
        "mean_spectral_contrast", "mean_spectral_flatness", "mean_spectral_rolloff", "zero_crossing_rate",
        "rms_energy", "mean_pitch", "min_pitch", "max_pitch", "std_pitch", "spectral_skew",
        "spectral_kurtosis", "energy_entropy", "log_energy"
    ]

    for i in range(1, 14):
        features.append(f"mfcc_{i}_mean")
    for i in range(1, 14):
        features.append(f"mfcc_{i}_std")

    input_data = {}
    with st.form(key='feature_form'):
        for feature in features:
            input_data[feature] = st.number_input(f"{feature}", format="%.6f")
        submit_button = st.form_submit_button(label='Predict Gender')

    if submit_button:
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]
        gender = "Male" if prediction == 1 else "Female"
        st.success(f"🎤 Predicted Gender: **{gender}**")

# --- Page 3: About Me ---
def my_intro():
    st.title("👨‍💻 About Me")

    st.markdown("**Prabhakaran Kumar**")
    st.markdown("**2018–2022 Mechanical Engineering Batch - SKCET, Coimbatore**")
    st.markdown("**2.5 years of experience as a Programmer Analyst**")

    st.write("""
    I’m a **Programmer Analyst** with over **2 years** of experience in developing and maintaining **RESTful APIs** using **Java** and **Flask (Python)**.
    
    I have hands-on experience in **full-stack development**, building interfaces with **JavaScript, CSS, HTML**, and backend services using **Java with Hibernate**.
    
    This voice classification project reflects my interest in AI-powered solutions for real-world problems.
    """)

    st.subheader("📬 Contact Information")
    st.write("📧 Email: prabhakarankumar28@gmail.com")
    st.write("💼 LinkedIn: [linkedin.com/in/prabhakaran-kumar-661441223](https://www.linkedin.com/in/prabhakaran-kumar-661441223/)")
    st.write("📂 GitHub: [github.com/prabhakaran2809](https://github.com/prabhakaran2809)")

# --- Sidebar Navigation ---
st.sidebar.title("📁 Human Voice Classification")
page = st.sidebar.radio("navigate to", ["Project Introduction", "Classifier", "About Me"])

if page == "Project Introduction":
    project_intro()
elif page == "Classifier":
    gender_prediction()
elif page == "About Me":
    my_intro()
