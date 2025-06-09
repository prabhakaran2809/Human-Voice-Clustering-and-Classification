# Human-Voice-Clustering-and-Classification

## 📌 Project Overview
This project aims to classify and cluster human voice samples using machine learning based on extracted audio features like spectral properties, MFCCs, and pitch metrics. The end-to-end solution performs data preprocessing, trains various classification models, evaluates their performance, and deploys the best model via a user-friendly Streamlit web application for real-time voice gender predictions.

---

## 🧠 Problem Statement
Develop a machine learning-based system that:
- Classifies voice samples by gender (Male/Female)
- Clusters similar voice patterns
- Enables real-time predictions through a deployed interface

---

## 💼 Business Use Cases
- **Speaker Identification**: Differentiate users based on unique voice signatures  
- **Gender Classification**: Enhance analytics in call centers, media, and assistive tools  
- **Speech Analytics**: Extract meaningful patterns from audio data  
- **Accessibility**: Improve speech-driven AI tools for people with disabilities

---

## 🧹 Data Cleaning & Preprocessing
- Handled missing and inconsistent data
- Normalized all numerical features
- Split the dataset into train, validation, and test subsets
- Prepared a robust dataset ready for clustering and classification

---

## 🧪 Feature Engineering
- Included spectral, pitch, MFCC mean/std features
- Total of 43 features extracted per audio sample
- Features used for both clustering and classification

---

## 🤖 Model Development & Evaluation

### Models Trained:
- Logistic Regression  
- Random Forest  
- Gradient Boosting  
- Support Vector Machine (RBF Kernel)  
- MLPClassifier (Baseline + GridSearch Tuned)

### Evaluation Metrics:
- Accuracy, Precision, Recall, F1-score  
- ROC-AUC (for probability-based models)  
- Confusion Matrix Visualization  
- Silhouette Score for clustering (planned)

### Hyperparameter Tuning:
- GridSearchCV on MLPClassifier for optimal architecture and learning settings

---

## 🔁 Final Model Pipeline
- Final pipeline built using:
  - `StandardScaler` + Best `MLPClassifier`
  - Saved as: `voice_gender_classifier_all_features.pkl` (via joblib)

---

## 🌐 Streamlit Web App

The deployed Streamlit app allows users to manually input extracted audio features and receive real-time gender predictions.

### Pages:
- Project Introduction  
- Classifier Form  
- About Me

### Features:
- Input 43 audio features (MFCCs, spectral, pitch)  
- Outputs: **Male** or **Female** prediction

---

## 🛠️ Tech Stack
- **Language**: Python
- **ML Libraries**: scikit-learn, seaborn, matplotlib
- **Model Serialization**: joblib
- **Deployment**: Streamlit
- **Evaluation**: accuracy, F1-score, ROC-AUC, confusion matrix

---

## 🚀 Usage Instructions

1. **Clone the Repository**  
```bash
git clone https://github.com/your-username/voice-gender-classifier.git
cd voice-gender-classifier
```

2. **Install Dependencies**  
```bash
pip install -r requirements.txt
```

3. **Launch the App**  
```bash
streamlit run app.py
```

4. **Predict Gender**  
- Enter the 43 extracted features  
- Click **Predict Gender**

---

## 📈 Future Work
- Add clustering visualization (e.g., PCA + K-Means plots)
- Add audio upload + feature extraction module
- Integrate DBSCAN or other density-based clustering
- Extend model to age group or emotion classification
- Deploy REST API using Flask or FastAPI
