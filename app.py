import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
import xgboost as xgb
import shap

# Set page configuration
st.set_page_config(
    page_title="Organ Transplant Rejection Predictor",
    page_icon="🫀",
    layout="wide"
)

# Load the saved model
@st.cache_resource
def load_model():
    try:
        with open('transplant_rejection_model_improved.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except:
        st.error("Model file not found. Please ensure the model is properly trained and saved.")
        return None

# Function to make predictions
def predict_rejection(input_data, model):
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Make prediction
    rejection_prob = model.predict_proba(input_df)[0, 1]
    rejection_binary = model.predict(input_df)[0]
    
    return rejection_prob, rejection_binary

# Function to calculate risk level
def get_risk_level(probability):
    if probability < 0.3:
        return "Low Risk"
    elif probability < 0.6:
        return "Medium Risk"
    else:
        return "High Risk"

# Function to get color based on risk level
def get_color(probability):
    if probability < 0.3:
        return "green"
    elif probability < 0.6:
        return "orange"
    else:
        return "red"

# Function to explain the prediction using SHAP
def explain_prediction(model, input_df):
    # Get preprocessor and classifier from pipeline
    preprocessor = model.named_steps['preprocessor']
    classifier = model.named_steps['classifier']
    
    # Transform input using preprocessor
    transformed_input = preprocessor.transform(input_df)
    
    # Create explainer
    explainer = shap.TreeExplainer(classifier)
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(transformed_input)
    
    # If shap_values is a list (for multi-class), take the one for positive class
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    return explainer, shap_values, transformed_input

# Main application
def main():
    # Load the model
    model = load_model()
    
    if model is None:
        return
    
    # Application title and description
    st.title("⚕️ Post Organ Transplant Rejection Prediction")
    st.markdown("""
    Under 2C-Second Chance
    
    This application predicts the likelihood of organ rejection post-transplantation based on
    donor and recipient characteristics. It uses an AI model trained on transplantation data
    to provide insights for medical professionals.
    
                 🧑‍⚕️ Who Can Use This?
👨‍⚕️ **Transplant Surgeons** – To assess rejection risk pre- and post-surgery  
🧑‍🔬 **Medical Researchers** – To analyze factors influencing organ rejection  
🏥 **Hospital Administrators** – To improve transplant success rates  
👨‍💻 **Healthcare AI Developers** – To integrate AI into medical decision-making 
    
     ⚠️Note: This tool is meant to assist medical professionals and should not replace clinical judgment.
    """)
    
    # Create tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["Prediction", "About the Model", "Documentation"])
    
    with tab1:
        st.header("Predict Transplant Rejection Risk")
        
        # Create two columns for donor and recipient information
        col1, col2 = st.columns(2)
        
        # Donor information
        with col1:
            st.subheader("Donor Information")
            donor_blood_type = st.selectbox("Donor Blood Type", ["A", "B", "AB", "O"], index=0)
            donor_cmv_status = st.selectbox("Donor CMV Status", ["Positive", "Negative"], index=0)
            donor_age = st.slider("Donor Age", 18, 75, 45)
            donor_gfr = st.slider("Donor GFR (mL/min/1.73m²)", 30, 120, 90)
        
        # Recipient information
        with col2:
            st.subheader("Recipient Information")
            recipient_blood_type = st.selectbox("Recipient Blood Type", ["A", "B", "AB", "O"], index=0)
            recipient_cmv_status = st.selectbox("Recipient CMV Status", ["Positive", "Negative"], index=0)
            recipient_age = st.slider("Recipient Age", 18, 75, 50)
            recipient_bmi = st.slider("Recipient BMI", 15.0, 40.0, 26.0)
            prior_transplant = st.selectbox("Prior Transplant", ["Yes", "No"], index=1)
        
        # Transplant details
        st.subheader("Transplant Details")
        col3, col4 = st.columns(2)
        
        with col3:
            hla_mismatch_score = st.slider("HLA Mismatch Score", 0.0, 6.0, 3.0, 0.1)
            cold_ischemia_time = st.slider("Cold Ischemia Time (hours)", 0.0, 24.0, 8.0, 0.5)
            immunosuppressant_levels = st.slider("Immunosuppressant Levels", 5.0, 15.0, 10.0, 0.1)
        
        with col4:
            infection_risk = st.selectbox("Infection Risk", ["Low", "Medium", "High"], index=1)
            donor_recipient_weight_ratio = st.slider("Donor-Recipient Weight Ratio", 0.5, 1.5, 1.0, 0.01)
            waiting_list_time_days = st.slider("Waiting List Time (days)", 0, 1000, 300)
        
        # Calculate additional features
        age_difference = abs(recipient_age - donor_age)
        blood_match = 1 if donor_blood_type == recipient_blood_type else 0
        cmv_match = 1 if donor_cmv_status == recipient_cmv_status else 0
        age_ischemia_interaction = (recipient_age * cold_ischemia_time) / 100
        
        # Create input data dictionary
        input_data = {
            'donor_blood_type': donor_blood_type,
            'recipient_blood_type': recipient_blood_type,
            'infection_risk': infection_risk,
            'donor_cmv_status': donor_cmv_status,
            'recipient_cmv_status': recipient_cmv_status,
            'prior_transplant': prior_transplant,
            'hla_mismatch_score': hla_mismatch_score,
            'recipient_age': recipient_age,
            'donor_age': donor_age,
            'cold_ischemia_time': cold_ischemia_time,
            'immunosuppressant_levels': immunosuppressant_levels,
            'donor_recipient_weight_ratio': donor_recipient_weight_ratio,
            'recipient_bmi': recipient_bmi,
            'donor_gfr': donor_gfr,
            'waiting_list_time_days': waiting_list_time_days,
            'age_difference': age_difference,
            'blood_match': blood_match,
            'cmv_match': cmv_match,
            'age_ischemia_interaction': age_ischemia_interaction
        }
        
        # Predict button
        if st.button("Predict Rejection Risk"):
            # Make prediction
            rejection_prob, rejection_binary = predict_rejection(input_data, model)
            risk_level = get_risk_level(rejection_prob)
            color = get_color(rejection_prob)
            
            # Display results
            st.subheader("Prediction Results")
            
            # Create three columns for the results
            res1, res2, res3 = st.columns(3)
            
            with res1:
                st.metric("Rejection Probability", f"{rejection_prob:.2%}")
                
            with res2:
                st.markdown(f"<h3 style='color:{color}'>{risk_level}</h3>", unsafe_allow_html=True)
                
            with res3:
                outcome = "Rejection Likely" if rejection_binary == 1 else "No Rejection Likely"
                st.markdown(f"<h3>Prediction: {outcome}</h3>", unsafe_allow_html=True)
            
            # Explanation section
            st.subheader("Prediction Explanation")
            
            # Create input dataframe for SHAP
            input_df = pd.DataFrame([input_data])
            
            try:
                # Get SHAP explanation
                explainer, shap_values, transformed_input = explain_prediction(model, input_df)
                
                # Plot SHAP values
                st.write("Feature Importance for This Prediction:")
                
                # Create matplotlib figure for SHAP
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.summary_plot(shap_values, transformed_input, plot_type="bar", show=False)
                st.pyplot(fig)
                
                # Add interpretations
                st.subheader("Key Factors Affecting This Prediction")
                
                # Create a more detailed explanation based on the input values
                st.markdown("### Medical Insights")
                
                factors = []
                
                # HLA mismatch
                if hla_mismatch_score > 3:
                    factors.append("⚠️ **High HLA mismatch score** increases rejection risk.")
                
                # Blood type compatibility
                if blood_match == 0:
                    factors.append("⚠️ **Blood type mismatch** may increase immunological challenge.")
                else:
                    factors.append("✅ **Matching blood types** is favorable for compatibility.")
                
                # Cold ischemia time
                if cold_ischemia_time > 12:
                    factors.append("⚠️ **Extended cold ischemia time** may damage the organ.")
                
                # Age factors
                if age_difference > 20:
                    factors.append("⚠️ **Large age difference** between donor and recipient.")
                
                # Prior transplant
                if prior_transplant == "Yes":
                    factors.append("⚠️ **Prior transplantation** increases sensitization risk.")
                
                # Infection risk
                if infection_risk == "High":
                    factors.append("⚠️ **High infection risk** may complicate post-transplant recovery.")
                
                # Display the factors
                for factor in factors:
                    st.markdown(factor)
                
            except Exception as e:
                st.error(f"Could not generate SHAP explanation: {e}")
                st.write("However, here are some general factors that influence rejection risk:")
                st.markdown("""
                - HLA matching is crucial for transplant success
                - Blood type compatibility impacts rejection risk
                - Longer cold ischemia time increases organ damage risk
                - Prior transplants may increase sensitization
                - Immunosuppressant levels must be carefully monitored
                """)
    
    # About the Model tab
    with tab2:
        st.header("About the Prediction Model")
        st.markdown("""
        ### Model Information
        
        This application uses an **XGBoost classifier** that was trained on transplantation data to predict the likelihood of organ rejection. The model considers multiple factors from both donors and recipients to make its predictions.
        
                    
        
        ### Key Features Used
        
        The model analyzes these critical factors:
        
        1. **HLA matching** - Human Leukocyte Antigen compatibility between donor and recipient
        2. **Blood type compatibility** - ABO blood group matching
        3. **Cold ischemia time** - Duration the organ remains without blood flow during transport
        4. **CMV status** - Cytomegalovirus status of both donor and recipient
        5. **Demographic factors** - Age, BMI, and other patient characteristics
        6. **Medical history** - Prior transplantations and other medical conditions
        7. **Waiting time** - Duration on the transplant waiting list
        
        ### Model Performance
        
        The model was evaluated using various metrics including:
        - Accuracy
        - Precision & Recall
        - F1 Score
        - ROC-AUC
        
        ### Ethical Considerations
        
        This model is designed as a decision support tool for medical professionals and should not replace clinical judgment. Patient welfare and ethical considerations should always take precedence over model predictions.
        """)
        
        # Display model performance metrics if available
        st.subheader("Model Performance Visualization")
        
        # Create sample confusion matrix for demonstration
        fig, ax = plt.subplots(figsize=(8, 6))
        cm = np.array([[85, 15], [10, 90]])  # Sample confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['No Rejection', 'Rejection'],
                    yticklabels=['No Rejection', 'Rejection'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Sample Confusion Matrix')
        st.pyplot(fig)
        
        # Sample ROC curve
        fig, ax = plt.subplots(figsize=(8, 6))
        fpr = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        tpr = [0, 0.4, 0.65, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.99, 1.0]
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = 0.88)')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Sample ROC Curve')
        plt.legend()
        st.pyplot(fig)

    # Documentation tab
    with tab3:
        st.header("Documentation & User Guide")
        st.markdown("""
        ### How to Use This Tool
        
        1. **Enter donor information** in the left panel
        2. **Enter recipient information** in the right panel
        3. **Provide transplant details** in the bottom section
        4. Click **Predict Rejection Risk** to get results
        5. Review the prediction and explanation
        
        ### Interpreting Results
        
        - **Rejection Probability**: The likelihood of rejection expressed as a percentage
        - **Risk Level**: Categorized as Low (green), Medium (orange), or High (red)
        - **Prediction**: The binary outcome (Rejection Likely or No Rejection Likely)
        - **Feature Importance**: Shows which factors most influenced the prediction
        
        ### Integration with NOTTO Platform
        
        This tool is designed to work as part of a comprehensive organ transplantation management system. It can be integrated with:
        
        - Donor management systems
        - Recipient databases
        - Organ matching algorithms
        - Post-transplant monitoring systems
        
        ### Data Privacy & Security
        
        - All data entered is processed locally and not stored
        - No patient identifiable information is required
        - The system complies with healthcare data regulations
        
        ### References & Additional Resources
        
        - [NOTTO Guidelines](https://www.notto.mohfw.gov.in/)
        - [Transplant Society of India](https://isot.co.in/)
        - [International Society for Heart and Lung Transplantation](https://www.ishlt.org/)
        """)
        
        # Add a disclaimer
        st.warning("""
        **Disclaimer:** This application is designed as a decision support tool for medical professionals. 
        It should not replace clinical judgment or standard medical protocols. 
        Always consult with a qualified transplant team when making medical decisions.
        """)

# Run the application
if __name__ == "__main__":
    main()
