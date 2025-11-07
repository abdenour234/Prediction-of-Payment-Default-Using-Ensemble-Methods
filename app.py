import streamlit as st
import pickle
import numpy as np
import os

# Set page configuration
st.set_page_config(page_title="Payment Default Prediction", page_icon="💳", layout="wide")

# Title
st.title("💳 Payment Default Prediction System")
st.markdown("---")

# Model loading function with verification
def load_model(model_name):
    """Load the selected model and verify it loaded successfully"""
    model_paths = {
        "Decision Tree": "Models/Decision_tree_best_model.pkl",
        "Naive Bayes": "Models/naive_bayes_best_model.pkl",
        "SVM": "Models/best_svm_model2.pkl",
        "Random Forest (RUS)": "Models/random_forest_rus_model.pkl",
        "Voting Ensemble": "Models/voting_ensemble_model.pkl"
    }
    
    model_path = model_paths.get(model_name)
    
    if not model_path:
        st.error(f"❌ Model '{model_name}' not found in configuration!")
        return None
    
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found at: {model_path}")
        return None
    
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        st.success(f"✅ Model '{model_name}' loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

# Sidebar for model selection
st.sidebar.header("🎯 Model Selection")
model_choice = st.sidebar.selectbox(
    "Choose a prediction model:",
    ["Decision Tree", "Naive Bayes", "SVM", "Random Forest (RUS)", "Voting Ensemble"]
)

st.sidebar.markdown("---")
st.sidebar.info("Enter the customer features on the main panel and click 'Predict' to get the prediction.")

# Main content
st.header("📊 Enter Customer Features")

# Create two columns for input fields
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1, 
                          help="Customer's age in years")
    
    income = st.number_input("Income ($)", min_value=0, max_value=1000000, value=50000, step=1000,
                             help="Annual income in dollars")
    
    loan_amount = st.number_input("Loan Amount ($)", min_value=0, max_value=1000000, value=100000, step=1000,
                                  help="Total loan amount in dollars")
    
    months_employed = st.number_input("Months Employed", min_value=0, max_value=600, value=12, step=1,
                                      help="Number of months employed at current job")

with col2:
    interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=30.0, value=5.0, step=0.1,
                                    help="Interest rate as a percentage")
    
    risk_score = st.number_input("Risk Score", min_value=0.0, max_value=500.0, value=10.0, step=0.1,
                                 help="Credit risk score")
    
    affordability_index = st.number_input("Affordability Index", min_value=0.0, max_value=1000.0, value=50.0, step=0.1,
                                         help="Measure of ability to afford the loan")
    
    employment_maturity = st.number_input("Employment Maturity", min_value=0.0, max_value=10.0, value=5.0, step=0.1,
                                         help="Employment maturity score")

st.markdown("---")

# Prediction button
if st.button("🔮 Predict Default Status", type="primary", use_container_width=True):
    # Load the selected model
    model = load_model(model_choice)
    
    if model is not None:
        # Prepare the input features
        features = np.array([[age, income, loan_amount, months_employed, interest_rate, 
                             risk_score, affordability_index, employment_maturity]])
        
        try:
            # Make prediction
            prediction = model.predict(features)
            prediction_proba = model.predict_proba(features) if hasattr(model, 'predict_proba') else None
            
            # Display results
            st.markdown("---")
            st.header("🎯 Prediction Results")
            
            # Create columns for results
            result_col1, result_col2 = st.columns(2)
            
            with result_col1:
                if prediction[0] == 1:
                    st.error("### ⚠️ HIGH RISK - DEFAULT PREDICTED")
                    st.markdown("**Status:** The customer is likely to default on the loan.")
                else:
                    st.success("### ✅ LOW RISK - NO DEFAULT PREDICTED")
                    st.markdown("**Status:** The customer is likely to repay the loan.")
            
            with result_col2:
                if prediction_proba is not None:
                    st.metric("Probability of No Default", f"{prediction_proba[0][0]:.2%}")
                    st.metric("Probability of Default", f"{prediction_proba[0][1]:.2%}")
                else:
                    st.info("Probability scores not available for this model.")
            
            # Display input summary
            st.markdown("---")
            st.subheader("📋 Input Summary")
            summary_col1, summary_col2 = st.columns(2)
            
            with summary_col1:
                st.write(f"**Age:** {age} years")
                st.write(f"**Income:** ${income:,}")
                st.write(f"**Loan Amount:** ${loan_amount:,}")
                st.write(f"**Months Employed:** {months_employed}")
            
            with summary_col2:
                st.write(f"**Interest Rate:** {interest_rate}%")
                st.write(f"**Risk Score:** {risk_score}")
                st.write(f"**Affordability Index:** {affordability_index}")
                st.write(f"**Employment Maturity:** {employment_maturity}")
                
        except Exception as e:
            st.error(f"❌ Error making prediction: {str(e)}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Payment Default Prediction System | Built with Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)
