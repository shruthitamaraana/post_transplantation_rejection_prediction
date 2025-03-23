# Transplant Rejection Prediction App

This Streamlit application predicts the likelihood of organ transplant rejection based on donor and recipient characteristics.

## Setup

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Place your trained model in the `models/` directory
4. Run the application: `streamlit run app.py`

## Usage

1. Enter patient and donor information in the sidebar
2. Click "Predict Rejection Risk" to get results
3. Review the prediction and risk visualization

## Deployment

This application can be deployed on Streamlit Cloud:
1. Push the repository to GitHub
2. Connect it to Streamlit Cloud
3. Deploy the application

## Requirements

- Python 3.7+
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn