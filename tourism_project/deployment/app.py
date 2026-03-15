import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib
import warnings
warnings.filterwarnings("ignore")

# Download the model from the Model Hub
model_path = hf_hub_download(repo_id="navzen2000/tourism-model", filename="best_tourism_model_prod.joblib")

# Load the model
model = joblib.load(model_path)

# Streamlit UI for Tourism Customer Prediction
st.title("Visit with Us - Tourism Prediction App")
st.write("The Tourism Prediction App is an internal tool for tourism company \"Visit with Us\", that integrates customer data, predicts potential buyers, and enhances decision-making for marketing strategies.")
st.write("Kindly enter the customer details to check whether they are likely to buy the tourism product.")

# Collect user input
Age = st.number_input("Age (Customer's age in years)", min_value=18, max_value=100, value=30)
TypeofContact = st.selectbox("TypeofContact (Method used to contact customer)", ["Self Enquiry", "Company Invited"])
CityTier = st.selectbox("City Tier (City category)", [1,2,3])
Occupation = st.selectbox("Occupation (Customer's occupation)", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
Gender = st.selectbox("Gender (Gender of the customer)", ["Male", "Female"])
NumberOfPersonVisiting = st.selectbox("NumberOfPersonVisiting (Total number of people accompanying the customer on the trip)",[1,2,3,4,5])
PreferredPropertyStar = st.selectbox("PreferredPropertyStar (Preferred hotel rating by the customer)",[3.0,4.0,5.0])
MaritalStatus = st.selectbox("MaritalStatus (Marital status of the customer)", ["Single", "Married", "Divorced"])
NumberOfTrips = st.number_input("NumberOfTrips (Average number of trips the customer takes annually)", min_value=0, max_value=30, value=4)
Passport = st.selectbox("Passport (Whether the customer holds a valid passport)",["No", "Yes"])
OwnCar = st.selectbox("OwnCar (Whether the customer owns a car)",["No", "Yes"])
NumberOfChildrenVisiting = st.selectbox("NumberOfChildrenVisiting (Number of children below age 5 accompanying the customer)",[0,1,2,3])
Designation = st.selectbox("Designation (Customer's designation in their current organization)",["Executive", "Manager", "Senior Manager", "AVP", "VP"])
MonthlyIncome = st.number_input("MonthlyIncome (Gross monthly income of the customer)", min_value=1000, max_value=100000, value=25000)
PitchSatisfactionScore = st.selectbox("PitchSatisfactionScore (Score indicating the customer's satisfaction with the sales pitch)",[1,2,3,4,5])
ProductPitched = st.selectbox("ProductPitched (The type of product pitched to the customer)", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])
NumberOfFollowups = st.number_input("NumberOfFollowups (Total number of follow-ups by the salesperson after the sales pitch)",min_value=1, max_value=10, value=4)
DurationOfPitch = st.number_input("DurationOfPitch (Duration of the sales pitch delivered to the customer)",min_value=1, max_value=150, value=16)


# Convert categorical inputs to match model training
input_data = pd.DataFrame([{
    'Age': Age,
    'TypeofContact': TypeofContact,
    'CityTier': CityTier,
    'Occupation': Occupation,
    'Gender': Gender,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'PreferredPropertyStar': PreferredPropertyStar,
    'MaritalStatus' : MaritalStatus,
    'NumberOfTrips': NumberOfTrips,
    'Passport': 1 if Passport == "Yes" else 0,
    'OwnCar': 1 if OwnCar == "Yes" else 0,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'Designation' : Designation,
    'MonthlyIncome' : MonthlyIncome,
    'PitchSatisfactionScore' : PitchSatisfactionScore,
    'ProductPitched': ProductPitched,
    'NumberOfFollowups' : NumberOfFollowups,
    'DurationOfPitch' : DurationOfPitch
}])

# Set the classification threshold
classification_threshold = 0.45

# Predict button
if st.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = (prediction_proba >= classification_threshold).astype(int)
    result = "purchase" if prediction == 1 else "not purchase"
    st.write(f"Based on the information provided, the customer is likely to {result} the tourism product.")
