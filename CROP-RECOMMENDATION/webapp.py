## Importing necessary libraries for the web app
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn import tree
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Display Images
# import Image from pillow to open images
from PIL import Image
img = Image.open("crop.png")
# display image using streamlit
# width is used to set the width of an image
st.image(img)

df= pd.read_csv('Crop_recommendation.csv')

#features = df[['temperature', 'humidity', 'ph', 'rainfall']]
X = df[['N', 'P','K','temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']
labels = df['label']

# Split the data into training and testing sets
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.3, random_state=42)
RF = RandomForestClassifier(n_estimators=20, random_state=5)
RF.fit(Xtrain,Ytrain)
predicted_values = RF.predict(Xtest)
x = metrics.accuracy_score(Ytest, predicted_values)


# Function to load and display an image of the predicted crop
def show_crop_image(crop_name):
    # Assuming we have a directory named 'crop_images' with images named as 'crop_name.jpg'
    image_path = os.path.join('crop_images', crop_name.lower()+'.jpg')
    if os.path.exists(image_path):
        st.image(image_path, caption=f"Recommended crop: {crop_name}", use_column_width=True)
    else:
        st.error("Image not found for the predicted crop.")


import pickle
# Dump the trained Naive Bayes classifier with Pickle
RF_pkl_filename = 'RF.pkl'
# Open the file to save as pkl file
RF_Model_pkl = open(RF_pkl_filename, 'wb')
pickle.dump(RF, RF_Model_pkl)
# Close the pickle instances
RF_Model_pkl.close()


#model = pickle.load(open('RF.pkl', 'rb'))
RF_Model_pkl=pickle.load(open('RF.pkl','rb'))

## Function to make predictions
def predict_crop(nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall):
    # # Making predictions using the model
    prediction = RF_Model_pkl.predict(np.array([nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]).reshape(1, -1))
    return prediction

def validate_npk_combination(n, p, k):
    """Validate if NPK combination is agriculturally realistic"""
    total_npk = n + p + k
    if total_npk > 300:
        return False, "⚠️ NPK combination too high (>300) - may be toxic to plants"
    
    # Check for extremely imbalanced ratios
    if n > 0 and p > 0 and k > 0:
        max_val = max(n, p, k)
        min_val = min(n, p, k)
        if max_val / min_val > 15:
            return False, "⚠️ Nutrient ratio is too imbalanced - may harm plant growth"
    
    return True, ""

def get_npk_warning(n, p, k):
    """Generate warning messages for high NPK values"""
    warnings = []
    
    if n > 120:
        warnings.append("Very high Nitrogen - may cause excessive leaf growth")
    if p > 120:
        warnings.append("Very high Phosphorus - may inhibit other nutrient uptake")
    if k > 180:
        warnings.append("Very high Potassium - may cause salt stress")
    
    return warnings

## Streamlit code for the web app interface
def main():  
    # # Setting the title of the web app
    st.markdown("<h1 style='text-align: center;'>SMART CROP RECOMMENDATIONS", unsafe_allow_html=True)
    
    st.sidebar.title("AgriSens")
    # # Input fields for the user to enter the environmental factors
    st.sidebar.header("Enter Crop Details")
    nitrogen = st.sidebar.number_input("Nitrogen", min_value=0.0, max_value=140.0, value=0.0, step=0.1)
    phosphorus = st.sidebar.number_input("Phosphorus", min_value=0.0, max_value=145.0, value=0.0, step=0.1)
    potassium = st.sidebar.number_input("Potassium", min_value=0.0, max_value=205.0, value=0.0, step=0.1)
    temperature = st.sidebar.number_input("Temperature (°C)", min_value=0.0, max_value=51.0, value=0.0, step=0.1)
    humidity = st.sidebar.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
    ph = st.sidebar.number_input("pH Level", min_value=0.0, max_value=14.0, value=0.0, step=0.1)
    rainfall = st.sidebar.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=0.0, step=0.1)
    inputs=[[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]]                                               
   
    # # Validate inputs and make prediction
    inputs = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])
    if st.sidebar.button("Predict"):
        if not inputs.any() or np.isnan(inputs).any() or (inputs == 0).all():
            st.error("Please fill in all input fields with valid values before predicting.")
        else:
            # Validate NPK combination first
            is_valid, npk_error = validate_npk_combination(nitrogen, phosphorus, potassium)
            if not is_valid:
                st.error(npk_error)
                st.info("💡 Tip: Try more balanced NPK values. Most crops need N:P:K ratios between 1:1:1 to 4:2:3")
            else:
                # Show warnings for high values
                warnings = get_npk_warning(nitrogen, phosphorus, potassium)
                if warnings:
                    st.warning("⚠️ **Agricultural Advisory:**")
                    for warning in warnings:
                        st.write(f"• {warning}")
                    st.write("**Recommendation:** Consider soil testing before applying these nutrient levels.")
                
                # Make prediction
                prediction = predict_crop(nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall)
                st.success(f"🌱 **Recommended crop:** {prediction[0]}")
                
                # Show additional info for extreme cases
                if nitrogen == 140 and phosphorus == 140 and potassium == 140:
                    st.info("📋 **Note:** This NPK combination (140,140,140) is found in the training data but may not be practical for most farming situations.")


## Running the main function
if __name__ == '__main__':
    main()

