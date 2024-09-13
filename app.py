import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained model
with open('xgb_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the dataset to get feature columns (without the target column)
data = pd.read_csv("solarpowergeneration.csv")

# Rename columns as per the training dataset
new_column_names = {
    'distance-to-solar-noon': 'distance_to_solar_noon',
    'wind-direction': 'wind_direction',
    'wind-speed': 'wind_speed',
    'sky-cover': 'sky_cover',
    'average-wind-speed-(period)': 'average_wind_speed',
    'average-pressure-(period)': 'average_pressure',
    'power-generated': 'power_generated'
}
data = data.rename(columns=new_column_names)

# Select feature columns (exclude 'power_generated')
features = data.drop(columns=['power_generated']).columns

# Streamlit interface
st.set_page_config(page_title="Solar Power Generation Prediction", layout="wide")
st.title("🌞 Solar Power Generation Prediction 🌞")

st.markdown("""
    <style>
    .main {background-color: #f0f2f6;}
    .sidebar .sidebar-content {background-color: #ffffff;}
    </style>
    """, unsafe_allow_html=True)

# Create input fields for each feature
st.write("### Enter the values for the features to predict the power generated:")

user_input = {}
for feature in features:
    if feature in ['temperature', 'humidity']:
        user_input[feature] = st.slider(f"Enter {feature}:", min_value=0, max_value=100, value=0)
    else:
        user_input[feature] = st.number_input(f"Enter {feature}:", value=0.0, format="%.2f")

# Convert user input to DataFrame
input_df = pd.DataFrame([user_input])

# Predict button
if st.button("Predict"):
    # Predict without scaling
    prediction = model.predict(input_df)
    predicted_value = prediction[0]

    # Post-processing to handle specific cases
    if np.isclose(input_df.values, 0).all():
        predicted_value = 0.0  # Set prediction to 0 if all inputs are zero
    
    # Determine background color based on prediction
    if predicted_value > 1000:  # Example threshold for "good" prediction
        background_color = "#d4edda"  # Light green for good prediction
    elif predicted_value < 100:  # Example threshold for "bad" prediction
        background_color = "#f8d7da"  # Light red for bad prediction
    else:
        background_color = "#ffffff"  # White for normal prediction

    st.markdown(f"""
        <style>
        .main {{background-color: {background_color};}}
        </style>
        """, unsafe_allow_html=True)
    
    st.write(f"### Predicted Power Generated: {predicted_value:.2f} Watts (W)")
    
    # Display feature importance
st.write("### Feature Importance")

# Extract feature importances from the model
importances = model.feature_importances_

# Create a DataFrame for feature importances
feature_importances = pd.DataFrame({'Feature': features, 'Importance': importances})

# Sort the DataFrame by importance
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

# Create a bar plot for feature importance
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importances, ax=ax, palette='viridis')

# Add title and labels
ax.set_title('Feature Importance')
ax.set_xlabel('Importance')
ax.set_ylabel('Feature')

# Display the plot in Streamlit
st.pyplot(fig)


# Visualization of user input vs dataset distribution
st.write("### User Input vs Dataset Distribution")

# Loop through each feature to compare dataset distribution with user input
for feature in features:
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the dataset distribution with KDE
    sns.histplot(data[feature], kde=True, label=f"Dataset {feature}", ax=ax, color='lightblue')
    
    # Plot the user input as a vertical line for visibility
    ax.axvline(x=input_df[feature].values[0], color='darkorange', linewidth=2, label=f"User Input {feature}")
    
    # Set legend and title
    ax.legend(loc='upper right')
    ax.set_title(f"{feature} Distribution")
    
    # Display the plot in Streamlit
    st.pyplot(fig)


    st.write("### Prediction Explanation")
    st.markdown("""
    The prediction is based on the following features you have entered:
    - **Distance to Solar Noon**: Affects the angle of sunlight reaching the panels.
    - **Wind Direction**: Influences the cooling of solar panels and their efficiency.
    - **Wind Speed**: Affects cooling and energy generation efficiency.
    - **Sky Cover**: Determines the amount of sunlight reaching the panels.
    - **Average Wind Speed (Period)**: Averages wind speed over a certain period.
    - **Average Pressure (Period)**: Influences weather patterns affecting solar radiation.
    """)
