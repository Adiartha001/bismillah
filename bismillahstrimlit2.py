#!/usr/bin/env python
# coding: utf-8

# In[6]:


import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Load the machine learning model and encoders
model = joblib.load('XG_booking_status.pkl')  # Load the XGBoost model
booking_status_encode = joblib.load('booking_status_encode.pkl')  # Load the booking status encoding
oneHot_encode_room = joblib.load('oneHot_encode_room.pkl')  # Load the encoder for room type reserved
oneHot_encode_meal = joblib.load('oneHot_encode_meal.pkl')  # Load the encoder for meal plan type
oneHot_encode_mark = joblib.load('oneHot_encode_mark.pkl')  # Load the encoder for market segment type

def preprocess_data(data, encoder):
    try:
        # Normalize categorical input (lowercase and strip spaces)
        for col in encoder.feature_names_in_:
            data[col] = data[col].str.lower().str.strip()

        # Split columns
        cat_cols = encoder.feature_names_in_

        # Transform categorical data (allow unknown categories)
        data_encoded = pd.DataFrame(
            encoder.transform(data[cat_cols]),
            columns=encoder.get_feature_names_out(),
            index=data.index
        )

        # Concatenate the transformed data with the original data
        all_features = pd.concat([data, data_encoded], axis=1)

        # Drop original categorical columns that have been encoded
        all_features = all_features.drop(cat_cols, axis=1)

        return all_features

    except Exception as e:
        st.error(f"An error occurred during data preprocessing: {e}")
        st.stop()

def main():
    st.title('Booking Status Prediction Model')

    # Add user input components for all features
    no_of_adults = st.number_input("No of Adults", 0, 100)
    no_of_children = st.number_input("No of Children", 0, 100)
    no_of_weekend_nights = st.number_input('No of Weekend Night', 0, 2)
    no_of_week_nights = st.number_input('No of Week Night', 0, 5)
    type_of_meal_plan = st.selectbox('Meal Plan', ['Meal Plan 1', 'Meal Plan 2', 'Meal Plan 3', 'Not Selected'])
    required_car_parking_space = st.radio('Required Car Parking Space (0 for No, 1 for Yes)', [0, 1])
    room_type_reserved = st.selectbox('Room Type Reserved', ['Room_Type 1', 'Room_Type 2', 'Room_Type 3', 
                                                            'Room_Type 4', 'Room_Type 5', 'Room_Type 6', 'Room_Type 7'])
    lead_time = st.number_input("Lead Time (days)", 0, 360)
    arrival_year = st.number_input("Arrival Year", 2017, 2018)
    arrival_month = st.selectbox('Arrival Month', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    arrival_date = st.selectbox("Arrival Date", list(range(1, 32)))  # Selectbox for dates 1-31
    market_segment_type = st.selectbox('Market Segment Type', ['Offline', 'Online', 'Corporate', 'Aviation', 'Complementary'])
    repeated_guest = st.radio('Repeated Guest (0 for No, 1 for Yes)', [0, 1])
    no_of_previous_cancellations = st.number_input('Previous Cancellations', 0, 100)
    no_of_previous_bookings_not_canceled = st.number_input('Previous Bookings Not Canceled', 0, 100)
    avg_price_per_room = st.number_input('Average Price Per Room (in Euros)', 0.00, 10000.00)
    no_of_special_requests = st.number_input('Number of Special Requests', 0, 100)

    # Prepare input data for prediction
    input_data = pd.DataFrame({
        'no_of_adults': [no_of_adults],
        'no_of_children': [no_of_children],
        'no_of_weekend_nights': [no_of_weekend_nights],
        'no_of_week_nights': [no_of_week_nights],
        'type_of_meal_plan': [type_of_meal_plan],
        'required_car_parking_space': [required_car_parking_space],
        'room_type_reserved': [room_type_reserved],
        'lead_time': [lead_time],
        'arrival_year': [arrival_year],
        'arrival_month': [arrival_month],
        'arrival_date': [arrival_date],
        'market_segment_type': [market_segment_type],
        'repeated_guest': [repeated_guest],
        'no_of_previous_cancellations': [no_of_previous_cancellations],
        'no_of_previous_bookings_not_canceled': [no_of_previous_bookings_not_canceled],
        'avg_price_per_room': [float(avg_price_per_room)],
        'no_of_special_requests': [no_of_special_requests]
    })

    # Preprocess the input data
    processed_data = preprocess_data(input_data, oneHot_encode_meal)

    # Make prediction if button is pressed
    if st.button('Predict Booking Status'):
        # Make a prediction using the model
        prediction = model.predict(processed_data)

        # Display prediction result
        if prediction == 1:
            st.success("The booking status is: Confirmed")
        else:
            st.error("The booking status is: Not Confirmed")

    # Test Case 1: Prediksi berdasarkan data yang sudah ditentukan (Cancelled)
    if st.button('Test Case 1 (Cancelled)'):
        test_case_1 = {
            'no_of_adults': 2, 'no_of_children': 1, 'no_of_weekend_nights': 1,
            'no_of_week_nights': 2, 'type_of_meal_plan': 'Meal Plan 1', 'required_car_parking_space': 1,
            'room_type_reserved': 'Room_Type 2', 'lead_time': 15, 'arrival_year': 2017, 'arrival_month': 5,
            'arrival_date': 15, 'market_segment_type': 'Online', 'repeated_guest': 0,
            'no_of_previous_cancellations': 1, 'no_of_previous_bookings_not_canceled': 5, 'avg_price_per_room': 100.0,
            'no_of_special_requests': 0
        }
        # Create DataFrame for Test Case 1
        test_case_1_data = pd.DataFrame([test_case_1])
        processed_data_1 = preprocess_data(test_case_1_data, oneHot_encode_meal)
        prediction_1 = model.predict(processed_data_1)
        st.write("Test Case 1 Prediction Result:", "✅ Confirmed" if prediction_1 == 1 else "❌ Not Confirmed")

    # Test Case 2: Prediksi berdasarkan data yang sudah ditentukan (Not Cancelled)
    if st.button('Test Case 2 (Not Cancelled)'):
        test_case_2 = {
            'no_of_adults': 1, 'no_of_children': 0, 'no_of_weekend_nights': 2,
            'no_of_week_nights': 3, 'type_of_meal_plan': 'Meal Plan 3', 'required_car_parking_space': 0,
            'room_type_reserved': 'Room_Type 5', 'lead_time': 30, 'arrival_year': 2018, 'arrival_month': 9,
            'arrival_date': 10, 'market_segment_type': 'Corporate', 'repeated_guest': 1,
            'no_of_previous_cancellations': 0, 'no_of_previous_bookings_not_canceled': 10, 'avg_price_per_room': 200.0,
            'no_of_special_requests': 2
        }
        # Create DataFrame for Test Case 2
        test_case_2_data = pd.DataFrame([test_case_2])
        processed_data_2 = preprocess_data(test_case_2_data, oneHot_encode_meal)
        prediction_2 = model.predict(processed_data_2)
        st.write("Test Case 2 Prediction Result:", "✅ Confirmed" if prediction_2 == 1 else "❌ Not Confirmed")

if __name__ == "__main__":
    main()


# In[ ]:




