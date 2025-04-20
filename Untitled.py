#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model dan encoder
model = joblib.load('XG_booking_status.pkl')
booking_status_encode = joblib.load('booking_status_encode.pkl')
oneHot_encode_room = joblib.load('oneHot_encode_room.pkl')
oneHot_encode_meal = joblib.load('oneHot_encode_meal.pkl')
oneHot_encode_mark = joblib.load('oneHot_encode_mark.pkl')

st.title("Hotel Booking Status Prediction")

# Input pengguna
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
arrival_month = st.selectbox('Arrival Month', list(range(1, 13)))
arrival_date = st.selectbox("Arrival Date", list(range(1, 32)))
market_segment_type = st.selectbox('Market Segment Type', ['Offline', 'Online', 'Corporate', 'Aviation', 'Complementary'])
repeated_guest = st.radio('Repeated Guest (0 for No, 1 for Yes)', [0, 1])
no_of_previous_cancellations = st.number_input('Previous Cancellations', 0, 100)
no_of_previous_bookings_not_canceled = st.number_input('Previous Bookings Not Canceled', 0, 100)
avg_price_per_room = st.number_input('Average Price Per Room (in Euros)', 0.00, 10000.00)
no_of_special_requests = st.number_input('Number of Special Requests', 0, 100)

if st.button('Predict Booking Status'):
    input_data = pd.DataFrame({
        'no_of_adults': [no_of_adults],
        'no_of_children': [no_of_children],
        'no_of_weekend_nights': [no_of_weekend_nights],
        'no_of_week_nights': [no_of_week_nights],
        'required_car_parking_space': [required_car_parking_space],
        'lead_time': [lead_time],
        'arrival_year': [arrival_year],
        'arrival_month': [arrival_month],
        'arrival_date': [arrival_date],
        'repeated_guest': [repeated_guest],
        'no_of_previous_cancellations': [no_of_previous_cancellations],
        'no_of_previous_bookings_not_canceled': [no_of_previous_bookings_not_canceled],
        'avg_price_per_room': [avg_price_per_room],
        'no_of_special_requests': [no_of_special_requests]
    })

    meal_encoded = oneHot_encode_meal.transform([[type_of_meal_plan]])
    room_encoded = oneHot_encode_room.transform([[room_type_reserved]])
    market_encoded = oneHot_encode_mark.transform([[market_segment_type]])

    # Gabungkan fitur numerik + one-hot encoded fitur kategori
    full_input = np.hstack([input_data.values, meal_encoded.toarray(), room_encoded.toarray(), market_encoded.toarray()])

    # Prediksi
    prediction = model.predict(full_input)
    output = booking_status_encode.inverse_transform(prediction)[0]

    st.success(f"Hasil Prediksi Booking Status: {output}")


# In[ ]:




