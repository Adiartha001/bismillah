#!/usr/bin/env python
# coding: utf-8

# In[15]:


pip install joblib


# In[16]:


import pandas as pd
import numpy as np
import pickle as pkl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, RobustScaler, MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import classification_report
import warnings

class DataProcessor:
    def __init__(self):
        self.df = None
        self.room_enc = OneHotEncoder(sparse=False)
        self.meal_enc = OneHotEncoder(sparse=False)
        self.mark_enc = OneHotEncoder(sparse=False)

    def load_data(self):
        """Load the dataset from a fixed filename."""
        self.df = pd.read_csv('Dataset_B_hotel.csv')
        print("Dataset loaded successfully.")

    def preview_data(self):
        """Preview the first few rows of the dataset."""
        if self.df is not None:
            return self.df.head()
        else:
            print("Data not loaded yet.")
            return None

    def data_summary(self):
        """Return summary statistics of the dataset."""
        if self.df is not None:
            return self.df.describe(), self.df.info()
        else:
            print("Data not loaded yet.")
            return None

    def handle_missing_data(self):
        """Handle missing data by filling with mean or mode."""
        self.df['avg_price_per_room'].fillna(103.51, inplace=True)
        self.df['required_car_parking_space'].fillna(0, inplace=True)
        self.df['type_of_meal_plan'].fillna('Meal Plan 1', inplace=True)
        print("Missing data handled.")

    def encode_categorical_data(self):
        """Encode categorical data using OneHotEncoder."""
        self.room_enc_train = self.room_enc.fit_transform(self.df[['room_type_reserved']])
        self.meal_enc_train = self.meal_enc.fit_transform(self.df[['type_of_meal_plan']])
        self.mark_enc_train = self.mark_enc.fit_transform(self.df[['market_segment_type']])

        # Convert encoded features into DataFrame and append them to the original data
        room_enc_df = pd.DataFrame(self.room_enc_train, columns=self.room_enc.get_feature_names_out())
        meal_enc_df = pd.DataFrame(self.meal_enc_train, columns=self.meal_enc.get_feature_names_out())
        mark_enc_df = pd.DataFrame(self.mark_enc_train, columns=self.mark_enc.get_feature_names_out())

        self.df = pd.concat([self.df, room_enc_df, meal_enc_df, mark_enc_df], axis=1)

        # Drop original categorical columns
        self.df.drop(['room_type_reserved', 'type_of_meal_plan', 'market_segment_type'], axis=1, inplace=True)

        print("Categorical data encoded.")

        # Save the encoders to pickle for future use
        pkl.dump(self.room_enc, open('oneHot_encode_room.pkl', 'wb'))
        pkl.dump(self.meal_enc, open('oneHot_encode_meal.pkl', 'wb'))
        pkl.dump(self.mark_enc, open('oneHot_encode_mark.pkl', 'wb'))

        print("Encoders saved to pickle.")

    def split_data(self):
        """Split the dataset into training and testing sets."""
        input_df = self.df.drop(['booking_status', 'Booking_ID'], axis=1)
        output_df = self.df['booking_status']

        return train_test_split(input_df, output_df, test_size=0.2, random_state=42)

    def handle_outliers_and_scale(self, x_train, x_test, numerical_cols):
        """Handle outliers and apply scaling based on outliers and skewness."""
        exclude_cols = ['type_of_meal_plan', 'room_type_reserved', 'arrival_month',
                        'market_segment_type', 'booking_status', 'repeated_guest', 
                        'arrival_year']

        numerical_cols = [col for col in numerical_cols if col not in exclude_cols]

        # Loop through each column
        for col in numerical_cols:
            # Calculate IQR for detecting outliers
            Q1 = x_train[col].quantile(0.25)
            Q3 = x_train[col].quantile(0.75)
            IQR = Q3 - Q1

            # Calculate lower and upper bounds for outliers
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Detect outliers
            outliers_train = x_train[(x_train[col] < lower_bound) | (x_train[col] > upper_bound)]
            outliers_test = x_test[(x_test[col] < lower_bound) | (x_test[col] > upper_bound)]

            # Check for outliers
            if not outliers_train.empty or not outliers_test.empty:
                print(f"Outliers detected in {col}, applying RobustScaler.")
                # Apply RobustScaler to handle outliers
                scaler = RobustScaler()
                x_train[col] = scaler.fit_transform(x_train[[col]])
                x_test[col] = scaler.transform(x_test[[col]])
            else:
                # Check skewness
                skewness = x_train[col].skew()
                if abs(skewness) > 0.5:
                    print(f"Skewness detected in {col} (Skewness: {skewness:.2f}), applying MinMaxScaler.")
                    # Apply MinMaxScaler to handle skewness
                    scaler = MinMaxScaler()
                    x_train[col] = scaler.fit_transform(x_train[[col]])
                    x_test[col] = scaler.transform(x_test[[col]])
                else:
                    print(f"No significant outliers or skewness in {col}, applying StandardScaler.")
                    # Apply StandardScaler for columns with no outliers or skewness
                    scaler = StandardScaler()
                    x_train[col] = scaler.fit_transform(x_train[[col]])
                    x_test[col] = scaler.transform(x_test[[col]])


class ModelTrainer:
    def __init__(self):
        self.xgb_model = None
        self.rf_model = None

    def train_xgboost(self, x_train, y_train):
        """Train an XGBoost model."""
        self.xgb_model = xgb.XGBClassifier(max_depth=4)
        self.xgb_model.fit(x_train, y_train)
        print("XGBoost model trained.")

    def train_random_forest(self, x_train, y_train):
        """Train a Random Forest model."""
        self.rf_model = RandomForestClassifier(n_estimators=100, max_depth=4)
        self.rf_model.fit(x_train, y_train)
        print("Random Forest model trained.")

    def evaluate_model(self, model, x_test, y_test):
        """Evaluate the model using classification report."""
        y_pred = model.predict(x_test)
        return classification_report(y_test, y_pred, target_names=['1', '0'])


class ModelSaver:
    @staticmethod
    def save_model(model, filename):
        """Save the trained model using pickle."""
        pkl.dump(model, open(filename, 'wb'))
        print(f"Model saved as {filename}")


class BookingPredictionSystem:
    def __init__(self):
        self.data_processor = DataProcessor()
        self.model_trainer = ModelTrainer()

    def run(self):
        """Run the complete process from data loading to model evaluation."""
        # Load and preprocess data
        self.data_processor.load_data()
        self.data_processor.handle_missing_data()
        self.data_processor.handle_outliers_and_scale(self.data_processor.df, self.data_processor.df, self.data_processor.df.select_dtypes(include=[np.number]).columns.tolist())  # Handling outliers and scaling
        self.data_processor.encode_categorical_data()

        # Split data into train and test sets
        x_train, x_test, y_train, y_test = self.data_processor.split_data()

        # Encode the booking status labels (Canceled -> 1, Not Canceled -> 0)
        booking_status_encode = {"Canceled": 1, "Not_Canceled": 0}
        y_train = y_train.replace(booking_status_encode)
        y_test = y_test.replace(booking_status_encode)

        # Save the booking_status_encode dictionary to pickle for consistency during inference
        pkl.dump(booking_status_encode, open('booking_status_encode.pkl', 'wb'))
        print("Booking status encoding saved to pickle.")

        # Train models
        self.model_trainer.train_xgboost(x_train, y_train)
        self.model_trainer.train_random_forest(x_train, y_train)

        # Evaluate models
        xgb_report = self.model_trainer.evaluate_model(self.model_trainer.xgb_model, x_test, y_test)
        rf_report = self.model_trainer.evaluate_model(self.model_trainer.rf_model, x_test, y_test)

        print("\nXGBoost Model Evaluation Report:")
        print(xgb_report)

        print("\nRandom Forest Model Evaluation Report:")
        print(rf_report)

        # Save only the XGBoost model
        ModelSaver.save_model(self.model_trainer.xgb_model, 'XG_booking_status.pkl')


# Example usage
if __name__ == "__main__":
    booking_system = BookingPredictionSystem()
    booking_system.run()


# In[ ]:




