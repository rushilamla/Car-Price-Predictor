import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import streamlit as st
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

# Function to format numbers in Indian system


def format_indian_number(number):
    s = str(int(number))
    if len(s) <= 3:
        return s
    result = s[-3:]
    s = s[:-3]
    while s:
        result = s[-2:] + "," + result
        s = s[:-2]
    if result.startswith(","):
        result = result[1:]
    return result


# Custom CSS for enhanced UI
st.markdown("""
    <style>
    .main {background-color: #f4f6f9;}
    .stButton>button {
        background-color: #007bff;
        color: white;
        padding: 12px 24px;
        border: none;
        border-radius: 8px;
        cursor: pointer;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #0056b3;
    }
    .stSelectbox, .stSlider {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 8px;
        border: 1px solid #ced4da;
    }
    h1, h2, h3 {color: #1a3c66; font-family: 'Arial', sans-serif;}
    .stSuccess {background-color: #d4edda; color: #155724; padding: 15px; border-radius: 8px;}
    .stWarning {background-color: #fff3cd; color: #856404; padding: 15px; border-radius: 8px;}
    .sidebar .stSelectbox {font-size: 16px;}
    </style>
""", unsafe_allow_html=True)

# Function to load or train models


def load_or_train_model():
    model_file_rf = 'car_price_model_rf.pkl'
    model_file_gb = 'car_price_model_gb.pkl'
    scaler_file = 'scaler.pkl'
    le_brand_file = 'le_brand.pkl'
    le_model_file = 'le_model.pkl'
    le_fuel_file = 'le_fuel.pkl'
    le_trans_file = 'le_trans.pkl'
    le_seller_file = 'le_seller.pkl'
    brand_model_map_file = 'brand_model_map.pkl'

    files = [model_file_rf, model_file_gb, scaler_file, le_brand_file, le_model_file,
             le_fuel_file, le_trans_file, le_seller_file, brand_model_map_file]

    if all(os.path.exists(f) for f in files):
        with open(model_file_rf, 'rb') as file:
            model_rf = pickle.load(file)
        with open(model_file_gb, 'rb') as file:
            model_gb = pickle.load(file)
        with open(scaler_file, 'rb') as file:
            scaler = pickle.load(file)
        with open(le_brand_file, 'rb') as file:
            le_brand = pickle.load(file)
        with open(le_model_file, 'rb') as file:
            le_model = pickle.load(file)
        with open(le_fuel_file, 'rb') as file:
            le_fuel = pickle.load(file)
        with open(le_trans_file, 'rb') as file:
            le_trans = pickle.load(file)
        with open(le_seller_file, 'rb') as file:
            le_seller = pickle.load(file)
        with open(brand_model_map_file, 'rb') as file:
            brand_model_map = pickle.load(file)
        return model_rf, model_gb, scaler, le_brand, le_model, le_fuel, le_trans, le_seller, brand_model_map

    # Load dataset
    try:
        df = pd.read_csv('cardekho_data.csv')
    except FileNotFoundError:
        st.error(
            "Please save your dataset as 'cardekho_data.csv' in the same directory.")
        return None, None, None, None, None, None, None, None, None

    # Preprocess data
    df = df.dropna()
    X = df[['brand', 'model', 'vehicle_age', 'km_driven', 'fuel_type',
            'transmission_type', 'seller_type', 'engine', 'max_power', 'seats']]
    y = df['selling_price']

    # Create brand-model mapping
    brand_model_map = df.groupby('brand')['model'].unique().to_dict()
    for brand in brand_model_map:
        brand_model_map[brand] = list(brand_model_map[brand])

    # Encode categorical variables
    le_brand = LabelEncoder()
    le_model = LabelEncoder()
    le_fuel = LabelEncoder()
    le_trans = LabelEncoder()
    le_seller = LabelEncoder()

    X['brand'] = le_brand.fit_transform(X['brand'])
    X['model'] = le_model.fit_transform(X['model'])
    X['fuel_type'] = le_fuel.fit_transform(X['fuel_type'])
    X['transmission_type'] = le_trans.fit_transform(X['transmission_type'])
    X['seller_type'] = le_seller.fit_transform(X['seller_type'])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train models
    model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
    model_rf.fit(X_train_scaled, y_train)

    model_gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model_gb.fit(X_train_scaled, y_train)

    # Evaluate models
    rf_pred = model_rf.predict(X_test_scaled)
    gb_pred = model_gb.predict(X_test_scaled)
    st.write(
        f"Random Forest MAE: {mean_absolute_error(y_test, rf_pred):,.2f} INR")
    st.write(
        f"Gradient Boosting MAE: {mean_absolute_error(y_test, gb_pred):,.2f} INR")

    # Save models, encoders, and brand-model map
    with open(model_file_rf, 'wb') as file:
        pickle.dump(model_rf, file)
    with open(model_file_gb, 'wb') as file:
        pickle.dump(model_gb, file)
    with open(scaler_file, 'wb') as file:
        pickle.dump(scaler, file)
    with open(le_brand_file, 'wb') as file:
        pickle.dump(le_brand, file)
    with open(le_model_file, 'wb') as file:
        pickle.dump(le_model, file)
    with open(le_fuel_file, 'wb') as file:
        pickle.dump(le_fuel, file)
    with open(le_trans_file, 'wb') as file:
        pickle.dump(le_trans, file)
    with open(le_seller_file, 'wb') as file:
        pickle.dump(le_seller, file)
    with open(brand_model_map_file, 'wb') as file:
        pickle.dump(brand_model_map, file)

    return model_rf, model_gb, scaler, le_brand, le_model, le_fuel, le_trans, le_seller, brand_model_map

# Function to generate feature importance plot


def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=importances[indices], y=np.array(
        feature_names)[indices], palette='viridis')
    plt.title('Feature Importance in Price Prediction')
    plt.xlabel('Importance')
    plt.tight_layout()

    # Convert plot to base64 for Streamlit
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    plt.close()
    return img_str

# Streamlit app


def main():
    st.title("Used Car Price Prediction")
    st.markdown("Estimate the selling price of a used car in INR, including future projections with 5% annual inflation and model comparison.")

    # Sidebar for model selection
    st.sidebar.header("Model Selection")
    model_choice = st.sidebar.selectbox("Choose Prediction Model", [
                                        "Random Forest", "Gradient Boosting"])

    # Load models and encoders
    model_rf, model_gb, scaler, le_brand, le_model, le_fuel, le_trans, le_seller, brand_model_map = load_or_train_model()

    if model_rf is None:
        st.stop()

    model = model_rf if model_choice == "Random Forest" else model_gb

    # Input fields
    st.subheader("Car Details")
    col1, col2 = st.columns(2)
    with col1:
        brand = st.selectbox(
            "Car Brand", le_brand.classes_.tolist(), key="brand")
        models = brand_model_map.get(brand, le_model.classes_.tolist())
        model_name = st.selectbox("Car Model", models, key="model")
        fuel_type = st.selectbox("Fuel Type", le_fuel.classes_.tolist())
    with col2:
        transmission_type = st.selectbox(
            "Transmission Type", le_trans.classes_.tolist())
        seller_type = st.selectbox("Seller Type", le_seller.classes_.tolist())

    st.subheader("Vehicle Specifications")
    col3, col4 = st.columns(2)
    with col3:
        vehicle_age = st.slider("Vehicle Age (years)", 0, 20, 5)
        km_driven = st.slider("Kilometers Driven", 0, 200000, 50000, step=1000)
    with col4:
        engine = st.slider("Engine Size (cc)", 600, 4000, 1200, step=50)
        max_power = st.slider("Max Power (bhp)", 30.0, 300.0, 80.0, step=0.5)
        seats = st.slider("Number of Seats", 2, 9, 5)

    # Prepare input data
    input_data = pd.DataFrame({
        'brand': [brand],
        'model': [model_name],
        'vehicle_age': [vehicle_age],
        'km_driven': [km_driven],
        'fuel_type': [fuel_type],
        'transmission_type': [transmission_type],
        'seller_type': [seller_type],
        'engine': [engine],
        'max_power': [max_power],
        'seats': [seats]
    })

    # Encode categorical inputs
    try:
        input_data['brand'] = le_brand.transform(input_data['brand'])
        input_data['model'] = le_model.transform(input_data['model'])
        input_data['fuel_type'] = le_fuel.transform(input_data['fuel_type'])
        input_data['transmission_type'] = le_trans.transform(
            input_data['transmission_type'])
        input_data['seller_type'] = le_seller.transform(
            input_data['seller_type'])
    except ValueError as e:
        st.error(
            "Selected model is not valid for the chosen brand. Please select a valid combination.")
        st.stop()

    # Scale input data
    input_scaled = scaler.transform(input_data)

    # Predict price with confidence interval
    if st.button("Predict Price"):
        predictions = []
        for _ in range(100):  # Bootstrap for confidence interval
            indices = np.random.choice(
                input_scaled.shape[0], size=input_scaled.shape[0], replace=True)
            pred = model.predict(input_scaled[indices])
            predictions.append(pred[0])

        current_price = np.mean(predictions)
        conf_interval = np.percentile(predictions, [2.5, 97.5])

        st.success(
            f"Predicted Current Car Price (2025): ₹{format_indian_number(current_price)}")
        st.write(
            f"95% Confidence Interval: ₹{format_indian_number(conf_interval[0])} - ₹{format_indian_number(conf_interval[1])}")

        # Predict future prices
        current_year = 2025
        future_prices = [current_price]
        for year in range(1, 4):
            future_data = input_data.copy()
            future_data['vehicle_age'] += year
            future_data['km_driven'] += year * 10000
            future_scaled = scaler.transform(future_data)
            future_pred = []
            for _ in range(100):
                indices = np.random.choice(
                    future_scaled.shape[0], size=future_scaled.shape[0], replace=True)
                pred = model.predict(future_scaled[indices])
                future_pred.append(pred[0])
            future_price = np.mean(future_pred) * (1 + 0.05) ** year
            future_prices.append(future_price)

        # Create line chart for price trend
        st.subheader("Price Trend Over Next 3 Years")
        st.markdown(
            "*Note: Future prices assume 10,000 km driven per year and 5% annual inflation, with static market conditions.*")
        price_data = pd.DataFrame({
            'Year': [str(current_year), str(current_year + 1), str(current_year + 2), str(current_year + 3)],
            'Price (₹)': future_prices
        })
        st.line_chart(price_data.set_index('Year')['Price (₹)'])


if __name__ == '__main__':
    main()
