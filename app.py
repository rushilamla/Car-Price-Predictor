import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import streamlit as st
import pickle
import os
Function to format numbers in Indian system
def format_indian_number(number):
    # Convert to integer and string
    s = str(int(number))
    if len(s) <= 3:
        return s
    # Take last 3 digits
    result = s[-3:]
    # Process remaining digits from right to left in groups of 2
    s = s[:-3]
    while s:
        # Take up to 2 digits at a time
        result = s[-2:] + "," + result
        s = s[:-2]
    # Remove leading comma if present
    if result.startswith(","):
        result = result[1:]
    return result
st.markdown("""
    <style>
    .main {background-color: #f0f2f6;}
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 20px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stSelectbox>div>div>select {
        background-color: #ffffff;
        border-radius: 5px;
        padding: 5px;
    }
    h1, h3 {color: #2c3e50;}
    .stSuccess {background-color: #d4edda; color: #155724; padding: 10px; border-radius: 5px;}
    </style>
""", unsafe_allow_html=True)

def load_or_train_model():
    model_file = 'car_price_model.pkl'
    scaler_file = 'scaler.pkl'
    le_brand_file = 'le_brand.pkl'
    le_model_file = 'le_model.pkl'
    le_fuel_file = 'le_fuel.pkl'
    le_trans_file = 'le_trans.pkl'
    le_seller_file = 'le_seller.pkl'
    brand_model_map_file = 'brand_model_map.pkl'
    
    if all(os.path.exists(f) for f in [model_file, scaler_file, le_brand_file, le_model_file, le_fuel_file, le_trans_file, le_seller_file, brand_model_map_file]):
        with open(model_file, 'rb') as file:
            model = pickle.load(file)
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
        return model, scaler, le_brand, le_model, le_fuel, le_trans, le_seller, brand_model_map
    
    try:
        df = pd.read_csv('cardekho_data.csv')
    except FileNotFoundError:
        st.error("Please save your dataset as 'cardekho_data.csv' in the same directory.")
        return None, None, None, None, None, None, None, None
    
    df = df.dropna()
    X = df[['brand', 'model', 'vehicle_age', 'km_driven', 'fuel_type', 'transmission_type', 'seller_type', 'engine', 'max_power', 'seats']]
    y = df['selling_price']
    
    brand_model_map = df.groupby('brand')['model'].unique().to_dict()
    for brand in brand_model_map:
        brand_model_map[brand] = list(brand_model_map[brand])

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
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    with open(model_file, 'wb') as file:
        pickle.dump(model, file)
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
    
    return model, scaler, le_brand, le_model, le_fuel, le_trans, le_seller, brand_model_map

def main():
    st.title("Used Car Price Prediction")
    # st.markdown("**Predict the price of a used car in Jammu, India**")
    st.markdown("Enter the car details below to estimate the selling price in INR, including future projections with 5% annual inflation.")
    
    model, scaler, le_brand, le_model, le_fuel, le_trans, le_seller, brand_model_map = load_or_train_model()
    
    if model is None:
        st.stop()

    st.subheader("Car Details")
    brand = st.selectbox("Car Brand", le_brand.classes_.tolist(), key="brand")
    models = brand_model_map.get(brand, le_model.classes_.tolist())
    model_name = st.selectbox("Car Model", models, key="model")
    fuel_type = st.selectbox("Fuel Type", le_fuel.classes_.tolist())
    transmission_type = st.selectbox("Transmission Type", le_trans.classes_.tolist())
    seller_type = st.selectbox("Seller Type", le_seller.classes_.tolist())
    
    st.subheader("Vehicle Specifications")
    vehicle_age = st.slider("Vehicle Age (years)", 0, 20, 5)
    km_driven = st.slider("Kilometers Driven", 0, 200000, 50000)
    engine = st.slider("Engine Size (cc)", 600, 4000, 1200)
    max_power = st.slider("Max Power (bhp)", 30.0, 300.0, 80.0)
    seats = st.slider("Number of Seats", 2, 9, 5)
    
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

    try:
        input_data['brand'] = le_brand.transform(input_data['brand'])
        input_data['model'] = le_model.transform(input_data['model'])
        input_data['fuel_type'] = le_fuel.transform(input_data['fuel_type'])
        input_data['transmission_type'] = le_trans.transform(input_data['transmission_type'])
        input_data['seller_type'] = le_seller.transform(input_data['seller_type'])
    except ValueError as e:
        st.error("Selected model is not valid for the chosen brand. Please select a valid combination.")
        st.stop()

    input_scaled = scaler.transform(input_data)
    
    if st.button("Predict Price"):
        current_price = model.predict(input_scaled)[0]
        st.success(f"Predicted Current Car Price (2025): ₹{format_indian_number(current_price)}")
        
        current_year = 2025
        future_prices = [current_price]
        for year in range(1, 4):
            future_data = input_data.copy()
            future_data['vehicle_age'] += year
            future_data['km_driven'] += year * 10000  # Assume 10,000 km per year
            future_scaled = scaler.transform(future_data)
            future_price = model.predict(future_scaled)[0]
            # Apply 5% annual inflation
            inflation_adjusted_price = future_price * (1 + 0.05) ** year
            future_prices.append(inflation_adjusted_price)
        
        st.subheader("Price Trend Over Next 3 Years")
        st.markdown("*Note: Future prices assume 10,000 km driven per year and 5% annual inflation, with static market conditions.*")
        price_data = pd.DataFrame({
            'Year': [str(current_year), str(current_year + 1), str(current_year + 2), str(current_year + 3)],
            'Price (₹)': future_prices
        })
        st.line_chart(price_data.set_index('Year')['Price (₹)'])

if __name__ == '__main__':
    main()
