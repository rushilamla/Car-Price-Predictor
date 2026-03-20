# Car Price Predictor

A machine learning project that predicts car prices using the CardEkho dataset with pre-trained models and preprocessing pipelines.

## 📋 Project Overview

This project uses machine learning to estimate car prices based on various features including brand, model, fuel type, seller type, and other vehicle attributes. The model has been trained and saved for easy inference on new data.

## 📊 Dataset

**cardekho_data.csv** - Contains comprehensive car pricing information with features such as:
- Brand and Model
- Fuel Type
- Seller Type
- Transmission
- Mileage
- Engine Power
- Price (target variable)

## 🎯 Features

- Pre-trained machine learning model (`car_price_model.pkl`)
- Preprocessing transformers for scaling and encoding
- Ready-to-use label encoders for categorical features
- Easy-to-load predictions on new car data

## 📁 Project Files

```
├── cardekho_data.csv          # Dataset with car information
├── car_price_model.pkl        # Trained price prediction model
├── brand_model_map.pkl        # Brand-to-model mapping
├── le_brand.pkl               # Label encoder for brands
├── le_fuel.pkl                # Label encoder for fuel types
├── le_model.pkl               # Label encoder for car models
├── le_seller.pkl              # Label encoder for seller types
├── le_trans.pkl               # Label encoder for transmission
├── scaler.pkl                 # Feature scaler
└── README.md                  # This file
```

## 🚀 Getting Started

### Prerequisites
- Python 3.7 or higher
- pip (Python package manager)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd car-price-predictor
   ```

2. **Create a virtual environment (optional but recommended)**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage

### Loading and Using the Model

```python
import pickle
import pandas as pd

# Load the pre-trained model and preprocessing objects
with open('car_price_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('le_brand.pkl', 'rb') as f:
    le_brand = pickle.load(f)

# Prepare your data
# (encode categorical variables and scale features)

# Make predictions
predicted_price = model.predict(X_new)
```

## 🔧 Technologies Used

- **Python** - Programming language
- **Pandas** - Data manipulation
- **Scikit-learn** - Machine learning
- **Pickle** - Model serialization

## 📈 Model Details

- Pre-trained regression model
- Includes feature preprocessing and scaling
- Label encoders for categorical variables
- Ready for immediate predictions

## 📝 Notes

- All pickle files (.pkl) contain pre-trained models and preprocessing transformers
- Ensure features are in the correct order when making predictions
- The scaler should be applied before feeding data to the model

## 🤝 Contributing

Feel free to fork this project and submit pull requests for improvements.

## 📄 License

This project is open source and available under the MIT License.
