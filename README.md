# AIML Project Suite

A comprehensive collection of machine learning and data analysis projects built with Python, featuring multiple applications including placement prediction, car price estimation, and various data analysis tasks.

## 📋 Projects Overview

### 1. **Placement Prediction System** (`app.py`)
A Streamlit-based web application that predicts student placement outcomes using Logistic Regression.

**Features:**
- Input student metrics (IQ, Previous Grade, CGPA, Performance, Internships, ECA Score, Communication Score, Projects)
- Real-time placement prediction
- Interactive user interface

**Dataset:** `hello.csv` - Student performance and placement data

---

### 2. **Car Price Prediction** 
Machine learning models for car price estimation using the CardEkho dataset.

**Files:**
- `cardekho_data.csv` - Car pricing dataset
- `car_price_model.pkl` - Trained pricing model
- Associated preprocessing pickle files (`brand_model_map.pkl`, `le_brand.pkl`, `le_fuel.pkl`, `le_model.pkl`, `le_seller.pkl`, `scaler.pkl`)

---

### 3. **Job Fraud Detection**
Classification model to identify fraudulent job postings.

**Files:**
- `FAkeJOb.xlsx` - Fake job dataset
- `job_fraud_model.pkl` - Trained fraud detection model

---

### 4. **Data Analysis Projects**
Multiple exploratory data analysis notebooks:
- `day2.ipynb` - Data exploration exercises
- `day3.ipynb` - Data visualization
- `day5.ipynb` - Statistical analysis
- `day9.ipynb` - Advanced analysis
- `day11.ipynb` - Complex datasets
- `knn.ipynb` - K-Nearest Neighbors implementation
- `image.ipynb` - Image processing analysis
- `ds.ipynb` - Data science fundamentals
- `new.ipynb` - Additional analysis

---

## 📊 Datasets Included

- **cardekho_data.csv** - Car pricing information
- **hello.csv** - Student placement data
- **cereal.csv** - Cereal nutrition data
- **Student_Performance.csv** - Academic performance records
- **heart.csv** - Heart disease data
- **indian_food.csv** - Indian food dataset
- **imdb_data.csv** - IMDB movie data
- **phishing_site_urls.csv** - Phishing dataset
- **dublin_bus_data.csv** - Public transport data
- **employee.csv** - Employee records
- **Food_Preference.csv** - Food preferences
- **Iris - all-numbers.csv** - Iris flower classification data
- **Cleaned_Advertisements.csv** - Advertisement data

---

## 🚀 Getting Started

### Prerequisites
- Python 3.7 or higher
- pip (Python package manager)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd AIML
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

---

## 💻 Usage

### Running the Placement Prediction App

```bash
streamlit run app.py
```

The app will open in your default web browser at `http://localhost:8501`

**Input the following information:**
- IQ score
- Previous Grade
- CGPA (Cumulative GPA)
- Performance rating
- Number of Internships
- ECA Score
- Communication Score
- Number of Projects

Click **"Analyze"** to get the placement prediction.

---

### Running Jupyter Notebooks

```bash
jupyter notebook
```

Then select the notebook file you want to explore.

---

## 📁 Project Structure

```
AIML/
├── app.py                          # Main Streamlit placement prediction app
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── Models (Trained)
│   ├── car_price_model.pkl        # Car price prediction model
│   ├── job_fraud_model.pkl        # Job fraud detection model
│   └── *.pkl                      # Preprocessing and scaling files
│
├── Data Files (CSV)
│   ├── cardekho_data.csv
│   ├── hello.csv
│   ├── cereal.csv
│   ├── Student_Performance.csv
│   └── [other CSV files]
│
├── Notebooks (Jupyter)
│   ├── day2.ipynb through day11.ipynb
│   ├── knn.ipynb
│   └── [other analysis notebooks]
│
└── Python Scripts
    ├── abba.py
    ├── jfd.py
    ├── piyush.py
    └── project.py, project2.py
```

---

## 🔧 Technologies Used

- **Python** - Programming language
- **Streamlit** - Web app framework
- **Pandas** - Data manipulation
- **Scikit-learn** - Machine learning
- **Jupyter Notebook** - Interactive analysis

---

## 📈 Models & Algorithms

- **Logistic Regression** - Placement prediction
- **K-Nearest Neighbors (KNN)** - Classification tasks
- **Preprocessing techniques** - Scaling, encoding, feature engineering

---

## 📝 Notes

- Pickle files (.pkl) contain pre-trained models and preprocessing transformers
- CSV files are provided for training and testing
- Jupyter notebooks include exploratory data analysis and model experimentation

---

## 🤝 Contributing

Feel free to fork this project and submit pull requests for improvements.

---

## 📧 Contact

For questions or suggestions, please reach out!

---

## 📄 License

This project is open source and available under the MIT License.
