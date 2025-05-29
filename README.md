# 🏡 House Price Prediction using Linear Regression

A complete end-to-end machine learning pipeline for predicting house prices using **Linear Regression**. This project involves loading, preprocessing, encoding, training, evaluating, visualizing, and interpreting a regression model with scikit-learn.

---

## 📁 Project Structure

.
├── 01_data_loading_cleaning.py # Load data and check for missing values
├── 01_load_preproces.py # Preprocessing (overlap/duplicate file, can remove if unused)
├── 02_encode_split.py # Encode categorical features and split data
├── 03_train_model.py # Train the Linear Regression model
├── 04_evaluate_model.py # Evaluate model (MAE, MSE, R²)
├── 05_plot_results.py # Plot actual vs predicted results
├── 06_interpret_coeffs.py # Interpret model coefficients (feature importance)
├── housedata.zip # Raw dataset
└── README.md


---

## 🧠 Objective

To accurately predict house prices based on features like area, number of rooms, amenities, and more using a **Linear Regression** model.

---

## 🛠️ Technologies Used

- Python 3
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

---

## 🚀 How to Run the Project

1. **Install Dependencies**  
   Make sure Python is installed and run:

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
2. **Run the Scripts in Order**
Each script handles one part of the ML pipeline:
python 01_data_loading_cleaning.py
python 02_encode_split.py
python 03_train_model.py
python 04_evaluate_model.py
python 05_plot_results.py
python 06_interpret_coeffs.py

📊 Evaluation Metrics
Mean Absolute Error (MAE): Measures average magnitude of errors.
Mean Squared Error (MSE): Penalizes larger errors.
R² Score: Indicates model accuracy (~0.65 in this project).

📌 Key Highlights
✅ Cleaned and validated the dataset
✅ Encoded categorical features
✅ Trained Linear Regression model
✅ Evaluated with MAE, MSE, and R²
✅ Visualized predictions vs actuals
✅ Interpreted model coefficients to find feature importance

🧑‍💻 Author
Sanskriti Anya
