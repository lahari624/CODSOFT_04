📈 Advertising Sales Prediction
This project predicts sales based on advertising budgets for TV, Radio, and Newspaper using regression models.

📂 Dataset
The dataset used is advertising.csv, which includes:

   TV: Budget spent on TV ads

   Radio: Budget spent on radio ads

   Newspaper: Budget spent on newspaper ads

   Sales: Product sales in thousands of units


🧰 Technologies Used
Python 3.x

Pandas, NumPy — Data handling

Matplotlib, Seaborn — Data visualization

Scikit-learn — Machine learning

Joblib — Saving the model

🧪 How to Run This Project

🔧 1. Clone the Repository:
git clone https://github.com/your-username/advertising-sales-prediction.git

cd advertising-sales-prediction

📦 2. Install Required Libraries

Make sure Python is installed, then run:

pip install -r requirements.txt

Typical libraries needed:

pandas

numpy

matplotlib

seaborn

scikit-learn

joblib

🚀 3. Run the Notebook

You can open and run the project in Jupyter Notebook:

jupyter notebook advertising_sales_prediction.ipynb

Or run it using VS Code, Google Colab, or any Python IDE of your choice.

🧠 4. Predict on New Data

Once the model is trained, you can test predictions using:

joblib.load("sales_prediction_model.pkl")

model.predict(new_data)

🔍 Exploratory Data Analysis

No missing values in the dataset

Correlation observed between TV, Radio, and Sales

Visualization: pairplot, heatmap

🧠 Models Used

Linear Regression

MAE: 1.27, R² Score: 0.91

Random Forest Regressor

MAE: 0.92, R² Score: 0.95

Best model based on performance

🔮 Predictions Example

TV = 300, Radio = 20, Newspaper = 10 → Predicted Sales: 20.24

TV = 10, Radio = 10, Newspaper = 100 → Predicted Sales: 6.16

💾 Model Saving

The best model is saved as:

sales_prediction_model.pkl

You can reload it for deployment or batch prediction.

✅ Future Improvements

Add cross-validation

Hyperparameter tuning (e.g. with GridSearchCV)

Try more models like XGBoost, SVR

📁 Project Structure

advertising-sales-prediction/
├── advertising.csv
├── sales.ipynb
├── sales_prediction_model.pkl
├── README.md
└── requirements.txt
