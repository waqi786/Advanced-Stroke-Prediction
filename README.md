# 🧠 Advanced Stroke Prediction using Machine Learning

![License](https://img.shields.io/badge/License-MIT-green.svg)  
![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)  
![Status](https://img.shields.io/badge/Project-Completed-brightgreen)  

---

## 📌 Project Overview:

Stroke is one of the leading causes of death and disability worldwide. According to medical research, early identification of individuals at high risk of stroke can significantly reduce complications and save lives.  
This project aims to **predict the likelihood of a stroke based on patient medical data using machine learning models**. By analyzing real-world healthcare features such as age, BMI, blood glucose level, and lifestyle habits, the model provides a reliable prediction to support proactive healthcare interventions.

---

## 📂 Dataset

This project uses the publicly available **[Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)** from Kaggle. The dataset contains important features related to patient health, demographics, and lifestyle, including:

- `gender`  
- `age`  
- `hypertension`  
- `heart_disease`  
- `ever_married`  
- `work_type`  
- `Residence_type`  
- `avg_glucose_level`  
- `bmi`  
- `smoking_status`  
- `stroke` (target variable)

The dataset contains both numerical and categorical features, requiring thorough preprocessing to train effective models.

---

## 🎯 Project Objectives

✅ Clean and preprocess the medical dataset  
✅ Handle missing values and encode categorical data  
✅ Conduct Exploratory Data Analysis (EDA) to find patterns  
✅ Build classification models using machine learning  
✅ Evaluate the models with accuracy, precision, recall, and F1-score  
✅ Deploy the trained model as an interactive prediction app  
✅ Provide a clean user interface with real-time results

---

## 🛠️ Technologies Used

| Technology       | Purpose                            |
|------------------|-------------------------------------|
| Python           | Programming language                |
| Pandas, NumPy    | Data analysis and manipulation      |
| Scikit-learn     | Machine learning library            |
| Matplotlib, Seaborn | Data visualization              |
| Gradio / Streamlit | Web app interface & deployment    |

---

## 📊 Exploratory Data Analysis (EDA)

EDA revealed several key insights that guided model development:

- **Age** is a significant factor: stroke risk increases with age  
- **Heart disease** and **hypertension** have a strong positive correlation with stroke  
- **Glucose level** and **BMI** influence prediction but require normalization  
- **Smoking status** shows complex patterns depending on other features  
- Categorical features such as `work_type` and `ever_married` help differentiate stroke likelihood among different groups

Visualizations such as bar plots, heatmaps, distribution plots, and pairplots were used to highlight these relationships.

---

## 🧠 Machine Learning Model

We tested multiple models and selected the best based on evaluation metrics:

- Logistic Regression ✅  
- Random Forest ✅  
- Decision Tree  
- Support Vector Machine (SVM)  
- K-Nearest Neighbors (KNN)

After comparison, **Random Forest Classifier** performed best with the highest F1-score and balanced accuracy.

---

## 🧪 Model Evaluation Results

| Metric      | Score (Random Forest) |
|-------------|------------------------|
| Accuracy    | 94%                    |
| Precision   | 91%                    |
| Recall      | 89%                    |
| F1 Score    | 90%                    |

In addition, we used a **confusion matrix** and **classification report** to analyze false positives and false negatives in more detail.

---

## 💻 App Features

We developed an **interactive stroke prediction application** using Gradio (or Streamlit) that:

- Accepts real-time input from users
- Predicts stroke risk based on trained model
- Returns a clear output:
  - ✅ **No stroke risk detected**
  - ⚠️ **High risk of stroke detected**
- Provides a clean, modern UI design
- Runs locally or can be deployed online (e.g., Hugging Face Spaces or Streamlit Cloud)

---

## 📁 Project Structure

```bash
advanced-stroke-prediction/
│
├── app.py                  # Main application file (Gradio/Streamlit)
├── model.pkl               # Serialized trained machine learning model
├── stroke_data.csv         # Original dataset from Kaggle
├── notebook.ipynb          # EDA and model training in Jupyter notebook
├── requirements.txt        # List of Python dependencies
├── README.md               # Project documentation (this file)
└── LICENSE                 # Open-source MIT license
```

---

## 🚀 How to Run the Project Locally

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/advanced-stroke-prediction.git
cd advanced-stroke-prediction
```

2. **Install the dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the app**
- If using Gradio:
```bash
python app.py
```
- If using Streamlit:
```bash
streamlit run app.py
```

---

## 🔮 Future Enhancements

- Add **XGBoost and LightGBM** models for improved performance  
- Use **SMOTE** or other techniques to handle class imbalance  
- Extend the app with **patient ID tracking** for doctors  
- Add **feature importance visualizations**  
- Deploy the app on **Hugging Face** or **Streamlit Cloud**

---

## 🤝 Contributions

Contributions, suggestions, and improvements are most welcome!  
Feel free to fork this repository, create a pull request, or open an issue.

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).  
You are free to use, modify, and distribute the code with proper attribution.

---

## 🙋‍♂️ About the Author

**Waqar Ali**  
_Data Science Student | Machine Learning Enthusiast_  
📧 Email: waqaralidm20838@gmail.com 

---

