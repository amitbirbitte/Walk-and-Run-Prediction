# Walk vs Run Classification 🏃‍♂️🚶‍♀️

This project builds a machine learning model to classify human activity
as **walking** or **running** using motion-based numerical features. It
demonstrates a complete end-to-end ML workflow including data
preprocessing, exploratory data analysis, model training, evaluation,
cross-validation, and final model comparison.

## 📌 Project Overview

The goal of this project is to accurately predict whether a given
activity corresponds to walking or running. Multiple binary
classification models are trained and evaluated to identify the most
reliable and best-performing model.

## 📊 Dataset Description

-   Contains motion-related numerical features.
-   Includes a binary target variable:
    -   `0` → Walking\
    -   `1` → Running
-   Used for supervised binary classification.

## ⚙️ Workflow

1.  Data loading and preprocessing\
2.  Exploratory Data Analysis (EDA)\
3.  Model training (Logistic Regression, KNN, SVM, Random Forest,
    XGBoost, MLP)\
4.  Model evaluation using Accuracy, Precision, Recall, and F1-score\
5.  Cross-validation for model stability\
6.  Model comparison and final model selection

## 🧠 Models Used

-   Logistic Regression\
-   K-Nearest Neighbors (KNN)\
-   Support Vector Machine (SVM)\
-   Random Forest Classifier\
-   XGBoost Classifier\
-   Multi-Layer Perceptron (Neural Network)

## 📈 Evaluation Metrics

-   Accuracy\
-   Precision\
-   Recall\
-   F1-score\
-   Cross-validation mean accuracy

## 🏆 Final Outcome

The final model was selected based on high evaluation performance and
consistent cross-validation results, ensuring good generalization on
unseen data.

## 🛠️ Tools & Technologies

-   Python\
-   NumPy, Pandas\
-   Scikit-learn\
-   XGBoost\
-   Jupyter Notebook

## 📂 Project Structure

    ├── data/
    │   └── walkrun.csv
    ├── notebooks/
    │   └── WalkRun_Classification.ipynb
    ├── model/
    │   └── final_model.pkl
    └── README.md

## 🚀 Future Improvements

-   Hyperparameter tuning
-   Feature selection or dimensionality reduction
-   Deployment using Flask or Streamlit

------------------------------------------------------------------------

⭐ If you find this project useful, feel free to star the repository!
