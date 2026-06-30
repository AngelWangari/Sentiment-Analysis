# 🧠 Mental Health Sentiment Analysis

## Project Overview

This project evaluates and compares multiple machine learning models to identify the most robust approach for **neutral sentiment detection** in mental health-related customer reviews. The study also includes a web-based application for real-time sentiment prediction.

## Project Title

**Evaluation and Comparative Analysis of Sentiment Analysis Models to Identify the Most Robust Approach for Neutral Sentiment Detection in Mental Health–Related Customer Reviews with Web Application Integration**

## Features

* Text preprocessing and data cleaning
* TF-IDF feature extraction
* Hyperparameter tuning using GridSearchCV
* Comparative evaluation of:

  * Logistic Regression
  * Naive Bayes
  * Linear SVM
* Hybrid LR + SVM Ensemble model
* Exploratory Data Analysis (EDA) and visualizations
* Interactive Gradio web application for real-time sentiment prediction
* Streamlit dashboard for model performance visualization

## Dataset

The project uses a mental health text dataset containing reviews categorized into three sentiment classes:

* Very Negative
* Negative
* Neutral

## Technologies Used

* Python
* Scikit-learn
* Pandas
* NumPy
* Matplotlib & Seaborn
* Plotly
* Streamlit
* Gradio
* Joblib

## Model Performance

| Model                        | Accuracy   |
| ---------------------------- | ---------- |
| Logistic Regression          | 88.49%     |
| Linear SVM                   | 88.46%     |
| Naive Bayes                  | 79.03%     |
| **Hybrid LR + SVM Ensemble** | **89.60%** |

The Hybrid LR + SVM Ensemble achieved the highest performance and was selected for deployment in the web application.

## Project Structure

```text
project-dsa/
│── models/
│── app.py
│── predict.py
│── trainmodel.py
│── trainensemble.py
│── sentiment_dashboard.py
│── edavisuals.py
│── requirements.txt
│── README.md
```

## Running the Project

1. Clone the repository.
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Train the models:

```bash
python trainmodel.py
python trainensemble.py
```

4. Launch the Gradio application:

```bash
python app.py
```

5. Launch the Streamlit dashboard:

```bash
streamlit run sentiment_dashboard.py
```

## Author

Developed as a Data Science research project focusing on robust sentiment analysis techniques for mental health-related text classification.
