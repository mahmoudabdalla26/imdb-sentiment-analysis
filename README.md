# IMDb Sentiment Analysis

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Keras](https://img.shields.io/badge/Keras-TensorFlow-orange)

## Overview
This project performs **sentiment analysis on IMDb movie reviews**, classifying them as **positive** or **negative**. The dataset contains 50,000 labeled reviews, evenly split between positive and negative sentiments. The workflow includes **data preprocessing, exploratory data analysis, baseline modeling, deep learning sequence models (RNN, LSTM, GRU), hyperparameter optimization, and deployment via a Flask app**.

**Dataset:** [IMDb Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews/data)

## Key Features
- Clean and preprocess IMDb reviews.
- Tokenization and vectorization (TF-IDF & padded sequences).
- Train baseline ML models (Logistic Regression).
- Deep learning models: RNN, LSTM, GRU.
- Hyperparameter tuning with KerasTuner.
- Flask app for real-time sentiment prediction.

## Project Structure
```
IMDb-Sentiment-Analysis/
│
├── data/
│ ├── IMDB Dataset.csv
│ ├── X_train.csv
│ ├── X_test.csv
│ ├── y_train.csv
│ └── y_test.csv
│
├── models/
│ ├── baseline_logreg.pkl
│ ├── GRU_model.h5
│ ├── gru_tuned_model.h5
│ ├── gru_tuned_results.json
│ ├── LSTM_model.h5
│ ├── RNN_model.h5
│ ├── tfidf_vectorizer.pkl
│ └── tokenizer.pkl
│
├── Notebooks/
│ ├── IMDb_Sentiment_Analysis.ipynb
│ ├── Vectorization_and_Baseline_Model.ipynb
│ ├── Sequence_Models.ipynb
│ └── Best_Model_Optimization.ipynb
│
├── templates/
│ └── index.html
├── app.py
└── requirements.txt
```


## Model Comparison
| Model | Accuracy |
|-------|----------|
| RNN   | 0.4994   |
| LSTM  | 0.8454   |
| GRU   | 0.8833   |

The **GRU model** achieved the highest accuracy and was selected for deployment.

## Installation & Setup

1. Clone the repository:
```bash
git clone <repository_url>
cd imdb-sentiment-analysis
```

2.Create a virtual environment (optional):
```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```
3.Install dependencies:
```bash
pip install -r requirements.txt
```

4.Run the Flask app:
```bash
python app.py
```
Visit http://127.0.0.1:5000 in your browser.

## Notebooks Overview

- **IMDb_Sentiment_Analysis.ipynb**: EDA and preprocessing, cleaning text, encoding labels, train/test split.  
- **Vectorization_and_Baseline_Model.ipynb**: TF-IDF and tokenized sequences, baseline ML model, saving vectorizers and models.  
- **Sequence_Models.ipynb**: Train RNN, LSTM, GRU models, evaluate and compare performance.  
- **Best_Model_Optimization.ipynb**: Hyperparameter tuning of the GRU model using KerasTuner, save optimized model.  

## Deployment

- **app.py**: Flask application for live sentiment prediction.  
- **templates/index.html**: User interface for the web app.  

## Dependencies

- Python 3.8+  
- pandas  
- numpy  
- scikit-learn  
- tensorflow / keras  
- Flask  
- KerasTuner  
- nltk  

*(Full list available in `requirements.txt`)*

## License

This project is open-source under the MIT License.  

## Acknowledgments

- [IMDb Dataset of 50K Movie Reviews on Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews/data)  
- Keras & TensorFlow documentation  
- Scikit-learn documentation
