---
# Sentiment Analysis on Twitter US Airline Sentiment Dataset

This repository contains a project focused on sentiment analysis of airline-related tweets using the **Twitter US Airline Sentiment Dataset**. The primary objective is to explore and apply text preprocessing techniques similar to those provided by the **Natural Language Toolkit (NLTK)** and to build a machine learning pipeline for sentiment classification.
---

## Project Objective

The goal of this project is twofold:

1. Develop a deeper understanding of text preprocessing methods, such as tokenization, stemming, lemmatization, stopword removal, and other techniques to prepare text data for analysis.
2. Apply these techniques to analyze the sentiments expressed in tweets about various airlines and classify them into categories such as positive, neutral, or negative.

This project encompasses tasks such as data preprocessing, visualization, model training, evaluation, and insights generation.

---

## Dataset

- **Name:** [Twitter US Airline Sentiment Dataset](https://www.kaggle.com/crowdflower/twitter-airline-sentiment)
- **Description:** A collection of tweets directed at U.S. airlines, each labeled with sentiment categories (`positive`, `neutral`, `negative`) and additional metadata.
- **Usage:** The dataset is used for training and evaluating sentiment analysis models.

---

## Repository Structure

```
.
├── LICENSE
├── README.md
├── Tweets.csv
├── __pycache__
│   ├── part1.cpython-313.pyc
│   ├── part2.cpython-313.pyc
│   ├── part5.cpython-313.pyc
│   └── plot.cpython-313.pyc
├── classification_report
│   ├── classification_report_0.csv
│   └── classification_report_184910116753212.csv
├── confusion_csv
│   ├── confusion_matrix_0.csv
│   └── confusion_matrix_184910116753212.csv
├── confusion_matrix
│   ├── confusion_matrix_0.png
│   └── confusion_matrix_184910116753212.png
├── cv_results.csv
├── figures
│   ├── Part1A-Histogram-American.png
│   ├── Part1A-Histogram-Delta.png
│   ├── Part1A-Histogram-Southwest.png
│   ├── Part1A-Histogram-US Airways.png
│   ├── Part1A-Histogram-United.png
│   ├── Part1A-Histogram-Virgin America.png
│   └── Part1B.jpg
├── main.py
├── part1.py
├── part2.py
├── part3.py
├── part5.py
├── plot.py
├── requirements.txt
├── subset_dataset.csv
└── tweet_us_airline_data_processing.pdf
```

## Tasks Overview

### Part 1: Data Analysis and Visualization

- **Description:**
  - Analyze sentiment distribution by airline.
  - Visualize results using histograms.
- **Entrypoint:** `main.py`
- **Outputs:**
  - Histograms for each airline (e.g., `Part1A-Histogram-American.png`).

### Part 2: Data Preprocessing

- **Description:**
  - Prepare the dataset for machine learning tasks (e.g., cleaning, feature engineering).
- **Location:** Detailed in `reports/`.

### Part 3: Model Training and Evaluation

- **Description:**
  - Train classification models on the dataset.
  - Evaluate performance using confusion matrices and classification reports.
- **Entrypoint:** `part3.py`
- **Outputs:**
  - Confusion matrices (both CSV and image formats).
  - Classification reports (CSV).

### Part 4: Detailed Reports

- **Description:**
  - Includes insights and observations from tasks.
- **Location:** Available in `reports/`.

### Part 5: Post-Model Evaluation and Insights

- **Description:**
  - Additional analysis on model predictions and insights extraction.
- **Entrypoint:** `main.py`

---

## How to Use

### Prerequisites

- Python 3.13 or higher
- Required libraries (install via `requirements.txt` if available)

### Steps to Execute

1. Clone the repository.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the command for part 1 & 5
   ```
   python main.py
   ```
4. Run the command for part 3 model prediction
   ```
   python part3.py
   ```
