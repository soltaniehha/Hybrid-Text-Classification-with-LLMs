# Hybrid Text Classification with LLMs

This project implements a hybrid approach to text classification, combining traditional ML with LLMs to classify tweets as literal or sarcastic. The system uses sentence transformers for embeddings, neural networks for classification, and GPT-4.1 Mini for data validation.

## Environment Requirements

- Python 3.12 (TensorFlow is not compatible with Python 3.13)
- Dependencies listed in `requirements.txt`:
  ```
  openai
  pandas
  nltk
  kaggle
  scikit-learn
  seaborn
  matplotlib
  tqdm
  joblib
  tensorflow
  keras
  sentence-transformers
  transformers
  umap-learn
  ```

## Project Structure

The project follows a sequential pipeline:

1. `1-Pre-processing.py`: Data Acquisition and Preprocessing
   - Downloads and processes [Sentiment140 dataset](https://www.kaggle.com/datasets/kazanova/sentiment140) (1.6M tweets)
   - Cleans text (URLs, mentions, hashtags)
   - Filters tweets (min 10 words)
   - Normalizes and tokenizes

2. `2A1-Synthetic-Text-Generation.py`: Create Synthetic Dataset
   - Uses GPT-4.1-mini to generate synthetic tweets
   - Creates balanced literal/sarcastic examples
   - Maintains realistic tweet properties
   - Cost: ~$0.50 for 3000 samples per class

3. `2A2-Train-Naive-Classifier.py`: Train Initial Model
   - Uses all-MiniLM-L6-v2 for embeddings
   - Two-phase training with early stopping
   - Architecture: Dense(64) → Dropout(0.5) → Dense(32) → Dropout(0.5) → Dense(2)

4. `2B-Apply-Naive-Model-to-Real-Data.py`: Validate on Real Data
   - Applies model to real tweets
   - Extracts top 5000 high-confidence examples per class
   - Generates UMAP visualization
   - Saves embeddings and predictions

5. `2C-Refine-Labels-with-LLM-Classifier.py`: Validate Predictions
   - Uses GPT-4.1-mini to validate predictions
   - Identifies true/false positives/negatives
   - Creates refined labeled dataset

6. `3-Build-a-Balanced-Training-Dataset.py`: Create Final Dataset
   - Combines synthetic and real data
   - Balances classes (sarcastic/literal)
   - Checks for class imbalance (>20%)
   - Creates shuffled training set

7. `4-Train-Final-Classifier.py`: Train Production Model
   - Uses all-mpnet-base-v2 for embeddings
   - Architecture: Dense(32) → Dropout(0.5) → Dense(8) → Dropout(0.5) → Dense(1)
   - Two-phase training with early stopping
   - Evaluates and analyzes misclassifications

## Setup Instructions

### 1. Environment Setup
```bash
conda create -n text_class python=3.10
conda activate text_class
pip install -r requirements.txt
```

### 2. API Setup
1. Kaggle: Download `kaggle.json` from account settings and place in project root
2. OpenAI: Create `openai_api_key.json` with your API key in project root

## Usage

Run the notebooks in sequence:
```bash
jupytext --to notebook *.py
jupyter notebook
```

## Dataset

The project uses the [Sentiment140 dataset](https://www.kaggle.com/datasets/kazanova/sentiment140) (1.6M tweets) as a starting point, transforming it into a literal/sarcastic classification task through synthetic data generation and LLM validation.
