# CODSOFT_TASK1 — Movie Genre Classification

A machine learning project that predicts a movie's genre from its plot description, built as part of the CodSoft ML Internship (Task 1).

## Problem Statement

Create a machine learning model that can predict the genre of a movie based on its plot summary, using TF-IDF for text representation and classifiers like Naive Bayes, Logistic Regression, and Support Vector Machines.

Dataset: [Kaggle - Genre Classification Dataset](https://www.kaggle.com/datasets/hijest/genre-classification-dataset-imdb)

## Project Structure

| File | Description |
|---|---|
| `genre training.ipynb` | Loads and explores the training data, cleans text, builds TF-IDF features, trains and compares 3 models (Logistic Regression, Multinomial Naive Bayes, SVM), and saves the best model |
| `genre testing.ipynb` | Loads the saved model + vectorizer and evaluates them on the held-out `test_data.txt`, comparing predictions against `test_data_solution.txt` |
| `train_data.txt` | Training data (`ID ::: TITLE ::: GENRE ::: DESCRIPTION`) |
| `test_data.txt` | Test data without genre labels (`ID ::: TITLE ::: DESCRIPTION`) |
| `test_data_solution.txt` | Ground truth genres for the test set |
| `svm_model.pkl` | Saved trained SVM model (best performing) |
| `tfidf_vectorizer.pkl` | Saved fitted TF-IDF vectorizer |

## Approach

1. **Data Exploration** — 27 genre classes, heavily imbalanced (from ~130 to ~13,600 samples per class)
2. **Text Preprocessing** — lowercasing and punctuation removal on movie descriptions
3. **Feature Extraction** — TF-IDF vectorization (with `stop_words='english'`)
4. **Model Comparison** — trained and evaluated 3 models with class balancing:
   - Logistic Regression (`class_weight='balanced'`)
   - Multinomial Naive Bayes (`fit_prior=False`)
   - Linear SVM (`class_weight='balanced'`)
5. **Model Selection** — SVM performed best across accuracy, macro F1, and weighted F1
6. **Final Validation** — evaluated the saved SVM model on the untouched `test_data.txt`, confirming consistent performance on unseen data

## Results

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|
| Logistic Regression (balanced) | 49.4% | 0.37 | 0.51 |
| Multinomial Naive Bayes | 45.7% | 0.06 | 0.33 |
| **SVM (balanced)** | **55.3%** | **0.38** | **0.55** |

Final model (SVM) evaluated on the real test set (`test_data.txt`): **55.3% accuracy**, consistent with the train-test split results — indicating the model generalizes well without overfitting.

## Tech Stack

- Python, pandas, scikit-learn
- TF-IDF (`TfidfVectorizer`)
- Logistic Regression, Multinomial Naive Bayes, Linear SVC
- joblib (model persistence)

## How to Run

1. Clone the repo
2. Open `genre training.ipynb` to see the full training pipeline, or
3. Open `genre testing.ipynb` to load the saved model and evaluate directly on test data

## Author

**Purva Jain**  
[GitHub](https://github.com/darkmonnzz) | [LinkedIn](www.linkedin.com/in/purva-jain-83901b286)
