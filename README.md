# Spam or Ham Text Classifier

A Machine Learning-based Natural Language Processing (NLP) system that automatically classifies text messages into **Spam** or **Ham** (legitimate). 

This project demonstrates a complete machine learning pipeline, progressing from an initial baseline implementation to a highly optimized final model through hyperparameter tuning.

## 🚀 Key Features
* Full text preprocessing and tokenization pipeline.
* Term Frequency-Inverse Document Frequency (TF-IDF) feature extraction.
* Optimized hyperparameter tuning using a custom smoothing parameter (α).
* Comprehensive performance reporting via detailed evaluation metrics.
* Interactive CLI interface to test the final model with custom input sentences.

## 🛠️ Tech Stack & Core Concepts
* **Python 3.13**: Core runtime environment.
* **Scikit-learn**: Used for feature extraction, model training, and performance reporting.
* **TF-IDF Vectorizer**: Converts raw text into a weighted matrix based on word importance.
* **Multinomial Naive Bayes**: A probabilistic classifier optimized with custom smoothing parameters for discrete text features.

---

## 📈 Optimization Strategy & Hyperparameter Tuning

The initial baseline model achieved a perfect **100% Spam Precision** but suffered from an inadequate **Spam Recall (35.44%)**, failing to detect a significant portion of spam messages. 

To resolve this issue, hyperparameter tuning was applied to adjust the additive smoothing parameter:
* **Hyperparameter Applied**: `alpha = 0.01` (Laplace/Lidstone smoothing)
* **Impact**: Significantly lowered the model's conservative threshold for identifying minority-class vocabulary, drastically expanding spam sensitivity without sacrificing prediction security.

---

## 📊 Model Performance Comparison

The dataset consists of **479 Ham** and **79 Spam** test samples. Tuning the model yielded massive, production-grade performance enhancements:

### Metric Comparison

| Evaluation Metric | Initial Baseline Model | Final Tuned Model (`alpha = 0.01`) |
| :--- | :---: | :---: |
| **Overall Accuracy** | 90.86% | **99.28%** |
| **Spam Precision** | 100.0% | **100.0%** |
| **Spam Recall** | 35.44% | **94.94%** |
| **Spam F1-Score** | 52.34% | **97.40%** |

### Confusion Matrix Comparison

**Baseline Model:**
```text
[[ 479    0 ]   <-- True Ham (479), False Spam (0)
 [  51   28 ]]  <-- False Ham (51), True Spam (28)
```

**Final Tuned Model:**
```text
[[ 479    0 ]   <-- True Ham (479), False Spam (0)
 [   4   75 ]]  <-- False Ham (4)  , True Spam (75)
```
* **Performance Takeaway**: By tuning α to `0.01`, the final model safely extracted **75 out of 79 spam messages** (up from just 28), reducing missed spam by over **92%** while confidently retaining its flawless **100% precision score**.

---

### Final Classification Report
```text
              precision    recall  f1-score   support

         Ham       0.99      1.00      1.00       479
        Spam       1.00      0.95      0.97        79

    accuracy                           0.99       558
   macro avg       1.00      0.97      0.98       558
weighted avg       0.99      0.99      0.99       558
```

---

## 💻 Installation & Usage

### Prerequisites
Make sure you have Python installed. Then, install the required development packages:
```bash
pip install scikit-learn pandas numpy
```

### Running the Project
1. Clone this repository:
```bash
git clone https://github.com
```
2. Navigate to the project root directory and run the script:
```bash
python spam_detection.py
```

### Testing Custom Messages
The program includes an interactive command-line utility. Run the script and type any custom email content when prompted:
```text
TEST WITH YOUR OWN EMAIL
--------------------------------------------------

Enter an email message: hello click this

Result: HAM
```
