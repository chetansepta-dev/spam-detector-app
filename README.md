# 📧 Email Spam Detection using TF-IDF and Multinomial Naive Bayes

> An end-to-end machine learning project that classifies email messages as **Spam** or **Ham** using **TF-IDF feature extraction** and a tuned **Multinomial Naive Bayes** classifier, with an interactive **Streamlit web application** and online deployment.

## 🌐 Live Demo

**Live Streamlit App:** <https://spam-detector-app-qryxfnmateenhtxhwgor28.streamlit.app/>

**GitHub Repository:** <https://github.com/chetansepta-dev/spam-detector-app.git>

---

## 📌 Overview

Email spam is a common problem in digital communication. Spam messages may contain unwanted advertisements, fraudulent offers, phishing attempts, or other suspicious content.

This project develops a machine-learning based **Email Spam Detection System** that learns patterns from labelled email messages and predicts whether a new email is:

- 🟢 **Ham** — legitimate email
- 🔴 **Spam** — unwanted or suspicious email

The project follows an end-to-end workflow:

**Dataset → Preprocessing → Train/Test Split → TF-IDF → Naive Bayes → Hyperparameter Tuning → Evaluation → Model Saving → Streamlit → Deployment**

The final tuned model achieved strong performance on the held-out test set.

---

## ✨ Features

- Email dataset loading and preparation
- Text preprocessing and cleaning
- Duplicate handling
- Stratified train/test splitting
- TF-IDF text feature extraction
- Multinomial Naive Bayes classification
- Hyperparameter tuning of the Naive Bayes `alpha` parameter
- 5-fold stratified cross-validation
- Model comparison during development
- Accuracy, Precision, Recall and F1-score evaluation
- Confusion matrix visualization
- ROC-AUC analysis
- Precision-Recall analysis
- Model serialization using Joblib
- Interactive Streamlit application
- Online deployment
- GitHub version control

---

## 🏗️ Workflow

```text
Dataset
   ↓
Data Preprocessing
   ↓
Stratified Train/Test Split
   ↓
TF-IDF Feature Extraction
   ↓
Multinomial Naive Bayes
   ↓
Hyperparameter Tuning
   ↓
Cross-Validation
   ↓
Final Test Evaluation
   ↓
Model + Vectorizer Saving
   ↓
Streamlit Application
   ↓
Online Deployment
```

---

## 📊 Dataset

The project uses the **Apache SpamAssassin public email corpus**, containing examples of legitimate (`ham`) and spam emails.

The data preparation pipeline loads the email messages, creates the corresponding labels, cleans the records, removes duplicates where applicable, and prepares the data for machine learning.

The final held-out test set used for the reported evaluation contained:

- **Ham:** 479
- **Spam:** 79
- **Total:** 558

The test set is kept separate from model training and is used for the final performance evaluation.

---

## 🧹 Data Preprocessing

The raw email data is prepared before feature extraction. The project includes steps such as:

1. Loading the email files
2. Extracting email text
3. Handling empty or invalid records
4. Removing duplicate records
5. Preparing class labels
6. Converting labels into machine-readable form
7. Performing a stratified train/test split

Keeping the test set separate helps prevent information from the final evaluation data from influencing model training.

---

## 🔤 TF-IDF Feature Extraction

Machine-learning algorithms require numerical features rather than raw text. The project therefore uses **Term Frequency-Inverse Document Frequency (TF-IDF)** to transform email text into numerical vectors.

The final pipeline uses settings including:

```text
lowercase=True
stop_words="english"
ngram_range=(1, 2)
min_df=2
max_df=0.95
sublinear_tf=True
```

For the final split used in the project:

```text
Training TF-IDF shape: (2232, 59673)
Testing TF-IDF shape : (558, 59673)
Number of features   : 59673
```

TF-IDF gives greater importance to terms that are useful for distinguishing documents while reducing the influence of terms that occur very frequently across the corpus.

---

## 🤖 Model Selection

During development, multiple text-classification algorithms were considered using the same general TF-IDF based approach:

- Multinomial Naive Bayes
- Logistic Regression
- Linear Support Vector Machine

The final selected model was **Multinomial Naive Bayes**, which provided the strongest overall performance for the evaluated dataset and setup.

---

## ⚙️ Hyperparameter Tuning

The main Multinomial Naive Bayes hyperparameter tuned in this project was the smoothing parameter:

```text
alpha
```

The project evaluated multiple values, including:

```text
0.01, 0.05, 0.10, 0.20, 0.50, 1.00, 2.00
```

A **5-fold Stratified Cross-Validation** procedure was used during tuning.

### Best parameter

```text
Best alpha = 0.01
```

Cross-validation results for the selected value were:

| Metric | Cross-Validation Score |
|---|---:|
| Accuracy | 98.92% |
| Precision | 99.00% |
| Recall | 93.33% |
| F1-score | 96.08% |

After selecting `alpha=0.01`, the final model was trained and evaluated on the held-out test set.

---

## 📈 Final Model Performance

The tuned Multinomial Naive Bayes model achieved the following results on the held-out test set:

| Metric | Score |
|---|---:|
| Accuracy | **99.28%** |
| Spam Precision | **100.00%** |
| Spam Recall | **94.94%** |
| Spam F1-score | **97.40%** |

### What the metrics mean

- **Accuracy — 99.28%:** percentage of all test emails classified correctly.
- **Spam Precision — 100.00%:** every email predicted as spam in this test set was actually spam.
- **Spam Recall — 94.94%:** the model detected 94.94% of the actual spam emails in the test set.
- **Spam F1-score — 97.40%:** harmonic mean of spam precision and recall, showing a strong balance between the two.

---

## 🔲 Confusion Matrix

The final confusion matrix was:

```text
                 Predicted
              Ham       Spam

Actual Ham     479        0
Actual Spam      4       75
```

Therefore:

- True Negatives (TN) = 479
- False Positives (FP) = 0
- False Negatives (FN) = 4
- True Positives (TP) = 75

The model made **0 false-positive spam classifications** on this held-out test set and missed **4 spam messages**.

---

## 📊 Before vs After Tuning

| Metric | Before Tuning | After Tuning |
|---|---:|---:|
| Accuracy | 90.86% | **99.28%** |
| Precision | 100.00% | **100.00%** |
| Recall | 35.44% | **94.94%** |
| F1-score | 52.34% | **97.40%** |

The most significant improvement was in **spam recall**, which increased from 35.44% to 94.94%. This substantially improved the spam F1-score from 52.34% to 97.40%.

---

## 📈 ROC-AUC

ROC analysis was used to evaluate model discrimination across classification thresholds.

```text
Before Tuning ROC-AUC ≈ 0.9864
After Tuning  ROC-AUC ≈ 0.9998
```

The tuned model showed excellent class separation on the evaluated test data.

---

## 📉 Precision-Recall Analysis

A Precision-Recall curve was also generated to examine the trade-off between identifying more spam messages and maintaining high precision.

This is especially useful for spam detection because class-specific performance is more informative than accuracy alone.

---

## 💾 Model Saving

After final training, the trained components are saved so the Streamlit application can make predictions without retraining the model every time.

The project uses **Joblib** for model serialization. The project includes artifacts such as:

```text
spam_model_v1.pkl
tfidf_vectorizer.pkl
```

These files represent the trained model and text feature transformation required for prediction.

---

## 🖥️ Streamlit Application

An interactive web application was developed using **Streamlit**.

The prediction flow is:

```text
User enters email
       ↓
TF-IDF transformation
       ↓
Saved Naive Bayes model
       ↓
Prediction
       ↓
HAM / SPAM
```

The application allows users to test new email messages without directly running the training pipeline.

---

## 🌐 Deployment

The Streamlit application was deployed online, allowing the model to be accessed through the internet rather than only from the local development machine.

This demonstrates the transition from a machine-learning experiment to an accessible application:

```text
Machine Learning Model
        ↓
Python Training Pipeline
        ↓
Saved Model
        ↓
Streamlit Application
        ↓
Online Deployment
```

**Live Application:** <https://spam-detector-app-qryxfnmateenhtxhwgor28.streamlit.app/>

---

## 🐙 GitHub

The project source code is maintained in a public GitHub repository.

**Repository:** (https://github.com/chetansepta-dev/spam-detector-app.git)

The repository can contain the training code, Streamlit application, requirements, documentation, evaluation plots, and other project resources.

> **Security note:** Never commit API keys, passwords, private credentials, or other secrets to a public repository. Also avoid committing unnecessarily large raw datasets or generated files when they can be downloaded or reproduced separately.

---

## 📁 Suggested Repository Structure

```text
email_spam_detection/
│
├── spam_detection.py
├── app.py
├── requirements.txt
├── README.md
│
├── spam_model_v1.pkl
├── tfidf_vectorizer.pkl
│
├── graphs/
│   ├── performance_comparison.png
│   ├── confusion_matrix_before.png
│   ├── confusion_matrix_after.png
│   ├── roc_curve.png
│   └── precision_recall_curve.png
│
└── screenshots/
    ├── streamlit_home.png
    ├── spam_prediction.png
    └── ham_prediction.png
```

Adjust the structure to match the actual files in the repository.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data handling |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning |
| TF-IDF | Text feature extraction |
| Multinomial Naive Bayes | Spam classification |
| Matplotlib | Visualization |
| Joblib | Model serialization |
| Streamlit | Web application |
| Git | Version control |
| GitHub | Source-code hosting |

---

## 🚀 Run the Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/chetansepta-dev/spam-detector-app.git
```

### 2. Enter the project directory

```bash
cd email_spam_detection
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the environment on Windows

```bash
venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Streamlit application

```bash
streamlit run app.py
```

The application will then be available through the local Streamlit address shown in the terminal.

---

## 🧪 Example Predictions

### Spam-like email

```text
C0NGRATULATI0NS! You have been selected to receive
$5,000. Cl1ck the l1nk to claim your reward.
```

Expected type: **Spam**

### Legitimate email

```text
Hey, I'll be reaching Chennai around 6 pm.
Can you pick me up from the station?
I'll call you when I arrive.
```

Expected type: **Ham**

Testing also included intentionally tricky messages designed to resemble legitimate business communication while containing suspicious language.

---

## ⚠️ Limitations

Although the final model achieved excellent performance on the evaluated test set, benchmark performance does not guarantee perfect real-world spam detection.

Current limitations include:

1. The model primarily relies on email text.
2. Spam patterns can change over time.
3. Obfuscated or adversarial messages may reduce performance.
4. The evaluation is based on a specific dataset and held-out test set.
5. Real-world emails may contain signals not represented in the text features.
6. The system does not independently analyze sender reputation, URLs, attachments, or email authentication headers.

---

## 🔮 Future Improvements

Potential future improvements include:

- Probability and threshold-based decision making
- Probability calibration
- Larger and more diverse datasets
- Continuous retraining and monitoring
- Concept-drift detection
- URL and domain-based features
- Email-header analysis
- Character-level features for obfuscated spam
- Ensemble classifiers
- Production prediction APIs
- User feedback loops
- Automated model versioning
- Monitoring false positives and false negatives

---

## 🎓 Learning Outcomes

This project provided practical experience with:

- Natural Language Processing
- Text preprocessing
- TF-IDF feature engineering
- Supervised machine learning
- Multinomial Naive Bayes
- Hyperparameter tuning
- Cross-validation
- Model evaluation
- Confusion matrices
- ROC-AUC analysis
- Precision-Recall analysis
- Model serialization
- Streamlit development
- Git/GitHub
- Online deployment
- End-to-end machine learning workflows

---

## 📌 Project Status

**Version:** `1.0`

**Status:** Completed

The current version includes dataset preparation, TF-IDF feature extraction, Multinomial Naive Bayes classification, hyperparameter tuning, cross-validation, evaluation, model serialization, a Streamlit interface, and online deployment.

---

## 👨‍💻 Author

**Your Name**  
Computer Science and Engineering Student

GitHub: <https://github.com/chetansepta-dev>
---

## 🙏 Acknowledgements

This project was developed for educational purposes to explore practical machine learning and natural language processing techniques for email spam classification.

The project uses open-source Python libraries and a publicly available email spam corpus.

---

## ⭐ If You Find This Project Useful

If this project helped you understand practical machine learning, NLP, or spam classification, consider giving the repository a ⭐ on GitHub.
