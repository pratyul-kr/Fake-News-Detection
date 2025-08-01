# 📰 Fake News Detection using Machine Learning

This project applies various machine learning models to detect fake news from real news using natural language processing (NLP) and classification techniques.

# 🔍 Project Objective

To build a fake news classifier using multiple ML algorithms such as:

Multinomial Naive Bayes

Decision Tree

Passive Aggressive Classifier

Random Forest

Logistic Regression

And compare their performance using metrics like accuracy, precision, recall, and F1-score.

---

# ⚙️ Technologies Used

Python

Scikit-learn

NLTK

Pandas, NumPy

Matplotlib, Seaborn

TF-IDF Vectorization

---

# 📁 Dataset

True.csv and False.csv (combined into a labeled dataset)

Each contains news articles with title, text, subject, and date

---

# 🧠 Models Trained

Model	Accuracy	Precision	Recall	F1 Score

Naive Bayes	✅	✅	✅	✅

Decision Tree	✅	✅	✅	✅

Passive Aggressive	✅	✅	✅	✅

Random Forest	✅	✅	✅	✅

Logistic Regression	✅	✅	✅	✅

Metrics are plotted for comparison.

---

# 📈 Visualizations

Confusion Matrices for each model

Accuracy, Precision, Recall, and F1-score bar plots

Combined comparison chart using pandas + Matplotlib

---

# 🧪 Example Prediction

predict_title("Donald Trump Sends Out Embarrassing New Year")

Output: The news is likely fake.

---

# 🚀 How to Run

1. Clone the repo or download the notebook.

2. Ensure required libraries are installed:

pip install nltk scikit-learn pandas matplotlib seaborn

3. Download NLTK resources inside notebook:

import nltk
nltk.download('punkt')
nltk.download('stopwords')

4. Run the notebook cell by cell.
