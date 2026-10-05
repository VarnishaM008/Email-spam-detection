import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix

nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

data = pd.read_csv(
    'SMSSpamCollection',
    sep='\t',
    names=['label', 'message']
)

data['label'] = data['label'].map({'ham': 0, 'spam': 1})
print("Dataset Loaded Successfully")
print(data.head())

stop_words = set(stopwords.words('english'))

def clean_text(text):
    words = text.lower().split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

data['message'] = data['message'].apply(clean_text)

X_train, X_test, y_train, y_test = train_test_split(
    data['message'],
    data['label'],
    test_size=0.2,
    random_state=42
)

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": SVC()
}

results = {}

for name, model in models.items():

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    accuracy = accuracy_score(y_test, prediction)

    results[name] = accuracy

    print("\n", name)
    print("Accuracy:", accuracy)
    print(classification_report(
        y_test,
        prediction,
        target_names=['Not Spam', 'Spam']
    ))

print("\nModel Accuracy Comparison")

for model, accuracy in results.items():
    print(model, ":", round(accuracy * 100, 2), "%")

plt.figure(figsize=(7, 4))
sns.barplot(
    x=list(results.keys()),
    y=list(results.values())
)

plt.title("Spam Detection Model Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Algorithm")
plt.ylim(0, 1)
plt.show()

message = input("\nEnter an email/SMS message: ")

message_clean = clean_text(message)
message_vector = vectorizer.transform([message_clean])

print("\nPrediction Results:")

for name, model in models.items():

    prediction = model.predict(message_vector)

    if prediction[0] == 1:
        result = "SPAM"
    else:
        result = "NOT SPAM"

    print(name, ":", result)