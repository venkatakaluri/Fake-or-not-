# Fake-or-not-

pip install transformers
pip install pandas scikit-learn

# Importing required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Loading the dataset
# Replace with the path to your CSV file
df = pd.read_csv('fake_news_data.csv')

# Preprocessing the text
# Converting text to lowercase
df['text'] = df['text'].apply(lambda x: x.lower())

# Splitting dataset into features and labels
X = df['text']  # Features (news text)
y = df['label']  # Labels (0 = real, 1 = fake)

# Splitting dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Applying TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)  # Using 5000 most important words
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Training the Logistic Regression Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# Making predictions
y_pred = model.predict(X_test_tfidf)

# Evaluation Metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
