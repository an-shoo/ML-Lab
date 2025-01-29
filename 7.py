from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

# Dataset
data = [
    ("I love this sandwich", "pos"),
    ("This is an amazing place", "pos"),
    ("I feel very good about these cheese", "pos"),
    ("This is my best work", "pos"),
    ("What an awesome view", "pos"),
    ("I do not like this restaurant", "neg"),
    ("I am tired of this stuff", "neg"),
    ("I can’t deal with this", "neg"),
    ("He is my sworn enemy", "neg"),
    ("My boss is horrible", "neg")
]

# Split dataset into training and testing sets
texts, labels = zip(*data)
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.3, random_state=42)

# Convert text to feature vectors
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train the Naïve Bayes model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Predict and evaluate
y_pred = model.predict(X_test_vec)

# Performance metrics
print("Total Instances:", len(labels))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, pos_label="pos"))
print("Recall:", recall_score(y_test, y_pred, pos_label="pos"))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Test new predictions
new_texts = ["I love this view", "I hate this place", "This is amazing"]
new_texts_vec = vectorizer.transform(new_texts)
predictions = model.predict(new_texts_vec)

for text, pred in zip(new_texts, predictions):
    print(f"Prediction for '{text}': {pred}")
