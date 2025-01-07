from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
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

# Prepare the data
texts, labels = zip(*data)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# Train the model
model = MultinomialNB()
model.fit(X, labels)

# Predict and evaluate
y_pred = model.predict(X)
print("Total Instances:", len(labels))
print("Accuracy:", accuracy_score(labels, y_pred))
print("Precision:", precision_score(labels, y_pred, pos_label="pos"))
print("Recall:", recall_score(labels, y_pred, pos_label="pos"))
print("Confusion Matrix:\n", confusion_matrix(labels, y_pred))
