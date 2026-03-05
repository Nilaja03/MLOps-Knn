from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

# 1. Load data (Binary classification: Malignant vs Benign)
data = load_breast_cancer()
X, y = data.data, data.target

# 2. Split and Scale (Scaling helps the solver converge faster)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 3. Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# 4. Evaluate
accuracy = model.score(X_test, y_test)
print(f"Logistic Regression Accuracy: {accuracy * 100:.2f}%")

# 5. See the actual probabilities for the first 5 test samples
probs = model.predict_proba(X_test[:5])
print(f"Probabilities (Class 0, Class 1):\n{probs}")