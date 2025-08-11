import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ✅ New working dataset link
df = pd.read_csv("heart.csv")


print("First 5 rows of dataset:")
print(df.head())

# Visualization with Seaborn
sns.countplot(x="target", data=df)
plt.title("Heart Disease Count (0 = No Disease, 1 = Disease)")
plt.show()

# Features & Target
X = df.drop("target", axis=1)
y = df["target"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

print("\n--- Decision Tree ---")
print("Train Accuracy:", accuracy_score(y_train, dt.predict(X_train)))
print("Test Accuracy :", accuracy_score(y_test, dt.predict(X_test)))

# Visualize Decision Tree
plt.figure(figsize=(15, 8))
plot_tree(dt, feature_names=X.columns, class_names=["No Disease", "Disease"], filled=True)
plt.title("Decision Tree - Default")
plt.show()

# Decision Tree with limited depth
dt_limited = DecisionTreeClassifier(max_depth=4, random_state=42)
dt_limited.fit(X_train, y_train)

print("\n--- Decision Tree (max_depth=4) ---")
print("Train Accuracy:", accuracy_score(y_train, dt_limited.predict(X_train)))
print("Test Accuracy :", accuracy_score(y_test, dt_limited.predict(X_test)))

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

print("\n--- Random Forest ---")
print("Train Accuracy:", accuracy_score(y_train, rf.predict(X_train)))
print("Test Accuracy :", accuracy_score(y_test, rf.predict(X_test)))

# Feature Importance
importances = pd.Series(rf.feature_importances_, index=X.columns)
importances.sort_values(ascending=False).plot(kind="bar", figsize=(10, 5))
plt.title("Feature Importances - Random Forest")
plt.show()

# Cross-validation
cv_scores = cross_val_score(rf, X, y, cv=5)
print("\nRandom Forest Cross-Validation Accuracy:", cv_scores.mean())
