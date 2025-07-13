# === emg_gender_model_with_cv_and_test_plot.py ===

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN

# === Load Dataset ===
data = pd.read_csv('emgchunksfeaturesdata.csv')

# === Encode Categorical Columns ===
label_encoders = {}
for col in data.columns:
    if data[col].dtype == 'object':
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

# === Define Features and Target for Gender ===
X = data.drop(columns=['Gender']).values
y = data['Gender'].values

# === Safe Resampling ===
resampling_methods = {"Original": (X, y)}
resamplers = {
    "Oversampling": RandomOverSampler(random_state=42),
    "Undersampling": RandomUnderSampler(random_state=42),
    "SMOTE": SMOTE(random_state=42),
    "ADASYN": ADASYN(random_state=42),
    "SMOTE ENN": SMOTEENN(random_state=42)
}

for name, sampler in resamplers.items():
    try:
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        resampling_methods[name] = (X_resampled, y_resampled)
    except ValueError as e:
        print(f"[Warning] Skipping {name}: {e}")

# === Classifiers ===
models = {
    "KNN": KNeighborsClassifier(),
    "Logistic Regression": LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(class_weight='balanced', random_state=42),
    "Random Forest": RandomForestClassifier(class_weight='balanced', random_state=42),
    "SVM": SVC(kernel='linear', class_weight='balanced', random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "MLP": MLPClassifier(random_state=42, max_iter=1000),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "Bagging": BaggingClassifier(random_state=42)
}

# === Collect Results ===
results = {}

for method_name, (X_res, y_res) in resampling_methods.items():
    print(f"\n{'='*50}\nResampling Method: {method_name}\n{'='*50}")
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.1, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    method_results = {}

    for model_name, model in models.items():
        print(f"\n--- Model: {model_name} ---")
        cv_score = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        mean_cv_acc = cv_score.mean()
        std_acc = cv_score.std()
        print(f"CV Accuracy: {mean_cv_acc:.4f} ± {std_acc:.4f}")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred)
        print(f"Test Accuracy: {test_acc:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        method_results[model_name] = {
            "cv": mean_cv_acc,
            "test": test_acc
        }

    results[method_name] = method_results

# === Plotting CV vs Test Accuracy ===
for method_name, scores in results.items():
    model_names = list(scores.keys())
    cv_accuracies = [scores[model]["cv"] for model in model_names]
    test_accuracies = [scores[model]["test"] for model in model_names]

    x = np.arange(len(model_names))
    width = 0.35  # bar width

    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, cv_accuracies, width, label='CV Accuracy', color='cornflowerblue')
    plt.bar(x + width/2, test_accuracies, width, label='Test Accuracy', color='lightcoral')

    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.title(f'CV vs Test Accuracy - {method_name} Resampling')
    plt.xticks(x, model_names, rotation=45)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
