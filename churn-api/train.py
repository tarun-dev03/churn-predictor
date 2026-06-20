import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("data.csv")

# STEP 1: Handle missing values
df = df.fillna(0)

# STEP 2: Automatically detect target column
target = df.columns[-1]   # last column as target

# Split
X = df.drop(target, axis=1)
y = df[target]

# STEP 3: Encode target if categorical
if y.dtype == "object":
    le = LabelEncoder()
    y = le.fit_transform(y)

# STEP 4: Detect column types
categorical_cols = X.select_dtypes(include="object").columns.tolist()
numerical_cols = X.select_dtypes(exclude="object").columns.tolist()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# STEP 5: Preprocessing pipeline
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ("num", StandardScaler(), numerical_cols)
])

# STEP 6: Multiple models (Auto selection)
models = {
    "LogisticRegression": LogisticRegression(max_iter=5000),
    "RandomForest": RandomForestClassifier()
}

best_model = None
best_score = 0
best_name = ""

for name, m in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", m)
    ])

    pipeline.fit(X_train, y_train)
    score = pipeline.score(X_test, y_test)

    print(f"{name} Accuracy:", score)

    if score > best_score:
        best_score = score
        best_model = pipeline
        best_name = name

print("\nBest Model:", best_name)
print("Best Accuracy:", best_score)

# STEP 7: Save EVERYTHING
model_data = {
    "pipeline": best_model,
    "columns": X.columns,
    "categorical_cols": categorical_cols,
    "numerical_cols": numerical_cols,
    "target": target,
    "model_name": best_name
}

pickle.dump(model_data, open("model.pkl", "wb"))
