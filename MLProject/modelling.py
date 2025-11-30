import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import warnings
import sys

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(42)

    # Ambil parameter dari command line
    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 505
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 37
    train_path = sys.argv[3] if len(sys.argv) > 3 else "./dataset/train_dataset.csv"
    test_path  = sys.argv[4] if len(sys.argv) > 4 else "./dataset/test_dataset.csv"

    # Load dataset
    train_data = pd.read_csv(train_path)
    test_data  = pd.read_csv(test_path)

    X_train = train_data.drop(columns=["Churn"])
    y_train = train_data["Churn"]
    X_test  = test_data.drop(columns=["Churn"])
    y_test  = test_data["Churn"]

    input_example = X_train.head(5)

    # MLflow run
    with mlflow.start_run():
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            class_weight="balanced"  # opsional
        )
        model.fit(X_train, y_train)

        # Log model dan metrics
        mlflow.sklearn.log_model(sk_model=model, artifact_path="model", input_example=input_example)
        accuracy = model.score(X_test, y_test)
        mlflow.log_metric("accuracy", accuracy)

    print("Training selesai ✔")
    print("Akurasi Test:", accuracy)
