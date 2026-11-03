import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


def load_and_explore_data():
    iris = load_iris()
    data = pd.DataFrame(iris.data, columns=iris.feature_names)
    data["target"] = iris.target
    data["species"] = data["target"].map({
        0: "setosa",
        1: "versicolor",
        2: "virginica",
    })

    print("Dataset shape:", data.shape)
    print("\nFirst 5 rows:")
    print(data.head())
    print("\nSummary statistics:")
    print(data.describe())
    print("\nClass distribution:")
    print(data["species"].value_counts())

    return data, iris


def main():
    data, iris = load_and_explore_data()

    X = data[list(iris.feature_names)]
    y = data["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print("\nTraining and testing split completed.")
    print(f"Model accuracy: {accuracy:.2%}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, target_names=iris.target_names))

    sample = X_test.iloc[[0]]
    predicted_class = model.predict(sample)[0]
    print(f"\nExample prediction for the first test sample: {iris.target_names[predicted_class]}")


if __name__ == "__main__":
    main()
