import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
import joblib

DATA_DIR = "data"
RESULTS_DIR = "results"


def load_data():
    train_path = os.path.join(DATA_DIR, "topic_train.csv")
    test_path = os.path.join(DATA_DIR, "topic_test.csv")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"{train_path} not found")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"{test_path} not found")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    train_df, test_df = load_data()

    X_train = train_df["text"].astype(str)
    y_train = train_df["label"].astype(str)
    X_test = test_df["text"].astype(str)
    y_test = test_df["label"].astype(str)

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_df=0.9,
            min_df=1
        )),
        ("clf", LogisticRegression(max_iter=300))
    ])

    # ==== Пытаемся построить learning curve ====
    # если данных мало — просто пропускаем
    class_counts = y_train.value_counts()
    min_class_count = class_counts.min()

    plotted = False
    if min_class_count >= 2 and len(X_train) >= 3:
        n_splits = max(2, min(5, int(min_class_count)))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        train_sizes, train_scores, test_scores = learning_curve(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring="accuracy",
            train_sizes=[1.0] if len(X_train) < 20 else [0.1, 0.3, 0.5, 0.7, 1.0],
            n_jobs=-1
        )

        train_mean = train_scores.mean(axis=1)
        test_mean = test_scores.mean(axis=1)

        plt.figure()
        plt.plot(train_sizes, train_mean, marker="o", label="Training score")
        plt.plot(train_sizes, test_mean, marker="s", label="Validation score")
        plt.xlabel("Training examples")
        plt.ylabel("Accuracy")
        plt.title("Learning Curves")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "learning_curves.png"))
        print("Learning curves saved to results/learning_curves.png")
        plotted = True
    else:
        print("Not enough data per class to plot learning curve. Skipping plot.")

    # ==== Обучаем финальную модель ====
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Test accuracy:", acc)
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    model_path = os.path.join(RESULTS_DIR, "topic_classifier.pkl")
    joblib.dump(pipeline, model_path)
    print(f"Model saved to {model_path}")

    if not plotted:
        print("NOTE: when you have a bigger dataset, re-run to generate learning_curves.png")


if __name__ == "__main__":
    main()
