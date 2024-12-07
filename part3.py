import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from part2 import preprocess_data
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def split_vectorize_data(df):
    X = df['cleaned_text']
    y = df['airline_sentiment']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=3)

    # Print the distribution of the training and test set
    # print("Training set distribution:\n", y_train.value_counts(),
    #       y_train.value_counts(normalize=True) * 100)
    # print("\nTest set distribution:\n", y_test.value_counts(),
    #       y_test.value_counts(normalize=True) * 100)

    # Vectorize
    vectorizer = TfidfVectorizer(
        analyzer='word', ngram_range=(1, 1), max_features=99999)
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    return X_train, X_test, y_train, y_test


def analyze_ten_fold(model, X_train, y_train, order):
    # Perform cross validation

    cv_scores = cross_val_score(
        model, X_train, y_train, cv=10, scoring='accuracy')
    # print("\n10-Fold Cross-Validation Scores:", cv_scores)
    # print("\nMean Validation Accuracy: {:.6f}".format(cv_scores.mean()))
    # Save cross-validation results to CSV
    cv_results = pd.DataFrame(
        {'Fold': range(1, 11), 'Accuracy': cv_scores, 'Order': ",".join(order)})
    cv_results.to_csv('cv_results.csv', mode='a', header=False, index=False)


def evaluate_metric(model, y_test, y_pred, order):
    # Save classification metrics
    accuracy = accuracy_score(y_test, y_pred)
    print(f"accuracy: {accuracy:.4f}")
    report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Save metrics to CSV
    report_df = pd.DataFrame(report).transpose()
    f1_score = report_df["f1-score"].iloc[4]
    report_df.to_csv(
        f'classification_report/classification_report_{"".join(order)}.csv', index=True)
    # print("\nClassification Report saved to 'classification_report.csv'.")

    # Save confusion matrix
    conf_matrix_df = pd.DataFrame(conf_matrix)
    conf_matrix_df.to_csv(
        f'confusion_csv/confusion_matrix_{"".join(order)}.csv', index=False)
    # print("\nConfusion Matrix saved to 'confusion_matrix.csv'.")

    # Plot and save confusion matrix
    disp = ConfusionMatrixDisplay(
        confusion_matrix=conf_matrix, display_labels=model.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.savefig(f'confusion_matrix/confusion_matrix_{"".join(order)}.png')
    return f1_score


def train(df, order):

    # Split Data
    X_train, X_test, y_train, y_test = split_vectorize_data(df)

    # Initialized model
    model = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-4,
                          max_iter=100, tol=None, shuffle=True, random_state=3, n_jobs=-1)

    analyze_ten_fold(model=model, X_train=X_train,
                     y_train=y_train, order=order)

    # Train and Evaluate on the Test Set
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    f1_score = evaluate_metric(
        model=model, y_test=y_test, y_pred=y_pred, order=order)
    return f1_score


def main():

    possible_orders = [
        # ["1", "8", "4", "9", "3", "2", "7", "5", "6", "10", "11", "12"],
        # ["1", "8", "4", "3", "9", "2", "7", "10", "6", "5", "11", "12"],
        # ["7", "6", "5", "8", "1", "4", "3", "9", "2", "10", "11", "12"],
        # ["9", "6", "1", "8", "4", "3", "2", "7", "5", "10", "11", "12"],
        # ["10", "11", "1", "8", "4", "3", "9", "2", "7", "6", "5", "12"],
        # ["8", "1", "3", "9", "7", "6", "5", "10", "11", "4", "12", "2"],
        ["1", "8", "4", "9", "10", "11", "6", "7", "5", "3", "2", "12"],
        # ["1", "6", "8", "7", "4", "5", "3", "10", "11", "2", "9", "12"],
        # ["6", "7", "5", "10", "11", "12", "4", "2", "3", "9", "1", "8"],
        # ["1", "8", "4", "3", "2", "9", "10", "7", "6", "5", "11", "12"]
        ["0"]
    ]

    highest_f1score = 0
    for order in possible_orders:
        print(f"Processing with order: {order}")
        # Apply text cleaning
        df = preprocess_data(order)
        f1_score = train(df, order)
        if f1_score > highest_f1score:
            highest_f1score = f1_score
            print("new_highest_f1_score: ", highest_f1score)


if __name__ == "__main__":
    main()
