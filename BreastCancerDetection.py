import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
)
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier


def load_dataset():
    # Load the dataset from CSV
    dataset = pd.read_csv(r'E:\Machine Learning Projects\Breast cancer detection\data.csv')
    return dataset


def explore_dataset(dataset):
    # Display dataset information
    print(dataset.head())
    print(dataset.shape)
    print(dataset.info())
    print(dataset.select_dtypes(include='object').columns)
    print(len(dataset.select_dtypes(include='object').columns))
    print(dataset.select_dtypes(include=['float64', 'int64']).columns)
    print(len(dataset.select_dtypes(include=['float64', 'int64']).columns))

    # Statistical summary
    print(dataset.describe())
    print(dataset.columns)

    # Check for null values
    print(dataset.isnull().values.any())
    print(dataset.isnull().values.sum())
    print(dataset.columns[dataset.isnull().any()])
    print(len(dataset.columns[dataset.isnull().any()]))
    print(dataset['Unnamed: 32'].count())
    dataset = dataset.drop(columns='Unnamed: 32')
    print(dataset.shape)
    print(dataset.isnull().values.any())

    # Dealing with categorical data
    print(dataset.select_dtypes(include='object').columns)
    print(dataset['diagnosis'].unique())
    print(dataset['diagnosis'].nunique())

    # Convert categorical variables to numerical using one-hot encoding
    dataset = pd.get_dummies(data=dataset, drop_first=True)
    return dataset


def visualize_data(dataset):
    # Count plot
    sns.countplot(data=dataset, x='diagnosis_M', label='Count')
    plt.show()
    print(dataset['diagnosis_M'].value_counts())
    print(dataset['diagnosis_M'].dtype)
    print(dataset['diagnosis_M'].unique())

    # Another way to count the 0 and 1
    print((dataset.diagnosis_M == 0).sum())
    print((dataset.diagnosis_M == 1).sum())

    # Correlation matrix and heatmap
    dataset_2 = dataset.drop(columns='diagnosis_M')
    print(dataset_2.head())
    dataset_2.corrwith(dataset['diagnosis_M']).plot.bar(
        figsize=(20, 10), title='Correlated with diagnosis_M', rot=45, grid=True
    )
    plt.show()

    # Correlation matrix
    corr = dataset.corr()
    print(corr)

    # Heatmap to analyze data
    plt.figure(figsize=(20, 10))
    sns.heatmap(corr, annot=True)
    plt.show()


def split_dataset(dataset):
    # Split the dataset into train and test sets
    x = dataset.iloc[:, 1:-1].values
    y = dataset.iloc[:, -1].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    return x_train, x_test, y_train, y_test


def scale_features(x_train, x_test):
    # Feature scaling
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)
    return x_train, x_test


def train_logistic_regression(x_train, y_train):
    # Build the logistic regression model
    classifier_lr = LogisticRegression(random_state=0)
    classifier_lr.fit(x_train, y_train)
    return classifier_lr


def train_random_forest(x_train, y_train):
    # Build the random forest model
    classifier_rf = RandomForestClassifier(random_state=0)
    classifier_rf.fit(x_train, y_train)
    return classifier_rf


def evaluate_model(model, x_test, y_test):
    # Make predictions
    y_pred = model.predict(x_test)

    # Analyze model performance
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)

    results = pd.DataFrame([['Model', acc, f1, prec, rec]],
                           columns=['Model', 'Accuracy', 'F1 score', 'Precision', 'Recall'])
    print(results)

    cm = confusion_matrix(y_test, y_pred)
    print(cm)


def cross_validate_model(model, x_train, y_train):
    # Cross validation
    accuracies = cross_val_score(estimator=model, X=x_train, y=y_train, cv=10)
    print('Accuracy of model is {:.2f} %'.format(accuracies.mean() * 100))
    print('Standard deviation is {:.2f} %'.format(accuracies.std() * 100))


def main():
    # Load dataset
    dataset = load_dataset()

    # Explore dataset
    dataset = explore_dataset(dataset)

    # Visualize data
    visualize_data(dataset)

    # Split dataset
    x_train, x_test, y_train, y_test = split_dataset(dataset)

    # Scale features
    x_train, x_test = scale_features(x_train, x_test)

    # Train logistic regression model
    classifier_lr = train_logistic_regression(x_train, y_train)
    evaluate_model(classifier_lr, x_test, y_test)
    cross_validate_model(classifier_lr, x_train, y_train)

    # Train random forest model
    classifier_rf = train_random_forest(x_train, y_train)
    evaluate_model(classifier_rf, x_test, y_test)
    cross_validate_model(classifier_rf, x_train, y_train)


if __name__ == "__main__":
    main()
