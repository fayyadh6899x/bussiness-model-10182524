import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


def load_processed(filepath):
    df = pd.read_csv(filepath)
    return df

def label_processing(df):
    label_encoder = LabelEncoder()
    
    for column in df.columns:
        if df[column].dtype == 'object':  
            df[column] = label_encoder.fit_transform(df[column])

    return df

def save_model_processed(df, output_path='datasets/processed_model_data.csv'):

    df.to_csv(output_path, index=False)

    print(f'Data berhasil disimpan ke {output_path}')

def split_data(df, target_column, test_size=0.3):

    X = df.drop(columns=[target_column])

    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    return X_train, X_test, y_train, y_test

def check_label_distribution(df, target_column):

    print("Distribusi label : ")

    print(df[target_column]. value_counts(normalize=True) * 100)

def train_and_evaluate(X_train, X_test, y_train, y_test, max_depth=None, min_samples_split=2, min_samples_leaf=1):
    clf = DecisionTreeClassifier(criterion='entropy',
                                  max_depth=max_depth,
                                  min_samples_split=min_samples_split,
                                  min_samples_leaf=min_samples_leaf,
                                  random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Akurasi: {accuracy * 100:.2f}% | max_depth={max_depth} | min_samples_split={min_samples_split} | min_samples_leaf={min_samples_leaf}")
    return accuracy

def train_random_forest(X_train, X_test, y_train, y_test, n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1):
    rf = RandomForestClassifier(n_estimators=n_estimators,
                                max_depth=max_depth,
                                min_samples_split=min_samples_split,
                                min_samples_leaf=min_samples_leaf,
                                random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Akurasi Random Forest: {accuracy * 100:.2f}% | n_estimators={n_estimators} | max_depth={max_depth} | min_samples_split={min_samples_split} | min_samples_leaf={min_samples_leaf}")
    return accuracy

