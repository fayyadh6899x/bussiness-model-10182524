from data.data_preprocessing import load_clear_data, save_processed
from src.classification import load_processed, label_processing, split_data, train_and_evaluate, save_model_processed, check_label_distribution, train_random_forest

def main():
    df = load_clear_data('datasets/business_owner_dataset_natural.csv')
    save_processed(df)

    df_processed = load_processed('datasets/processed_data.csv')

    df_encoded = label_processing(df_processed) 
    save_model_processed(df_encoded) 

    X_train,X_test, y_train, y_test = split_data(df_encoded, 'Willingness to Develop')

    check_label_distribution(df_encoded, 'Willingness to Develop')

    train_and_evaluate(X_train, X_test, y_train, y_test, max_depth=6, min_samples_split=20, min_samples_leaf=5)
    train_and_evaluate(X_train, X_test, y_train, y_test, max_depth=7, min_samples_split=20, min_samples_leaf=10)
    train_and_evaluate(X_train, X_test, y_train, y_test, max_depth=8, min_samples_split=30, min_samples_leaf=10)

    train_random_forest(X_train, X_test, y_train, y_test, n_estimators=200, max_depth=15, min_samples_split=5, min_samples_leaf=3)
    train_random_forest(X_train, X_test, y_train, y_test, n_estimators=150, max_depth=10, min_samples_split=10, min_samples_leaf=5)


if __name__ == "__main__":
    main()
