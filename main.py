from data.data_preprocessing import load_clear_data, save_processed
from src.classification import (
    load_processed, label_and_onehot_processing, split_data,
    train_and_evaluate, train_random_forest,
    cross_validate_decision_tree, cross_validate_random_forest,
    extract_decision_rules, check_label_distribution,
    plot_feature_importance, predict_and_export
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def main():
    # Load dan simpan data hasil pembersihan
    df = load_clear_data('datasets/business_owner_dataset.csv')
    save_processed(df)

    # Load data hasil bersih
    df_processed = load_processed('datasets/processed_data.csv')

    # Preprocessing encoding
    X_encoded, y, feature_names, real_label = label_and_onehot_processing(df_processed)

    # Simpan versi final untuk modelling
    processed_df = X_encoded.copy()
    processed_df['Willingness to Develop'] = y
    processed_df.to_csv('datasets/processed_model_data.csv', index=False)

    # Split data
    X_train, X_test, y_train, y_test = split_data(X_encoded, y)

    # Cek distribusi label
    check_label_distribution(df_processed, 'Willingness to Develop')

    # Plot feature importance
    print("\nThe Most Important Features")
    plot_feature_importance(processed_df)

    # Training Decision Tree
    print("\nDecision Tree Model")
    train_and_evaluate(X_train, X_test, y_train, y_test, max_depth=6, min_samples_split=20, min_samples_leaf=5)

    # Training Random Forest
    print("\nRandom Forest Model")
    rf_model = RandomForestClassifier(
        n_estimators=150, max_depth=10,
        min_samples_split=10, min_samples_leaf=5,
        random_state=42
    )
    rf_model.fit(X_train, y_train)

    # Cross-validation
    print("\nCross Validation Decision Tree")
    cross_validate_decision_tree(X_encoded, y, max_depth=6, min_samples_split=20, min_samples_leaf=5, cv=5)

    print("\nCross Validation Random Forest")
    cross_validate_random_forest(X_encoded, y, n_estimators=150, max_depth=10, min_samples_split=10, min_samples_leaf=5, cv=5)

    # Ekstrak aturan Decision Tree
    print("\nExtract Decision Rules")
    clf = DecisionTreeClassifier(criterion='entropy', max_depth=6, min_samples_split=20, min_samples_leaf=5, random_state=42)
    clf.fit(X_encoded, y)
    extract_decision_rules(clf, feature_names, ['Tidak Bersedia', 'Bersedia'])

    # Simpan output JSON hasil prediksi random forest
    predict_and_export(
        model=rf_model,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_names,
        label_map=real_label,
        model_name="random_forest"
    )

if __name__ == "__main__":
    main()
