from data_preprosessing import *
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV
import itertools

CVDs_dataset_raw = load_data(r'prognosi-care\datasets\framingham.csv')

reg_imputer = IterativeImputer()
CVDs_dataset = reg_imputer.fit_transform(CVDs_dataset_raw)
CVDs_dataset = pd.DataFrame(CVDs_dataset, columns=CVDs_dataset_raw.columns)

X_categorical = CVDs_dataset.iloc[:, :-1]
Y_categorical = CVDs_dataset.iloc[:, -1]
categorical_columns = X_categorical.select_dtypes(include=['object']).columns
X = one_hot_code_dataset(X_categorical, categorical_columns)
Y = Y_categorical

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)
X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

model = DecisionTreeClassifier(criterion='entropy', random_state=0)
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

accuracy = accuracy_score(Y_test, Y_pred)
print("-----before hyperparameter tuning-----")
print("Accuracy:", accuracy)
print(classification_report(Y_test, Y_pred))
print(confusion_matrix(Y_test, Y_pred))
print("ROC AUC Score:", roc_auc_score(Y_test, model.predict_proba(X_test)[:, 1]))

dt_hyperparams = {
    
    'max_depth' : 4
}

improved_model = DecisionTreeClassifier(criterion='entropy', random_state=0, **dt_hyperparams)
improved_model.fit(X_train, Y_train)
Y_pred_improved = improved_model.predict(X_test)
accuracy_improved = accuracy_score(Y_test, Y_pred_improved)
print("-----after hyperparameter tuning-----")
print("Accuracy:", accuracy_improved)
print(classification_report(Y_test, Y_pred_improved))
print(confusion_matrix(Y_test, Y_pred_improved))
print("ROC AUC Score:", roc_auc_score(Y_test, improved_model.predict_proba(X_test)[:, 1]))

rf_model = RandomForestClassifier(n_estimators=100, random_state=0)
rf_model.fit(X_train, Y_train)
Y_pred_rf = rf_model.predict(X_test)
accuracy_rf = accuracy_score(Y_test, Y_pred_rf)
print("-----Random Forest Model-----")
print("Accuracy:", accuracy_rf)
print(classification_report(Y_test, Y_pred_rf))
print(confusion_matrix(Y_test, Y_pred_rf))
print("ROC AUC Score:", roc_auc_score(Y_test, rf_model.predict_proba(X_test)[:, 1]))

hyperparams = {
        'n_estimators': [150, 170, 200],
        'max_depth': [5, 6, 7],
        'min_samples_leaf': [3, 4],
        'min_samples_split': [2, 3, 4],
        'max_features': ['sqrt', 'log2'],
        'bootstrap': [True, False]
    }

fixed_hyperparams = {
        'random_state': 10,
    }

rf = RandomForestClassifier

grid_search = GridSearchCV(estimator=rf(**fixed_hyperparams), param_grid=hyperparams, scoring='roc_auc', cv=5, n_jobs=-1)
grid_search.fit(X_train, Y_train)
best_rf = grid_search.best_estimator_
best_hyperparams = grid_search.best_params_
print(f"Best hyperparameters:\n{best_hyperparams}")
print(f"Accuracy on test set: {best_rf.score(X_test, Y_test)}")

# Sample data point for testing the model's metrics
sample_data_point = {
    'age': 50,
    'sex': 1,  # 1 for male, 0 for female
    'education': 2,  # Example education level
    'currentSmoker': 0,
    'cigsPerDay': 0,
    'BPMeds': 0,
    'prevalentStroke': 0,
    'prevalentHyp': 1,
    'diabetes': 0,
    'totChol': 250,
    'sysBP': 140,
    'diaBP': 90,
    'BMI': 28.5,
    'heartRate': 85,
    'glucose': 100
}

# Convert the sample data point to a DataFrame
sample_data_df = pd.DataFrame([sample_data_point])

# Preprocess the sample data point to match training data preprocessing
sample_data_df_encoded = one_hot_code_dataset(sample_data_df, categorical_columns)

# Align the sample data columns with the training data columns
sample_data_df_encoded = sample_data_df_encoded.reindex(columns=X_train.columns, fill_value=0)

# Scale the sample data using the same scaler used for training
sample_data_scaled = sc.transform(sample_data_df_encoded)

# Predict using the improved model
sample_prediction = best_rf.predict(sample_data_scaled)

# Predict probabilities using the improved model
sample_prediction_prob = best_rf.predict_proba(sample_data_scaled)

# Print the chance of developing cardiovascular disease within 10 years, rounded to the nearest tenth
print(f"Chance of developing cardiovascular disease within 10 years: {round(sample_prediction_prob[0][1] * 100, 1)}%\n")