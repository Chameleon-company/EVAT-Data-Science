import regression

#####CLASSIFICATION MODEL###############ETA PREDICTION###############
# Create classification target: "Short ETA" if ETA <= 5 min
df['ETA_Class'] = df['ETA_min'].apply(lambda x: "Short" if x <= 5 else "Long")

features_clf = features_reg
X_clf = df[features_clf]
y_clf = df['ETA_Class']

X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)

# Encode classification labels
le_eta = LabelEncoder()
y_train_clf_enc = le_eta.fit_transform(y_train_clf)
y_test_clf_enc = le_eta.transform(y_test_clf)

# Model 1: Logistic Regression
log_clf = LogisticRegression(max_iter=1000)
log_clf.fit(X_train_clf, y_train_clf_enc)
y_pred_log = log_clf.predict(X_test_clf)

# Model 2: Random Forest Classifier
rf_clf = RandomForestClassifier(random_state=42)
rf_clf.fit(X_train_clf, y_train_clf_enc)
y_pred_rf_clf = rf_clf.predict(X_test_clf)

# Model 3: XGBoost Classifier
xgb_clf = XGBClassifier(random_state=42)
xgb_clf.fit(X_train_clf, y_train_clf_enc)
y_pred_xgb_clf = xgb_clf.predict(X_test_clf)

# Evaluation (Classification)
def classification_results(y_true, y_pred, model_name):
    print(f"\n{model_name} Classification Results:")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print(classification_report(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


classification_results(y_test_clf_enc, y_pred_log, "Logistic")
classification_results(y_test_clf_enc, y_pred_rf_clf, "Random Forest")
classification_results(y_test_clf_enc, y_pred_xgb_clf, "XGBoost")

#####CLASSIFICATION MODEL###############STATION PREDICTION###############

# Target: Station_Name (assuming the target variable is the Station_Name column)
y_clf_station = df['Station_Name']  # Station_Name is the target for multi-class classification

# Features remain the same (coordinates, ETA, etc.)
X_clf_station = df[features_reg]

# Train-Test Split
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X_clf_station, y_clf_station, test_size=0.2, random_state=42)

# Encode station names using LabelEncoder (since it’s a multi-class classification task)
le_station = LabelEncoder()
y_train_clf_enc = le_station.fit_transform(y_train_clf)
y_test_clf_enc = le_station.transform(y_test_clf)

# Model 1: Logistic Regression
log_clf_station = LogisticRegression(max_iter=1000, multi_class='ovr', solver='lbfgs')  # Multi-class handling
log_clf_station.fit(X_train_clf, y_train_clf_enc)
y_pred_log_station = log_clf_station.predict(X_test_clf)

# Model 2: Random Forest Classifier
rf_clf_station = RandomForestClassifier(random_state=42)
rf_clf_station.fit(X_train_clf, y_train_clf_enc)
y_pred_rf_clf_station = rf_clf_station.predict(X_test_clf)

# Model 3: XGBoost Classifier
xgb_clf_station = XGBClassifier(random_state=42)
xgb_clf_station.fit(X_train_clf, y_train_clf_enc)
y_pred_xgb_clf_station = xgb_clf_station.predict(X_test_clf)

# Evaluation (Multi-class Classification)
def classification_results_multi_class(y_true, y_pred, model_name):
    print(f"\n{model_name} Classification Results (Station Prediction):")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    
    # Get unique labels from y_true and y_pred (it should be consistent)
    unique_labels = np.unique(np.concatenate([y_true, y_pred]))

    # Adjusting classification report with correct label classes
    print(classification_report(y_true, y_pred, labels=unique_labels, target_names=le_station.classes_[:len(unique_labels)]))  # Matching the number of classes
    
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=le_station.classes_, yticklabels=le_station.classes_)
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()



# Evaluation for each model
classification_results_multi_class(y_test_clf_enc, y_pred_log_station, "Logistic Regression")
classification_results_multi_class(y_test_clf_enc, y_pred_rf_clf_station, "Random Forest")
classification_results_multi_class(y_test_clf_enc, y_pred_xgb_clf_station, "XGBoost")
