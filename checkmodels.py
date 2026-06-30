import joblib

model = joblib.load("models/hybrid_lr_svm.pkl")

print(
    "Expected Features:",
    model.estimators_[0].n_features_in_
)