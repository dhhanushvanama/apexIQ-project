import os
import sys
from dataclasses import dataclass

import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='xgboost')
warnings.filterwarnings("ignore", category=UserWarning, module='LightGBM')

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix, classification_report
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from src.utils import evaluate_models
from imblearn.over_sampling import SMOTE

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class ModelTrainingConfig:
    trained_model_file_path = os.path.join("artifacts", "startup_model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainingConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)

            logging.info(f"After SMOTE, training label distribution: {dict(pd.Series(y_train).value_counts())}")

            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Random Forest": RandomForestClassifier(n_jobs=1),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "XGBoost": XGBClassifier(eval_metric='mlogloss'),
                "LightGBM": LGBMClassifier(),
                "CatBoost": CatBoostClassifier(learning_rate=0.05, iterations=200)
            }

            params = {
                "Logistic Regression": {},
                "Random Forest": {
                    "n_estimators": [50],
                    "max_depth": [10]
                },
                "Decision Tree": {
                    "criterion": ["gini"]
                },
                "Gradient Boosting": {
                    "learning_rate": [0.1],
                    "n_estimators": [50]
                },
                "XGBoost": {
                    "learning_rate": [0.01, 0.1],
                    "n_estimators": [50, 100]
                },
                "LightGBM": {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.01, 0.1],
                    "num_leaves": [31, 50]
                },
                "CatBoost": {
                    "learning_rate": [0.05],
                    "iterations": [200]
                }
            }

            model_report = evaluate_models(
                X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test,
                models=models, param=params,
            )

            # To get best model score from dict
            best_model_score = max(model_report.values())

            # To get best model name from dict
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            best_model.fit(X_train, y_train)

            # Evaluate additional metrics before saving
            y_pred = best_model.predict(X_test)

            f1 = f1_score(y_test, y_pred, average='weighted')
            try:
                auc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])
            except Exception as e:
                logging.warning(f"AUC score could not be calculated: {e}")
                auc = 0.0

            if best_model_score < 0.7 or f1 < 0.65 or auc < 0.7:
                raise CustomException(f"No suitable model found. Conditions not met - Accuracy: {best_model_score:.2f}, F1 Score: {f1:.2f}, AUC: {auc:.2f}")

            logging.info(f"Best model found: {best_model_name}")
            logging.info(f"Accuracy: {best_model_score:.2f}, F1 Score: {f1:.2f}, AUC: {auc:.2f}")

            # Save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            y_train_pred = best_model.predict(X_train)
            train_accuracy = accuracy_score(y_train, y_train_pred)
            logging.info(f"Training Accuracy: {train_accuracy:.4f}")

            y_pred = best_model.predict(X_test)
            final_accuracy = accuracy_score(y_test, y_pred)

            return final_accuracy

        except Exception as e:
            raise CustomException(e, sys)
