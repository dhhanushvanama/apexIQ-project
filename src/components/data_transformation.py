import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import LabelEncoder

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def add_manual_features(self, df):
        # Adding features that might be useful, like business description length, etc.
        df["description_length"] = df["business_description"].apply(len)
        df["funding_rounds_count"] = df["funding_rounds"].apply(lambda x: len(str(x).split(',')))
        df["employees_count"] = df["employees"].apply(lambda x: len(str(x).split(',')))
        return df

    def get_data_transformer_object(self):
        '''
        This function is responsible for the data transformation pipeline
        '''
        try:
            # Define the columns for different features
            numerical_columns = ["revenue", "assets", "liabilities", "description_length", "funding_rounds_count", "employees_count"]
            categorical_columns = ["industry", "location", "founder"]
            text_column = "business_description"

            # Pipeline for numerical features
            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            # Pipeline for categorical features
            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder", OneHotEncoder(handle_unknown='ignore')),
                ("scaler", StandardScaler(with_mean=False))
            ])

            # Pipeline for text features
            text_pipeline = Pipeline(steps=[
                ("selector", FunctionTransformer(lambda x: x["business_description"], validate=False)),
                ("tfidf", TfidfVectorizer(max_features=3000, ngram_range=(1, 2), stop_words="english"))
            ])

            logging.info(f"Numerical columns: {numerical_columns}")
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Text column: {text_column}")

            # Combine all transformers
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", num_pipeline, numerical_columns),
                    ("cat", cat_pipeline, categorical_columns),
                    ("txt", text_pipeline, [text_column])
                ],
                remainder='drop'
            )
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Load train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            # Add manual features like business description length, funding rounds count, etc.
            train_df = self.add_manual_features(train_df)
            test_df = self.add_manual_features(test_df)

            logging.info("Manual features added")

            # Get preprocessing object (ColumnTransformer)
            preprocessing_obj = self.get_data_transformer_object()

            # Target column (Assuming 'bankruptcy' as the label indicating bankruptcy status)
            target_column_name = "bankruptcy"

            # Separate input features and target variable
            input_feature_train_df = train_df.drop(columns=[target_column_name])
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name])
            target_feature_test_df = test_df[target_column_name]

            # Label encoding for target variable (bankruptcy - 0 or 1)
            label_encoder = LabelEncoder()
            target_feature_train_df = label_encoder.fit_transform(target_feature_train_df)
            target_feature_test_df = label_encoder.transform(target_feature_test_df)

            logging.info("Label encoding applied to target variable")

            # Apply the preprocessing steps (scaling, one-hot encoding, etc.) on the input data
            logging.info("Applying preprocessing object on training and testing data")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Convert sparse matrix to dense if needed
            if hasattr(input_feature_train_arr, "toarray"):
                input_feature_train_arr = input_feature_train_arr.toarray()
            if hasattr(input_feature_test_arr, "toarray"):
                input_feature_test_arr = input_feature_test_arr.toarray()

            logging.info(f"input_feature_train_arr shape: {input_feature_train_arr.shape}")
            logging.info(f"target_feature_train_df shape: {np.array(target_feature_train_df).shape}")

            # Reshape target arrays
            target_feature_train_df = np.array(target_feature_train_df).reshape(-1, 1)
            target_feature_test_df = np.array(target_feature_test_df).reshape(-1, 1)

            # Combine transformed features and target variable into final train and test arrays
            train_arr = np.c_[input_feature_train_arr, target_feature_train_df]
            test_arr = np.c_[input_feature_test_arr, target_feature_test_df]

            logging.info("Saving preprocessing object")
            # Save the preprocessor for future use
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)

