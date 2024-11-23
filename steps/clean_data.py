import logging
import pandas as pd
from zenml import step
from src.data_cleaning import DataCleaning, DataDivideStrategy,DataPreprocessStrategy
from typing_extensions import Annotated
from typing import Tuple
@step
def clean_df(df: pd.DataFrame) ->Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
]:
    try:
        preprocess_statergy= DataPreprocessStrategy()
        data_cleaning= DataCleaning(df, preprocess_statergy)
        preprocess_data=data_cleaning.handle_data()

        divide_statergy= DataDivideStrategy()
        data_cleaning= DataCleaning(preprocess_data, divide_statergy)
        X_train, X_test,Y_train, y_test= data_cleaning.handle_data()
        logging.info("data cleaning completed")
        return X_train, X_test,Y_train, y_test
    except Exception as e:
        logging.error("Error in cleaning data: {}".format(e))
        raise e    

