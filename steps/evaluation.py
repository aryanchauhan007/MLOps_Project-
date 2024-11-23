import logging
import mlflow
import pandas as pd
from zenml import step
from sklearn.base import RegressorMixin
from typing import Tuple
from typing_extensions import Annotated
from src.evaluation import MSE, R2,RSME
from zenml.client import Client

experiment_tracker= Client().active_stack.experiment_tracker


@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(model: RegressorMixin,
     X_test: pd.DataFrame,
     y_test: pd.DataFrame
)-> Tuple[
    Annotated[float,"r2_score"],
    Annotated[float,"rsme" ],
]:
    """
    Evaluates the model on the given data.

    agrs:
        df: the ingested data
    """
    try:
        prediction= model.predict(X_test)
        mse_class= MSE()
        mse= mse_class.calculate_score(y_test, prediction)
        mlflow.log_metric("mse", mse)

        r2_class=R2()
        r2= r2_class.calculate_score(y_test, prediction)
        mlflow.log_metric("r2", r2)

        rmse_class=RSME()
        rmse= rmse_class.calculate_score(y_test, prediction)
        mlflow.log_metric("rsme", rmse)

        return r2, rmse
    except Exception as e:
        logging.error("Error in evaluating model: {}".format(e))
        raise e