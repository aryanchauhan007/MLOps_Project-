import logging
from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import  r2_score

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)
def calculate_rmse(y_true, y_pred):
     rmse = np.sqrt(mean_squared_error)


class Evaluation(ABC):
    """ 
    Abstract class defining statergy for evaluation our models
    """
    @abstractmethod
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculates the scores for the model 
        Args;
           y_true: True Labels
           y_pred: Predicted labels
        Returns:
            None   
        """
        pass 
class MSE(Evaluation):
    """
    Evaluation Statergy that uses Mean Squared Error
    """
    def calculate_score(self, y_true:np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculate MSE")
            mse= mean_squared_error(y_true, y_pred)
            logging.info("MSE: {}".format(mse))
            return mse
        except Exception as e:
            logging.error("Error in calculating MSE: {}".format(e))
            raise e    
            
class R2(Evaluation):
    """
    Evaluation Statergy that uses R2 Score
    """
    def calculate_score(self, y_true:np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating R2 Score")
            r2= r2_score(y_true, y_pred)
            logging.info("R2 Score: {}".format(r2))
            return r2
        except Exception as e:
            logging.error("Error in calculating R2 Score: {}".format(e))
            raise e  
        

class RSME(Evaluation):
    """
    Evaluation Statergy that uses Root Mean Squared Error
    """   
    def calculate_score(self, y_true:np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculate RSME")
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            logging.info("RSME: {}".format(rmse))
            return rmse
        except Exception as e:
            logging.error("Error in calculating RSME: {}".format(e))
            raise e
        

        
