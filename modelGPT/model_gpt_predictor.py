"""The modelGPT predictor class. 

This class contains functions to predict dataset rank, model ranking or model prediction. 

Example using ModelGPTPredictor to predict dataset rank:
    .. highlight:: python
    .. code-block:: python

    >> from model_gpt_predictor import ModelGPTPredictor
    >> model_gpt = ModelGPTPredictor(df)
    >> dataset_rank_pred = model_gpt.loo_dataset_rank()
"""

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from typing import Iterable, Type, Tuple, Callable

import pandas as pd 
import numpy as np

from modelGPT.constants import ALL_FEATURES, MODEL_NAME_COL, DATASET_COL, PRED_TARGET, GROUND_TRUTH_CSV, DATASET_TO_REMOVE, PARAMETERS, MODELS



class ModelGPTPredictor:
    f"""ModelGPTPredictor class
    
    Args:
        df (pd.DataFrame): dataframe containing tested models, tested datasets,
            their corresponding performance metrics, and features used to train
            the model.
        features (Iterable[str], optional): features used to train the model.
            Defaults to {ALL_FEATURES}.
        model (Object, optional): model used to train the model. 
            Defaults to LinearRegression().
        grid_search_params (dict, optional): grid search parameters. 
            Defaults to None, which does not perform grid search.
        model_name_col (str, optional): column name of the pretrained LVM models
            in the dataframe. Defaults to {MODEL_NAME_COL}.
        dataset_col (str, optional): column name of the evaluatstion datasets 
            in the dataframe. Defaults to {DATASET_COL}.
    """
    
    def __init__(
            self, 
            df:pd.DataFrame, 
            features:Iterable[str] = ALL_FEATURES,
            model:Type = MODELS['linear_regression'],
            grid_search_params:dict = PARAMETERS['linear_regression'],
            model_name_col:str =MODEL_NAME_COL,
            dataset_col:str =DATASET_COL,
        ):
        self.df = df

        self.grid_search_params = grid_search_params
        self.features = features
        self.model = model
        self.model_name_col = model_name_col
        self.dataset_col = dataset_col

        self.all_datasets = self.df.dataset.unique()
        self.all_models = self.df.model_fullname.unique()


    def fit_predict_model(
            self, x_train:pd.DataFrame, x_test:pd.DataFrame, 
            y_train:pd.DataFrame, 
        ) -> Tuple[np.ndarray, dict]:
        """Fit the prediction model based on parameters x_trian and y_train, 
        and subsequently predict the target values based on x_test.

        Args:
            x_train (pd.DataFrame): training features.
            x_test (pd.DataFrame): testing features.
            y_train (pd.DataFrame): training target.

        Returns:
            Tuple[np.ndarray, dict]: predicted target values and best parameters
        """

        # fit and predict model
        if self.grid_search_params is not None:
            gsc = GridSearchCV(
                estimator=self.model,
                param_grid=self.grid_search_params,
                cv=10, 
                scoring='neg_mean_squared_error',
                n_jobs=3
            )
            gsc.fit(x_train, y_train)
            best_params, best_model = gsc.best_params_, gsc.best_estimator_
            model_pred = best_model.predict(x_test)
        else: 
            self.model.fit(x_train, y_train)
            best_params = None
            model_pred = self.model.predict(x_test)

        return model_pred, best_params   
    
    def _predict_target(
            self, split_col:str, split_val:str, target_type:str, 
            pred_target=PRED_TARGET
        ) -> Tuple[pd.DataFrame, dict]:
        f"""Generic function to predict the pred_target based on features.
        Since this is an evaluation of leave one out, we split the data
        based on the split_col and split_val.

        Args:
            split_col (str): column name to split the data (model or dataset).
            split_val (str): value to split the data (the specific model 
                or dataset to split on).
            target_type (str): column name of the target (model or dataset).
            pred_target (str, optional): column name of the target to predict.
        
        Returns:
            Tuple[pd.DataFrame, dict]: predicted target values mapped to their 
                corresponding target_type and best parameters for the 
                prediction model 
        """

        # leave one out evaluation
        x_train = self.df.loc[self.df[split_col] != split_val][self.features]
        x_test = self.df.loc[self.df[split_col] == split_val][self.features]
        y_train = self.df.loc[self.df[split_col] != split_val][pred_target].values.ravel()
        
        # fit predict model
        model_pred, best_params = self.fit_predict_model(x_train, x_test, y_train) 

        # map target to prediction
        test_pred = self.df[self.df[split_col] == split_val].reset_index()
        test_pred[f"{split_val}"] = model_pred
        test_pred = test_pred[[target_type, f"{split_val}"]]
        test_pred = test_pred.set_index(target_type)
        return test_pred, best_params

    def predict_dataset_rank(self, test_model:str) -> Tuple[pd.DataFrame, dict]:
        """Wrapper function around self._predict_target to predict dataset ranking

        Args:
            test_model (str): model split used as test set. 

        Returns:
            Tuple[pd.DataFrame, dict]: predicted target values mapped to their 
                corresponding target_type and best parameters for the 
                prediction model 
        """
        return self._predict_target(
            split_col=self.model_name_col, 
            split_val=test_model, 
            target_type=self.dataset_col,
            pred_target=PRED_TARGET,
        )
        
    def predict_model_rank(self, test_dataset:set) -> Tuple[pd.DataFrame, dict]:
        """Wrapper function around self._predict_target to predict model ranking

        Args:
            test_dataset (str): dataset split used as test set. 

        Returns:
            Tuple[pd.DataFrame, dict]: predicted target values mapped to their 
                corresponding target_type and best parameters for the 
                prediction model 
        """

        # predict normalied metric
        df_list= []
        for d in np.unique(self.df.dataset):
            df_dataset = self.df.loc[self.df.dataset == d].copy()
            dataset_mean = df_dataset[PRED_TARGET].mean()
            df_dataset[f"norm_{PRED_TARGET}"] = df_dataset[PRED_TARGET] - dataset_mean
            df_list.append(df_dataset)
        self.df = pd.concat(df_list)

        return self._predict_target(
            split_col=self.dataset_col, 
            split_val=test_dataset, 
            target_type=self.model_name_col, 
            pred_target=f'norm_{PRED_TARGET}',
        )

    def predict_model_prediction(
            self, test_dataset:set
        ) -> Tuple[pd.DataFrame, dict]:
        """Wrapper function around self._predict_target to predict model prediction

        Args:
            test_dataset (str): dataset split used as test set. 

        Returns:
            Tuple[pd.DataFrame, dict]: predicted target values mapped to their 
                corresponding target_type and best parameters for the 
                prediction model 
        """
        return self._predict_target(
            split_col=self.dataset_col, 
            split_val=test_dataset, 
            target_type=self.model_name_col, 
            pred_target=PRED_TARGET,
        )
    
    def _loo_prediction(
            self, loo_params:Iterable, prediction_func:Callable
        ) -> pd.DataFrame:
        """Generic function for leave one out (loo) prediction.

        Args:
            loo_params (Iterable): list of parameters to leave out 
                (model_name or dataset)
            prediction_func (Callable): function used to predict the target
                (self.predict_dataset_rank or self.predict_model_rank 
                or self.predict_model_prediction)
        
        Returns:
            pd.DataFrame: dataframe of all predictions
        """
        all_test_pred = []
        for test_param in loo_params:
            test_pred, best_params = prediction_func(test_param)
            test_pred = test_pred.drop_duplicates()
            all_test_pred.append(test_pred)
        return pd.concat(all_test_pred, axis=1), best_params
   
    def loo_dataset_rank(self):
        """Wrapper function around self._loo_prediction for dataset ranking 
        leave one out (loo) prediction. calls self._loo_prediction with 
        self.predict_dataset_rank
        
        Returns:
            pd.DataFrame: dataframe of all predictions
        """
        return self._loo_prediction(self.all_models, self.predict_dataset_rank)

    def loo_model_rank(self):
        """Wrapper function around self._loo_prediction for model ranking 
        leave one out (loo) prediction. calls self._loo_prediction with 
        self.predict_model_rank

        Returns:
            pd.DataFrame: dataframe of all predictions
        """
        return self._loo_prediction(self.all_datasets, self.predict_model_rank)
    
    def loo_model_pred(self):
        """Wrapper function around self._loo_prediction for model prediction 
        leave one out (loo) prediction. calls self._loo_prediction with 
        self.predict_model_prediction
        
        Returns:
            pd.DataFrame: dataframe of all predictions
        """
        return self._loo_prediction(self.all_datasets, self.predict_model_prediction)
    

#### TODO
### REMOVE - for debug only
def main(): 

    gt_df = pd.read_csv(GROUND_TRUTH_CSV)
    gt_df = gt_df[~gt_df[DATASET_COL].isin(DATASET_TO_REMOVE)]

    model_gpt = ModelGPTPredictor(gt_df)
    print(model_gpt.loo_dataset_rank())

if __name__ == "__main__":
    main()
