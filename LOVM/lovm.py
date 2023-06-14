"""The evaluation class for LOVM.

LOVM is a benchmark for evaluating the performance of LVMs using langauge alone.
The LOVM class is used to evaluate the performance of a LVM on the LOVM benchmark.
The class contains evalautions functions, where users can input their predictions.

Example using ModelGPTPredictor to predict dataset rank and evaluate the
performance of the model using LOVM:
    .. highlight:: python
    .. code-block:: python

    >> from model_gpt_predictor import ModelGPTPredictor
    >> model_gpt = ModelGPTPredictor(gt_df)
    >> dataset_rank_pred = model_gpt.loo_dataset_rank()

    >> lovm = LOVM()
    >> metric = lovm.evaluate_model_pred(dataset_rank_pred)
    >> print(metric)

    {'l1': 0.13488552697637365}

"""
#import ipdb; ipdb.set_trace()
import pandas as pd
import numpy as np
from collections import defaultdict
from scipy.stats import kendalltau
from modelGPT.model_gpt_predictor import ModelGPTPredictor
from typing import Iterable, Callable
from modelGPT.constants import (
    ALL_FEATURES, GROUND_TRUTH_CSV, MODEL_NAME_COL, DATASET_COL, PRED_TARGET,
    DATASET_TO_REMOVE, NUM_RANK, MODEL_TO_REMOVE
)


def get_acc(pred, true, num_rank):
    pred = pred[:num_rank]
    true = true[:num_rank]
    return np.sum(
        [i in pred.index for i in true.index]
    )/len(true)

def get_k_tau(pred, true, num_rank):
    pred = pred[:num_rank]
    true = true[:num_rank]
    true = true.rename_axis('index').reset_index()
    pred = pred.rename_axis('index').reset_index()
    col_name = true.columns[-1]

    merge = pd.merge(true, pred, on='index', suffixes=('_true', '_pred'))

    true = merge[f"{col_name}_true"].rank(ascending=False).astype(int)
    pred = merge[f"{col_name}_pred"].rank(ascending=False).astype(int)

    if len(true) > 2:
        return kendalltau(true, pred).correlation
    return 0


def get_l1(pred, true, num_rank):
    pred = pred[:num_rank]
    return np.abs(np.nanmean(pred-true))

METRIC_DICT = {
    'acc': get_acc,
    'k_tau': get_k_tau,
    'l1': get_l1,
}


class LOVM:
    f"""LOVM benchmark evalation class.

    Dataset ranking:
        - Input: a dataframe with columns: model_fullname, rows: dataset
        - Function: Use the self.evaluate_dataset_ranking() method to evaluate
        - Returns: accuracy, Kendall's tau

    Model ranking:
        - Input: a dataframe with columns: dataset, rows: model_fullname
        - Function: Use the self.evaluate_model_ranking() method to evaluate
        - Returns: accuracy, Kendall's tau

    Model prediction:
        - Input: a dataframe with columns: dataset, rows: model_fullname
        - Function: Use the self.evaluate_model_pred() method to evaluate
        - Returns: L1 loss

    Args:
        num_rank (int, optional): number of datasets to rank. Defaults to {NUM_RANK}.
        return_mean (bool, optional): whether to return the mean of the metrics.
    """

    def __init__(self, num_rank:int = NUM_RANK, return_mean:bool = True):

        self.num_rank = num_rank
        self.return_mean = return_mean
        self.gt_df = pd.read_csv(GROUND_TRUTH_CSV)

        self.imagenet_df = self.gt_df[~self.gt_df[DATASET_COL].isin([d for d in DATASET_TO_REMOVE if d != 'imagenet1k'])]
        self.imagenet_df = self.imagenet_df[~self.imagenet_df[MODEL_NAME_COL].isin(MODEL_TO_REMOVE)]
        self.gt_df = self.gt_df[~self.gt_df[DATASET_COL].isin(DATASET_TO_REMOVE)]
        self.gt_df = self.gt_df[~self.gt_df[MODEL_NAME_COL].isin(MODEL_TO_REMOVE)]

        self.dataset_rank_gt_df = pd.pivot_table(
            self.gt_df, values=PRED_TARGET, index=[DATASET_COL],
            columns=MODEL_NAME_COL
        )
        self.model_rank_gt_df = pd.pivot_table(
            self.gt_df, values=PRED_TARGET, index=[MODEL_NAME_COL],
            columns=DATASET_COL
        )

        self.imagenet_df = pd.pivot_table(
            self.imagenet_df, values=PRED_TARGET, index=[MODEL_NAME_COL],
            columns=DATASET_COL
        )


    def _evaluate(
            self, pred_df:pd.DataFrame, target_df:pd.DataFrame,
            eval_func_names:Iterable[Callable]
        )->pd.DataFrame:
        """Genernic function for evalauting a dataframe of predictions against
        a dataframe of targets. The evaluation functions are specified by the
        eval_func_names argument.

        Args:
            pred_df (pd.DataFrame): dataframe of predictions
            target_df (pd.DataFrame): dataframe of targets
            eval_func_names (Iterable[Callable]): list of evaluation functions

        Returns:
            pd.DataFrame: dataframe of evaluation results
        """

        result_dict = defaultdict(list)
        for metric in eval_func_names:
            for c in pred_df.columns:
                pred = pred_df.sort_values(c, ascending=False)[c]
                target = target_df.sort_values(c, ascending=False)[c]
                result_dict[metric].append(METRIC_DICT[metric](pred, target, self.num_rank))

        # convert to dataframe
        result_dict['cols'] = list(pred_df.columns)
        result_df = pd.DataFrame(result_dict)
        result_df = result_df.set_index('cols').sort_index()

        # return mean or full dataframe
        if self.return_mean:
            result_df.loc['mean'] = result_df.mean()

        return result_df


    def evaluate_dataset_rank(
            self, dataset_rank_pred_df:pd.DataFrame
        )->pd.DataFrame:
        """Wrappper function around self._evalate to evalaute a dataframe of
        dataset ranks against a dataframe of true dataset ranks.
        The evaluation functions are specified by the eval_func_names argument.

        Args:
            pred_df (pd.DataFrame): dataframe of predictions
            target_df (pd.DataFrame): dataframe of targets
            eval_func_names (Iterable[Callable]): list of evaluation functions

        Returns:
            pd.DataFrame: dataframe of evaluation results
        """
        return self._evaluate(
            dataset_rank_pred_df, self.dataset_rank_gt_df, ['acc', 'k_tau']
        )

    def evaluate_model_rank(
            self, model_rank_pred_df:pd.DataFrame
        )->pd.DataFrame:
        """Wrappper function around self._evalate to evalaute a dataframe of
        model ranks against a dataframe of true model ranks.
        The evaluation functions are specified by the eval_func_names argument.

        Args:
            pred_df (pd.DataFrame): dataframe of predictions
            target_df (pd.DataFrame): dataframe of targets
            eval_func_names (Iterable[Callable]): list of evaluation functions

        Returns:
            pd.DataFrame: dataframe of evaluation results
        """
        return self._evaluate(
            model_rank_pred_df, self.model_rank_gt_df, ['acc', 'k_tau']
        )

    def evaluate_model_pred(
            self, model_pred_pred_df:pd.DataFrame
        )->pd.DataFrame:
        """Wrappper function around self._evalate to evalaute a dataframe of
        model predictions against a dataframe of true model predictions.
        The evaluation functions are specified by the eval_func_names argument.

        Args:
            pred_df (pd.DataFrame): dataframe of predictions
            target_df (pd.DataFrame): dataframe of targets
            eval_func_names (Iterable[Callable]): list of evaluation functions

        Returns:
            pd.DataFrame: dataframe of evaluation results
        """
        return self._evaluate(
            model_pred_pred_df, self.model_rank_gt_df, ['l1']
        )

    def get_imagenet_model_rank(self) -> pd.DataFrame:
        """Get the imagenet model rank from the ground truth dataframe.

        Returns:
            pd.DataFrame: dataframe of imagenet model rank
        """

        imagenet_pred = self.imagenet_df.copy()
        for c in imagenet_pred.columns:
            imagenet_pred[c] = self.imagenet_df['imagenet1k']
        return imagenet_pred

    def get_imagenet_model_pred(self) -> pd.DataFrame:
        """Get the imagenet model rank from the ground truth dataframe.

        Returns:
            pd.DataFrame: dataframe of imagenet model rank
        """
        imagenet_pred = self.imagenet_df.copy()
        for c in imagenet_pred.columns:
            imagenet_pred[c] = self.imagenet_df['imagenet1k']
        return imagenet_pred

    def get_imagenet_dataset_rank(self) -> pd.DataFrame:
        """Get the imagenet model rank from the ground truth dataframe.

        Returns:
            pd.DataFrame: dataframe of imagenet model rank
        """
        imagenet_pred = self.imagenet_df.copy()
        for c in imagenet_pred.columns:
            imagenet_pred[c] = self.imagenet_df['imagenet1k']
        return imagenet_pred.T

def eval_lovm(pred, gt, type='model_rank'):
    lovm = LOVM()
    if type =='model_rank':
        imagenet_baseline = lovm.get_imagenet_model_pred()
        print("ImageNet baseline")
        print(lovm.evaluate_model_rank(imagenet_baseline))
        print("###############")
        print(lovm.evaluate_model_rank(model_rank_pred))


def main():
    gt_df = pd.read_csv(GROUND_TRUTH_CSV)
    gt_df = gt_df[~gt_df[DATASET_COL].isin(DATASET_TO_REMOVE)]
    gt_df = gt_df[~gt_df[MODEL_NAME_COL].isin(MODEL_TO_REMOVE)]


    lovm = LOVM()
    imagenet_baseline = lovm.get_imagenet_model_rank()
    model_gpt = ModelGPTPredictor(gt_df)
    model_rank_rank, best_params = model_gpt.loo_model_rank()

    import ipdb; ipdb.set_trace()
    df_eval = lovm.evaluate_model_rank(model_rank_rank)
    print(gen_latex(df_eval.acc.tolist()))
    print(gen_latex(df_eval.k_tau.tolist()))



def main_pred():
    gt_df = pd.read_csv(GROUND_TRUTH_CSV)
    gt_df = gt_df[~gt_df[DATASET_COL].isin(DATASET_TO_REMOVE)]
    gt_df = gt_df[~gt_df[MODEL_NAME_COL].isin(MODEL_TO_REMOVE)]


    lovm = LOVM()
    imagenet_baseline = lovm.get_imagenet_model_pred()
    model_gpt = ModelGPTPredictor(gt_df)
    model_pred, _ = model_gpt.loo_model_pred()

    import ipdb; ipdb.set_trace()
    df_eval = lovm.evaluate_model_pred(model_pred)
    print(gen_latex(df_eval.l1.tolist()))


def main_dataset():
    gt_df = pd.read_csv(GROUND_TRUTH_CSV)
    gt_df = gt_df[~gt_df[DATASET_COL].isin(DATASET_TO_REMOVE)]
    gt_df = gt_df[~gt_df[MODEL_NAME_COL].isin(MODEL_TO_REMOVE)]

    lovm = LOVM()
    imagenet_baseline = lovm.get_imagenet_model_rank()
    model_gpt = ModelGPTPredictor(gt_df, features=ALL_FEATURES)
    model_rank_pred, best_params = model_gpt.loo_dataset_rank()

    print("ImageNet baseline")
    df_eval = lovm.evaluate_model_rank(imagenet_baseline)
    print(gen_latex(df_eval.acc.tolist()))
    print(gen_latex(df_eval.k_tau.tolist()))

    df_eval = lovm.evaluate_model_rank(model_rank_rank)
    print(gen_latex(df_eval.acc.tolist()))
    print(gen_latex(df_eval.k_tau.tolist()))

def gen_latex(list):
    return " \hspace{-0.4em} & \hspace{-0.9em} ".join([str(round(f, 2)) for f in list[:-1]] + [str(round(list[-1],3))])

if __name__ == "__main__":
    main_pred()
