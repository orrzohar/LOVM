import pandas as pd
import argparse
import tqdm

from modelGPT.constants import MODELS, PARAMETERS, FEATURE_ORDER_DICT, ALL_FEATURES, FEATURES_CSV, FEATURES_SET
from itertools import chain, combinations
from modelGPT.model_gpt_predictor import ModelGPTPredictor
from LOVM.lovm import LOVM
from collections import defaultdict
from typing import Iterable, Union

from LOVM.latex_util import (
    dataset_rank_abalation_latex,
    model_rank_abalation_latex,
    model_pred_abalation_latex,
    model_main_table
)


# get all feature combinations
def create_all_subsets(ss):
    all_sets = list(chain(*map(lambda x: combinations(ss, x), range(0, len(ss) + 1))))
    return [list(s) for s in all_sets if len(list(s)) > 0]


def run_ablation(
        df_features: str = FEATURES_CSV,
        prediction: str = 'dataset_rank',
        features_set: Iterable[str] = ALL_FEATURES,
        model_set: Union[Iterable[str], str] = 'linear_regression',
        grid_search: bool = False,
        ablate_subset: bool = True,
        pred_target='acc1',
) -> pd.DataFrame:
    f"""Run ablation study for all models and feature combinations.

    Args:
        df_features (str, optional): csv name containing all features and target.
        prediction (str, optional): Prediction type 
            (one of dataset_rank, model_rank, model_pred). 
            Defaults to 'dataset_rank'.
        features_set (Iterable[str], optional): Features to ablate. 
            Defaults to {ALL_FEATURES}.
        model_set (Any(Iterable[str], str), optional): Models to ablate.
            Defaults to 'linear_regression'.
        grid_search (bool, optional): Whether to perform grid search.

    Returns:
        pd.DataFrame: Dataframe containing results.
    """

    # dict to store results
    full_results = []
    results = defaultdict(list)

    # check if prediciton type is valid
    if prediction not in ['dataset_rank', 'model_rank', 'model_pred']:
        raise ValueError(f"prediction must be either 'dataset_rank', 'model_rank' or 'model_pred', got {prediction}")

    # store model type in list if there is only one model type
    if type(model_set) == str:
        model_set = [model_set]

    # sort feature set
    features_set = sorted(features_set, key=lambda x: FEATURE_ORDER_DICT[x])

    # create all feature combinations for ablation
    if ablate_subset:
        all_subsets = create_all_subsets(features_set)
    else:
        all_subsets = [features_set]

    # loop through all models to ablate
    for model_type in tqdm.tqdm(model_set, total=len(model_set)):

        # select model and get grid search parameters
        model = MODELS[model_type]
        if grid_search:
            grid_search_params = PARAMETERS[model_type]
        else:
            grid_search_params = None

        # loop through all feature combinations
        for ss in tqdm.tqdm(all_subsets, total=len(all_subsets), desc=model_type):

            # get all models and datasets
            model_gpt = ModelGPTPredictor(
                df_features, features=ss, model=model, grid_search_params=grid_search_params, pred_target=pred_target)
            lovm = LOVM(pred_target=pred_target)

            # specific prediction task
            if prediction == 'dataset_rank':
                if len(ss) == 1 and ss[0] == 'IN-score':
                    pred = lovm.get_imagenet_dataset_rank()
                    best_param = None
                else:
                    pred, best_param = model_gpt.loo_dataset_rank()

                metric = lovm.evaluate_dataset_rank(pred)

                results['acc'].append(metric.loc['mean', 'acc'])
                results['k_tau'].append(metric.loc['mean', 'k_tau'])
            elif prediction == 'model_rank':
                if len(ss) == 1 and ss[0] == 'IN-score':
                    pred = lovm.get_imagenet_model_rank()
                    best_param = None
                else:
                    pred, best_param = model_gpt.loo_model_rank()

                metric = lovm.evaluate_model_rank(pred)
                results['acc'].append(metric.loc['mean', 'acc'])
                results['k_tau'].append(metric.loc['mean', 'k_tau'])
            else:
                if len(ss) == 1 and ss[0] == 'IN-score':
                    pred = lovm.get_imagenet_model_pred()
                    best_param = None
                else:
                    pred, best_param = model_gpt.loo_model_pred()

                metric = lovm.evaluate_model_pred(pred)
                results['l1'].append(metric.loc['mean', 'l1'])

            full_results.append(metric)
            results['features'].append(ss)
            results['model'].append(model_type)
            results['best_param'].append(best_param)

    # aggregate results to dataframe
    results_df = pd.DataFrame.from_dict(results)
    results_df = round(results_df, 3)
    results_df['num_features'] = results_df.features.apply(lambda x: len(x))
    return results_df, full_results


def main(args):
    # set parameters from argparse
    if args.model_type is None:
        model_set = MODELS.keys()
    else:
        model_set = [args.model_type]

    if args.scores is not None:
        features_set = []
        for f in args.scores.split(','):
            features_set += FEATURES_SET[f]
    elif args.subscores is not None:
        features_set = args.subscores.split(',')
    else:
        features_set = ALL_FEATURES

    # run ablation study
    results_df, full_results = run_ablation(
        args.features_csv,
        prediction=args.pred_type,
        features_set=features_set,
        model_set=model_set,
        grid_search=args.grid_search,
        ablate_subset=args.no_subsets,
        pred_target=args.pred_target
    )

    # print latex
    if args.pred_type == 'dataset_rank':
        print(dataset_rank_abalation_latex(results_df))
    elif args.pred_type == 'model_rank':
        if args.print_full_table:
            max_acc_row = results_df[results_df['acc'] == results_df['acc'].max()]

            # If there are multiple rows with the same maximum "acc" value,
            # select the one with the maximum "k_tau" value
            if len(max_acc_row) > 1:
                max_acc_row = max_acc_row[max_acc_row['k_tau'] == max_acc_row['k_tau'].max()]

            full_results = [f for f in full_results if round(f['acc']['mean'], 3) == max_acc_row['acc'].max()]
            full_results = [f for f in full_results if round(f['k_tau']['mean'], 3) == max_acc_row['k_tau'].max()]
            model_main_table(full_results[0])

        if args.print_ablation:
            max_acc_row = results_df[results_df['acc'] == results_df['acc'].max()]

            # If there are multiple rows with the same maximum "acc" value,
            # select the one with the maximum "k_tau" value
            if len(max_acc_row) > 1:
                max_acc_row = max_acc_row[max_acc_row['k_tau'] == max_acc_row['k_tau'].max()]

            max_acc_row['scores'] = '+'.join(args.scores.split(','))
            print(max_acc_row[['scores', 'k_tau', 'acc']].to_latex(index=False, escape=False, column_format='c|cc',
                                                                   float_format="%.3f"))

        else:
            print(model_rank_abalation_latex(results_df))


    elif args.pred_type == 'model_pred':
        if args.print_full_table:
            max_acc_row = results_df[results_df['l1'] == results_df['l1'].min()]
            full_results = [f for f in full_results if round(f['l1']['mean'], 3) == max_acc_row['l1'].min()]
            model_main_table(full_results[0])

        if args.print_ablation:
            min_l1_row = results_df[results_df['l1'] == results_df['l1'].min()]
            min_l1_row['scores'] = '+'.join(args.scores.split(','))
            print(min_l1_row[['scores', 'l1']].to_latex(index=False, escape=False, column_format='c|c',
                                                        float_format="%.3f"))

        else:
            print(model_pred_abalation_latex(results_df))
    else:
        print('unknown task requested')


if __name__ == "__main__":
    # parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--features_csv",
        type=str,
        help="Features csv path",
        default=FEATURES_CSV
    )
    parser.add_argument(
        "-m",
        "--model_type",
        type=str,
        help="model to fit",
        default=None
    )
    parser.add_argument(
        "-g",
        "--grid_search",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-p",
        "--pred_type",
        type=str,
        help="prediction type",
        default='model_pred'
    )
    parser.add_argument(
        "--scores",
        type=str,
        help="scores to use",
        default='G,C,INB',
    )
    parser.add_argument(
        "--subscores",
        type=str,
        help="subscores to use",
        default=None,
    )
    parser.add_argument(
        "--no_subsets",
        action="store_false",
        default=True,
    )
    parser.add_argument(
        "--print_full_table",
        type=str,
        help="print dataset breaks",
        default=False,
    )
    parser.add_argument(
        "--print_ablation",
        action="store_false",
        default=True,
    )
    parser.add_argument(
        "--pred_target",
        type=str,
        default='acc1',
    )
    # parse args
    args = parser.parse_args()

    main(args)
