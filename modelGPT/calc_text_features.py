from utils import calc_data_metrics,eval_on_dataset, calculate_synonym_simularity
import pickle
import pandas as pd
import numpy as np
import yaml
from tqdm import tqdm
import argparse


def get_args_parser():
    parser = argparse.ArgumentParser('modelGPT: encoding text prompts', add_help=False)
    parser.add_argument('--model_config_name', default='LOVM/models.yml', type=str)
    parser.add_argument('--datasets', default='LOVM/datasets.txt', type=str)
    parser.add_argument('--gt_eval_table', default='LOVM/eval_table.csv', type=str)
    parser.add_argument('--model_dir', default='modelGPT/models', type=str)
    parser.add_argument('--syn_dir', default='modelGPT/syn_dataset', type=str)
    parser.add_argument('--enc_dir', default='modelGPT/encoded_syn_dataset', type=str)
    parser.add_argument('--eval_table_metrics', default='modelGPT/eval_table_features.csv', type=str)
    parser.add_argument('--text_dataset_sigma', default=0.1, type=float)
    return parser

def main(args):
    eval_table = pd.read_csv(args.gt_eval_table)

    with open(args.datasets, 'r') as f:
        datasets = [line.rstrip('\n') for line in f.readlines()]

    with open(args.model_config_name, 'r') as file:
        model_names = yaml.safe_load(file)

    model_names = [tuple(m) for m in model_names]

    sigma=args.text_dataset_sigma

    for dataset in tqdm(datasets):

        captions_data ={}
        with open(f"modelGPT/encoded_captions_dataset_new/{dataset}.pkl", 'rb') as file:
            while True:
                try:
                    loaded_object = pickle.load(file)
                    captions_data.update(loaded_object)
                except EOFError:
                    break

        syn_data = {}
        with open(f"modelGPT/encoded_syn_dataset_new/{dataset}.pkl", 'rb') as file:
            while True:
                try:
                    loaded_object = pickle.load(file)
                    syn_data.update(loaded_object)
                except EOFError:
                    break

        models = {}
        with open(f"modelGPT/models_new/{dataset}.pkl", 'rb') as file:
            while True:
                try:
                    loaded_object = pickle.load(file)
                    models.update(loaded_object)
                except EOFError:
                    break

        for m in model_names:

            in_tmp=eval_table.loc[(eval_table.model_fullname==m[0]+" "+m[1])&(eval_table.dataset=="imagenet1k")]
            eval_table.loc[(eval_table.model_fullname==m[0]+" "+m[1])&(eval_table.dataset==dataset), 'IN-score']=float(in_tmp.mean_per_class_recall.iloc[0])

            tmp=eval_on_dataset(captions_data[m], models[m], sigma)
            eval_table.loc[(eval_table.model_fullname==m[0]+" "+m[1])&(eval_table.dataset==dataset), 'text-f1']=tmp[0]
            eval_table.loc[(eval_table.model_fullname==m[0]+" "+m[1])&(eval_table.dataset==dataset), 'text-acc1']=tmp[1]

            max_intraclass, inter_close, intra_close = calc_data_metrics(captions_data[m], models[m])
            eval_table.loc[(eval_table.model_fullname==m[0]+" "+m[1])&(eval_table.dataset==dataset), 'intraclass_sim']=float(max_intraclass)
            eval_table.loc[(eval_table.model_fullname==m[0]+" "+m[1])&(eval_table.dataset==dataset), 'inter_close']=inter_close
            eval_table.loc[(eval_table.model_fullname==m[0]+" "+m[1])&(eval_table.dataset==dataset), 'intra_close']=intra_close

            superclass_sim = calculate_synonym_simularity(syn_data[m], models[m])
            eval_table.loc[(eval_table.model_fullname==m[0]+" "+m[1])&(eval_table.dataset==dataset), 'superclass_metric']=superclass_sim
            #import ipdb; ipdb.set_trace()

    eval_table.dropna(subset=['IN-score'], inplace=True)
    eval_table.to_csv(args.eval_table_metrics)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('modelGPT encode', parents=[get_args_parser()])
    args = parser.parse_args()    
    main(args)