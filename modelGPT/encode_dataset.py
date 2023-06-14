import open_clip
import pandas as pd
import numpy as np
import pickle
import json
import time
import os
import yaml
import argparse
from tqdm import tqdm
import torch
import time

def get_args_parser():
    parser = argparse.ArgumentParser('modelGPT: encoding text prompts', add_help=False)
    parser.add_argument('--model_config_name', default='LOVM/models.yml', type=str)
    parser.add_argument('--datasets', default='LOVM/datasets.txt', type=str)
    parser.add_argument('--continue_encode', default=False, type=bool)
    parser.add_argument('--text_dir', default='modelGPT/captions_dataset', type=str)
    parser.add_argument('--enc_dir', default='modelGPT/encoded_captions_dataset', type=str)
    return parser
    


def generate_dataset(model, tokenizer, captions):
    with torch.no_grad():
        data = []
        for classname, texts in tqdm(captions.items(), desc = 'Captions', leave=False):
            texts = tokenizer(texts).cuda()
            text_features = model.encode_text(texts)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            data.append(text_features.detach().cpu())
    return data


def run_one_model(model, m, tokenizer, datasets):
    data={}
    for d, captions in tqdm(datasets.items(),  desc=f'{m[0]}, {m[1]}', leave=False):
        data[d] = generate_dataset(model, tokenizer, captions)
    return data
    

def flip_dict(original_dict):
    flipped_dict = {}

    for key1, inner_dict in original_dict.items():
        for key2, value in inner_dict.items():
            # create a new dictionary entry with the flipped keys and values
            if key2 not in flipped_dict:
                flipped_dict[key2] = {}
            flipped_dict[key2][key1] = value 
    return flipped_dict
        
    
def main(args):
    with open(args.model_config_name, 'r') as file:
        model_names = yaml.safe_load(file)
        
    model_names = [tuple(m) for m in model_names]
    
    with open(args.datasets, 'r') as f:
        dataset_names = [line.rstrip('\n') for line in f.readlines()]
    
    datasets={}
    tokenizers = {}
    for d in dataset_names:
        with open(f'{args.text_dir}/{d}.json', 'r') as f:
            datasets[d] = json.load(f)
    
    data={}
    if args.continue_encode:
        for dname in dataset_names:
            with open(f'{args.enc_dir}/{dname}.pkl', 'rb') as f:
                data[dname] = pickle.load(f)
                
        data = flip_dict(data)
    
    model_names = [m for m in model_names if m not in data.keys()]        
    for m in tqdm(model_names, desc="model"):
        model, _, _ = open_clip.create_model_and_transforms(m[0], m[1], cache_dir='./LOVM/.cache')
        model.to('cuda')
        tokenizer = open_clip.get_tokenizer(m[0])
        data[m] = run_one_model(model,m, tokenizer, datasets)
            
        tmp  = flip_dict(data)
        for dname, dvalue in tmp.items():
            with open(f'{args.enc_dir}/{dname}.pkl', 'wb') as file:
                pickle.dump(dvalue, file)
           
    
if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser('modelGPT encode', parents=[get_args_parser()])
    args = parser.parse_args()    
    os.makedirs(args.enc_dir, exist_ok=True)  
    main(args)
    print(f'runtime: {time.time()-start_time} [sec]')

    