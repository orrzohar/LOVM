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
import torch.nn.functional as F


def get_args_parser():
    parser = argparse.ArgumentParser('modelGPT: encoding text prompts', add_help=False)
    parser.add_argument('--model_config_name', default='LOVM/models.yml', type=str)
    parser.add_argument('--datasets', default='LOVM/datasets.txt', type=str)
    parser.add_argument('--templates_dir', default='LOVM/templates', type=str)
    parser.add_argument('--classes_dir', default='LOVM/classnames', type=str)
    parser.add_argument('--enc_dir', default='modelGPT/models', type=str)
    return parser


def calculate_classifier_weights(model, classnames, templates, tokenizer):
    """
    Calculates the zero-shot classifier weights using the contrastive (CLIP) model, the class names, the class templates, and the tokenizer. For every class, the text embeddings for all the templates is calculates via the model.text_encoder, and then averaged to get the final zero-shot weights.
    Args:
        model: a torch constrastive LVM model, wich has the '.encode_text' function that takes tokenized list-of-strings and outputs the text embeddings
        classnames: the class names for the classification problem
        templates: list of (function) templates (lambda a: f'___ {a} ____') to use.
        tokenizer: tokenizer function which takes in list of captions and returns list of tokens
    Returns:
        tensor of zeroshot weights (dim NxC, where N is the dimention of the featurespace of the LVM and C is the number of classes.
    """
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames, leave=False):
            texts = [template.replace('{c}', classname) for template in templates]  # format with class
            texts = tokenizer(texts).cuda()
            class_embeddings = model.encode_text(texts)
            class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1)
    return zeroshot_weights


def flip_dict(original_dict):
    flipped_dict = {}
    for key1, inner_dict in original_dict.items():
        for key2, value in inner_dict.items():
            # create a new dictionary entry with the flipped keys and values
            if key2 not in flipped_dict:
                flipped_dict[key2] = {}
            flipped_dict[key2][key1] = value
    return flipped_dict


existing_models = []


def main(args):
    with open(args.model_config_name, 'r') as file:
        model_names = yaml.safe_load(file)

    os.makedirs(args.enc_dir, exist_ok=True)
    print(f'running on models: {model_names}')
    model_names = [tuple(m) for m in model_names]
    model_names = [m for m in model_names if m not in existing_models]
    print('##############################')

    print(f'running on models: {model_names}')

    with open(args.datasets, 'r') as f:
        dataset_names = [line.rstrip('\n') for line in f.readlines()]

    for m in tqdm(model_names, desc="model"):
        model, _, _ = open_clip.create_model_and_transforms(m[0], m[1], cache_dir='./LOVM/.cache')
        model.to('cuda')
        tokenizer = open_clip.get_tokenizer(m[0])

        for dataset in tqdm(dataset_names, desc=str(m)):
            data = {m: {}}

            with open(f'{args.templates_dir}/{dataset}.txt', 'r') as f:
                templates = [line.rstrip('\n') for line in f.readlines()]

            with open(f'{args.classes_dir}/{dataset}.txt', 'r') as f:
                classes = [line.rstrip('\n') for line in f.readlines()]

            data[m][dataset] = calculate_classifier_weights(model, classes, templates, tokenizer)

            tmp = flip_dict(data)
            for dname, dvalue in tmp.items():
                with open(f'{args.enc_dir}/{dname}.pkl', 'ab') as file:
                    pickle.dump(dvalue, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('modelGPT encode', parents=[get_args_parser()])
    args = parser.parse_args()
    os.makedirs(args.enc_dir, exist_ok=True)
    main(args)

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
import torch.nn.functional as F


def get_args_parser():
    parser = argparse.ArgumentParser('modelGPT: encoding text prompts', add_help=False)
    parser.add_argument('--model_config_name', default='LOVM/models.yml', type=str)
    parser.add_argument('--datasets', default='LOVM/datasets.txt', type=str)
    parser.add_argument('--templates_dir', default='LOVM/templates', type=str)
    parser.add_argument('--classes_dir', default='LOVM/classnames', type=str)
    parser.add_argument('--enc_dir', default='modelGPT/models4', type=str)
    return parser


def calculate_classifier_weights(model, classnames, templates, tokenizer):
    """
    Calculates the zero-shot classifier weights using the contrastive (CLIP) model, the class names, the class templates, and the tokenizer. For every class, the text embeddings for all the templates is calculates via the model.text_encoder, and then averaged to get the final zero-shot weights.
    Args:
        model: a torch constrastive LVM model, wich has the '.encode_text' function that takes tokenized list-of-strings and outputs the text embeddings
        classnames: the class names for the classification problem
        templates: list of (function) templates (lambda a: f'___ {a} ____') to use.
        tokenizer: tokenizer function which takes in list of captions and returns list of tokens
    Returns:
        tensor of zeroshot weights (dim NxC, where N is the dimention of the featurespace of the LVM and C is the number of classes.
    """
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames, leave=False):
            texts = [template.replace('{c}', classname) for template in templates]  # format with class
            texts = tokenizer(texts).to('cuda')
            class_embeddings = model.encode_text(texts)
            class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding.detach().cpu())
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1)
    return zeroshot_weights


def flip_dict(original_dict):
    flipped_dict = {}
    for key1, inner_dict in original_dict.items():
        for key2, value in inner_dict.items():
            # create a new dictionary entry with the flipped keys and values
            if key2 not in flipped_dict:
                flipped_dict[key2] = {}
            flipped_dict[key2][key1] = value
    return flipped_dict


existing_models = []


def main(args):
    with open(args.model_config_name, 'r') as file:
        model_names = yaml.safe_load(file)

    print(f'running on models: {model_names}')
    model_names = [tuple(m) for m in model_names]
    model_names = [m for m in model_names if m not in existing_models]
    print('##############################')

    print(f'running on models: {model_names}')

    with open(args.datasets, 'r') as f:
        dataset_names = [line.rstrip('\n') for line in f.readlines()]

    for m in tqdm(model_names, desc="model"):
        model, _, _ = open_clip.create_model_and_transforms(m[0], m[1], cache_dir='./LOVM/.cache')
        model.to('cuda')
        tokenizer = open_clip.get_tokenizer(m[0])

        for dataset in tqdm(dataset_names, desc=str(m)):
            data = {m: {}}

            with open(f'{args.templates_dir}/{dataset}.txt', 'r') as f:
                templates = [line.rstrip('\n') for line in f.readlines()]

            with open(f'{args.classes_dir}/{dataset}.txt', 'r') as f:
                classes = [line.rstrip('\n') for line in f.readlines()]

            data[m][dataset] = calculate_classifier_weights(model, classes, templates, tokenizer)

            tmp = flip_dict(data)
            for dname, dvalue in tmp.items():
                with open(f'{args.enc_dir}/{dname}.pkl', 'ab') as file:
                    pickle.dump(dvalue, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('modelGPT encode', parents=[get_args_parser()])
    args = parser.parse_args()
    os.makedirs(args.enc_dir, exist_ok=True)
    main(args)

