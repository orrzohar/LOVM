from scipy.stats import pearsonr, kendalltau
from sklearn.metrics import r2_score, f1_score, accuracy_score
import torch.nn.functional as F
from numpy.linalg import norm
from tqdm import tqdm
from numpy import dot
import numpy as np
import torch


def gen_caption(template, classname):
    return template[0] + classname + template[1]


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
            texts = tokenizer(texts)
            class_embeddings = model.encode_text(texts)
            class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1)
    return zeroshot_weights


def generate_dataset(model, classnames, templates, tokenizer):
    with torch.no_grad():
        data = []
        for classname in tqdm(classnames):
            texts = [gen_caption(template, classname) for template in templates]  # same template for all classes
            texts = tokenizer(texts)
            text_features = model.encode_text(texts)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            data.append(text_features)
    return data


def eval_on_dataset(dataset, zero_shot_model, sigma):
    pred = []
    for i in range(len(dataset)):
        data = (dataset[i] + torch.randn(dataset[i].shape) * sigma).to(zero_shot_model.device)
        pred.append(np.argmax((data @ zero_shot_model).cpu().numpy(), axis=1))

    pred = np.concatenate(pred)
    y_true = np.concatenate([i * np.ones(data.shape[0], dtype=np.int32) for i, data in enumerate(dataset)])
    return f1_score(y_true, pred, average='macro'), accuracy_score(y_true, pred)


def cos_sim(a, b):
    return dot(a, b) / (norm(a) * norm(b))


def calc_rank_metrics(pred, gt, on):
    tmp = pred.merge(gt, on=on).rename(columns={'values_x': 'prediction', 'values_y': 'ground_truth'})
    tmp['pred_rank'] = tmp['prediction'].rank(ascending=False).astype(int)
    tmp['gt_rank'] = tmp['ground_truth'].rank(ascending=False).astype(int)
    return kendalltau(tmp['pred_rank'], tmp['gt_rank']), pearsonr(tmp['pred_rank'], tmp['gt_rank']), r2_score(
        tmp['prediction'], tmp['ground_truth'])


def calc_data_metrics(data, zeroshot_weights):
    inter_close = []
    intra_close = []
    for i in range(len(data)):
        d = F.normalize(data[i], dim=1).to(zeroshot_weights.device)
        cos_sim = (d @ zeroshot_weights)
        inter_close.append(cos_sim[:, i].mean().cpu().numpy())
        import ipdb; ipdb.set_trace()
        intra_close.append(cos_sim[:, [ii for ii in range(len(data)) if ii != i]].mean(dim=0).max().cpu().numpy())

    cosine_sim_matrix = (zeroshot_weights.T @ zeroshot_weights)
    cosine_sim_matrix.fill_diagonal_(0)
    max_per_class, _ = cosine_sim_matrix.max(dim=0)
    return float(max_per_class.mean()), np.mean(inter_close), np.mean(intra_close)


def calculate_synonym_simularity(syn_data, model):
    syn_weights = []
    for i, v in enumerate(syn_data.values()):
        try:
            syn = torch.cat(list(v.values()), dim=0).to(model.device)
            syn_weights.append((model[:, i] @ syn.T).mean().detach().cpu().numpy())
        except:
            import ipdb; ipdb.set_trace()
    return np.mean(syn_weights)


################### non-functional ####################
from joblib import Parallel, delayed


def compute_text_features(text, model, tokenizer):
    text = tokenizer(text)
    text_features = model.encode_text(text)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features


def generate_dataset_v2(model, classnames, templates, tokenizer, n_jobs=2):
    with torch.no_grad():
        text_captions = [[template(classname) for template in templates] for classname in classnames]
        data = Parallel(n_jobs=n_jobs)(
            delayed(compute_text_features)(text, model, tokenizer) for text in tqdm(text_captions))

    return data


def split_list_into_sublists(input_list, sublist_length):
    return [input_list[i:i + sublist_length] for i in range(0, len(input_list), sublist_length)]


def generate_dataset_v1(model, classnames, templates, tokenizer):
    with torch.no_grad():
        text_captions = tokenizer([template(classname) for template in templates for classname in classnames])
        text_features = model.encode_text(text_captions)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    return split_list_into_sublists(text_features, len(templates))
