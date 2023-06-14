# LOVM - Language-Only Vision Model Selection
#### [Orr Zohar](https://orrzohar.github.io/), Mars Huang, [Jackson Wang](https://wangkua1.github.io/), [Serena Yeung](https://marvl.stanford.edu/people.html)

## Abstract

Pre-trained multi-modal vision-language models (VLMs) are becoming increasingly popular due to their exceptional performance on downstream vision applications, particularly in the few- and zero-shot settings. 
However, selecting the best-performing VLM for some downstream applications is non-trivial, as it is dataset and task-dependent. Meanwhile, the exhaustive evaluation of all available VLMs on a novel application is not only time and  computationally demanding but also necessitates the collection of a labeled dataset for evaluation. 
As the number of open-source VLM variants increases, there is a need for an efficient model selection strategy that does not require access to a curated evaluation dataset. 
This paper proposes a novel task and benchmark for efficiently evaluating VLMs' zero-shot performance on downstream applications without access to the downstream task dataset. 
Specifically, we introduce a new task LOVM: **L**anguage-**O**nly  **V**ision  **M**odel Selection, where methods are expected to perform both model selection and performance prediction based solely on a text description of the desired downstream application.
We then introduced an extensive LOVM benchmark consisting of ground-truth evaluations of 35 pre-trained VLMs and 23 datasets, where methods are expected to rank the pre-trained VLMs and predict their zero-shot performance. 

# Installation

Start by cloning this repository and install the dependencies.  

```bash
$ git clone git@github.com:orrzohar/LOVM.git
```
And installing the requirements in requirements.txt or environment.yml.



# LOVM evaluation

Evaluate your prediction results using LOVM: 

```python
from lovm import LOVM

lovm = LOVM()
metrics = lovm.evaluate_model_pred(model_pred)
print(metrics)
```

## LOVM Ablation 

To ablate modelGPT, and generate the ablation tables in the manscript, we run:
```bash 
python generate_results.py --model_type linear_regression --pred_type model_rank --ablate_subset true
python generate_results.py --model_type linear_regression --pred_type model_pred --ablate_subset true
```

Do ablation on the model type by removing the --model_type flag
```bash 
python generate_results.py --model_type linear_regression --pred_type dataset_rank --ablate_subset true
```

Add in hyperparameter search by add in --grid_search flag
```bash 
python generate_results.py --model_type linear_regression --pred_type dataset_rank --grid_search --ablate_subset true
```

To ablate specific set of features
```bash 
python generate_results.py --model_type linear_regression --pred_type dataset_rank --features text-f1,intraclass_sim,inter_close --ablate_subset true
```

To evaluate specific set of features
```bash 
python generate_results.py --model_type linear_regression --pred_type dataset_rank --features text-f1,intraclass_sim,inter_close
```

## Adding your own LOVM method:
To add your own LOVM method, please implement it in a subdir. 

It should be capable of both performing VLM prediction and ranking. 
1. When evaluating model ranking on a dataset, you cannot use ground-truth evaluations of that dataset to make your prediction
2. When evaluation performance prediction of some model on some dataset, you cannot use ground-truth evaluations that include either the model or the dataset in question.
