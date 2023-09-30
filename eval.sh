#!/bin/bash

python generate_results.py --model_type linear_regression --pred_type model_rank --scores 'INB' >> acc.txt
python generate_results.py --model_type linear_regression --pred_type model_rank --scores 'C' >> acc.txt
python generate_results.py --model_type linear_regression --pred_type model_rank --scores 'G' >> acc.txt
python generate_results.py --model_type linear_regression --pred_type model_rank --scores 'G,C' >> acc.txt
python generate_results.py --model_type linear_regression --pred_type model_rank --scores 'G,INB' >> acc.txt
python generate_results.py --model_type linear_regression --pred_type model_rank --scores 'C,INB' >> acc.txt
python generate_results.py --model_type linear_regression --pred_type model_rank --scores 'G,C,INB' >> acc.txt


python generate_results.py --model_type linear_regression --pred_type model_rank --scores 'INB' --pred_target 'mean_per_class_recall' >> mpcr.txt
python generate_results.py --model_type linear_regression --pred_type model_rank --scores 'C' --pred_target 'mean_per_class_recall' >> mpcr.txt
python generate_results.py --model_type linear_regression --pred_type model_rank --scores 'G' --pred_target 'mean_per_class_recall' >> mpcr.txt
python generate_results.py --model_type linear_regression --pred_type model_rank --scores 'G,C' --pred_target 'mean_per_class_recall' >> mpcr.txt
python generate_results.py --model_type linear_regression --pred_type model_rank --scores 'G,INB' --pred_target 'mean_per_class_recall' >> mpcr.txt
python generate_results.py --model_type linear_regression --pred_type model_rank --scores 'C,INB' --pred_target 'mean_per_class_recall' >> mpcr.txt
python generate_results.py --model_type linear_regression --pred_type model_rank --scores 'G,C,INB' --pred_target 'mean_per_class_recall' >> mpcr.txt