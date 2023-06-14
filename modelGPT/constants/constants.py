from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor


GROUND_TRUTH_CSV = './modelGPT/eval_table_features.csv'

ALL_FEATURES = ['text-f1', 'text-acc1', 'intraclass_sim','inter_close','intra_close', 'superclass_metric', 'IN-score']
DATASET_TO_REMOVE = []
MODEL_TO_REMOVE = [ ]

MODELS = {
    'linear_regression': LinearRegression(),
    'lasso': Lasso(),
    'gbr': GradientBoostingRegressor(),
    'svr': SVR(),
    'nn': MLPRegressor(),
}

PRED_TARGET = 'acc1'#'mean_per_class_recall'
NUM_RANK = 5
MODEL_NAME_COL = 'model_fullname'
DATASET_COL = 'dataset'

PARAMETERS = {
    'linear_regression': None,
    'lasso': {
        'alpha': [0.01, 0.1, 1.0],
    }, 
    'gbr': {
        'loss': ['absolute_error', 'huber', 'quantile'],
        'alpha': [0.01, 0.1 ,1.0],
    },
    'svr': {
        'kernel': ['poly', 'rbf', 'sigmoid'],
        'degree': [2, 4, 5],
    },
    'nn': {
        'hidden_layer_sizes': [(100,), (10,), (5,)],
        'learning_rate_init': [0.001, 0.01, 0.1],
    }
}

FEATURE_ORDER_DICT = {
    'IN-score': 0,
    'text-f1': 1,
    'text-acc1': 2,
    'superclass_metric': 3,
    'inter_close': 4,
    'intra_close': 5,
    'intraclass_sim': 6
    }

FEATURE_ORDER = [
    'IN-score',
    'text-f1',
    'text-acc1',
    'superclass_metric',
    'inter_close',
    'intra_close',
    'intraclass_sim'
    ]