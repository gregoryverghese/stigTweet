import numpy as np

rflearning_rate = [1, 0.5, 0.25, 0.1, 0.05, 0.01]
rfn_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
rfmax_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
rfmin_samples_split = [2, 5, 10]
rfmin_samples_leaf = [1, 2, 4]
rfbootstrap = [True, False]

gblearning_rate = [1, 0.5, 0.25, 0.1, 0.05, 0.01]
gbn_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
gbmax_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
gbmin_samples_split = [2, 5, 10]
gbmin_samples_leaf = [1, 2, 4]
gbbootstrap = [True, False]

c = [0.001, 0.01, 0.1, 1, 10]
gamma = [0.001, 0.01, 0.1, 1, 'auto']
dfs = ['ovo','ovr']
shrink = [True, False]
kernel = ['linear', 'poly', 'sigmoid']


rfRandomGrid = {'n_estimators': rfn_estimators,
               'max_depth': rfmax_depth,
               'min_samples_split': rfmin_samples_split,
               'min_samples_leaf': rfmin_samples_leaf,
               'bootstrap': rfbootstrap}

gbRandomGrid = {'learning_rate': gblearning_rate,
               'n_estimators': gbn_estimators,
               'max_depth': gbmax_depth,
               'min_samples_split': gbmin_samples_split,
               'min_samples_leaf': gbmin_samples_leaf}

knnRandomGrid = {'k_neighbors': [1, 3, 5, 10, 15, 30],
                'weights':['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']}

svmRandomGrid = {'kernel': kernel, 'C': c,'gamma': gamma,'decision_function_shape':dfs,'shrinking': shrink}


wordParameters = [{'bootstrap': False,
  'max_depth': 90,
  'min_samples_leaf': 4,
  'min_samples_split': 10,
  'n_estimators': 800},
 {'learning_rate': 0.5,
  'max_depth': 80,
  'min_samples_leaf': 4,
  'min_samples_split': 2,
  'n_estimators': 1200},
 {'C': 10,
  'decision_function_shape': 'ovr',
  'gamma': 1,
  'kernel': 'linear',
  'shrinking': False},
 {}]

charParameters = [{'bootstrap': False,
  'max_depth': 70,
  'min_samples_leaf': 1,
  'min_samples_split': 2,
  'n_estimators': 1800},
 {'learning_rate': 0.05,
  'max_depth': 80,
  'min_samples_leaf': 4,
  'min_samples_split': 2,
  'n_estimators': 1800},
 {'C': 10,
  'decision_function_shape': 'ovr',
  'gamma': 1,
  'kernel': 'linear',
  'shrinking': False},
 {}]
