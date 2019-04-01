import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn import tree

import h2o
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.grid.grid_search import H2OGridSearch
from h2o.estimators.stackedensemble import H2OStackedEnsembleEstimator

import seaborn as sns
import matplotlib.pyplot as plt
import pylab as pl


'''

Here is the list of Philter's vars:

Vendor = a1
Geographic distribution = a2
Regional Presence = a3
Vulnerabilities list = a4
Breaches history = a5
Bug bounty programs = a6
Device Density = a7
Device type = a8
Tech Stack = a9
Jurisdiction = a10
3rd party integration = a11
Tech Support = a12
Open Access = a13
Educational research/ licensing = a14

IP address = b1
ISP = b2
ASN = b3
Services = b4
Port number = b5
Organization = b6
Traceroute hops = b7
Ping = b8

NIDS = c1
Packet header extraction (TCP/IP) = c2
MAC = c3
NetFlow records = c4
Number of packets = c5
Average packet size = c6
Number of single packets per flow = c7
HTTP headers = c8
HTTP tunneling = c9
DNS tunneling = c10
Proxy usage = c11

'''

h2o.init()

# Loading synthetic dataset
philterSyntheticData = h2o.import_file("philterSynthetic.csv")
type(philterSyntheticData)
philterSyntheticData.head()

# Check dimensions of the data
philterSyntheticData.shape
philterSyntheticData.columns
philterSyntheticData.types

# Device density
philterSyntheticData['a7'].table()
philterSyntheticData = philterSyntheticData.drop(["a8"], axis=1)
philterSyntheticData.head()
philterSyntheticData[['a3', 'a9', 'a11', 'a13', 'a14', 'a4', 'b1', 'b2', 'b4', 'b5', 'c2', 'c3', 'c5', 'c6', 'c7', 'c11' ]].as_data_frame().hist(figsize=(20, 20))
pl.show()


# Defaulters by 3rd party integration
columns = ["a7", "a11"]
default_by_gender = philterSyntheticData.group_by(by=columns).count(na="all")
print(default_by_gender.get_frame())

# Defaulters by services
columns = ["a7", "b4"]
default_by_education = philterSyntheticData.group_by(by=columns).count(na="all")
print(default_by_education.get_frame())

# Defaulters by tech stack
columns = ["a7", "a9"]
default_by_marriage = philterSyntheticData.group_by(by=columns).count(na="all")
print(default_by_marriage.get_frame())

# Convert the categorical variables into factors
philterSyntheticData['a4'] = philterSyntheticData['a4'].asfactor()
philterSyntheticData['a6'] = philterSyntheticData['a6'].asfactor()
philterSyntheticData['a2'] = philterSyntheticData['a2'].asfactor()
philterSyntheticData['a7'] = philterSyntheticData['a7'].asfactor()
philterSyntheticData['a10'] = philterSyntheticData['a10'].asfactor()
philterSyntheticData['a14'] = philterSyntheticData['a14'].asfactor()
philterSyntheticData['b1'] = philterSyntheticData['b1'].asfactor()
philterSyntheticData['b2'] = philterSyntheticData['b2'].asfactor()
philterSyntheticData['b3'] = philterSyntheticData['b3'].asfactor()
philterSyntheticData['b5'] = philterSyntheticData['b5'].asfactor()
philterSyntheticData['b8'] = philterSyntheticData['b8'].asfactor()
philterSyntheticData['c2'] = philterSyntheticData['c2'].asfactor()
philterSyntheticData['c5'] = philterSyntheticData['c5'].asfactor()
philterSyntheticData['c7'] = philterSyntheticData['c7'].asfactor()
philterSyntheticData['c8'] = philterSyntheticData['c8'].asfactor()
philterSyntheticData['c11'] = philterSyntheticData['c11'].asfactor()


philterSyntheticData.types

# Encode the binary response variable as a factor, I chose 3rd party integration
philterSyntheticData['a11'] = philterSyntheticData['a11`'].asfactor()
philterSyntheticData['a11'].levels()

# Define predictors manually
predictors = ['a1', 'a2', 'a3', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10',
              'a12', 'a13', 'a14', 'b1', 'b2', 'b3', 'b4', 'b5',
              'b6', 'b7', 'b8', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8']

# Select vulnerabilities list as target
target = 'a4'

# Split the H2O data frame into training/test sets, use 80% for training
splits = philterSyntheticData.split_frame(ratios=[0.8], seed=1)

train = splits[0]
test = splits[1]

# **GENERALIZED LINEAR MODEL (Defaut Settings)**
#
# STANDARDIZATION is enabled by default
#
# GLM with default setting
# GLM using lmbda search
# GLM using Grid search
# GLM WITH DEFAULT SETTINGS
#
# Logistic Regression (Binomial Family)
#
# H2O's GLM has the "family" argument, where the family is 'binomial' if the data is categorical 2 levels/classes or binary (Enum or Int).


GLM_default_settings = H2OGeneralizedLinearEstimator(family='binomial', model_id='GLM_default', nfolds=10,
                                                     fold_assignment="Modulo", keep_cross_validation_predictions=True)

GLM_default_settings.train(x=predictors, y=target, training_frame=train)

# ### **GLM WITH LAMBDA SEARCH**
#
# The model parameter, lambda, controls the amount of regularization in a GLM model
# Setting  lambda_search = True gives us optimal lambda value for the regularization strength.


GLM_regularized = H2OGeneralizedLinearEstimator(family='binomial', model_id='GLM', lambda_search=True, nfolds=10,
                                                fold_assignment="Modulo", keep_cross_validation_predictions=True)

GLM_regularized.train(x=predictors, y=target, training_frame=train)

# ### **GLM WITH GRID SEARCH**
#
# GLM needs to find the optimal values of the regularization parameters α and λ
# lambda: controls the amount of regularization, when set to 0 it gets disabled
#
# alpha : controls the distribution between lasso & ridge regression penalties.
#
# random grid search: H2o supports 2 types of grid search, cartesian and random. We make use of the random as the search criteria for faster computation
#
# Stopping metric: we specify the metric used for early stopping. AUTO takes log loss as default
#
# source: http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/algo-params/lambda.html
#
#


hyper_parameters = {'alpha': [0.0001, 0.001, 0.01, 0.1],
                    'lambda': [0.001, 0.01, 0.1]}

search_criteria = {'strategy': "RandomDiscrete",
                   'stopping_metric': "AUTO",
                   'stopping_rounds': 5}

GLM_grid_search = H2OGridSearch(H2OGeneralizedLinearEstimator(family='binomial', nfolds=10, fold_assignment="Modulo",
                                                              keep_cross_validation_predictions=True), hyper_parameters,
                                grid_id="GLM_grid", search_criteria=search_criteria)

GLM_grid_search.train(x=predictors, y=target, training_frame=train)


# Get the grid results, sorted by validation AUC
GLM_grid_sorted = GLM_grid_search.get_grid(sort_by='auc', decreasing=True)
GLM_grid_sorted

# Extract the best model from random grid search
Best_GLM_model_from_Grid = GLM_grid_sorted.model_ids[0]

# model performance
Best_GLM_model_from_Grid = h2o.get_model(Best_GLM_model_from_Grid)
print(Best_GLM_model_from_Grid)


# Build a RF model with default settings
RF_default_settings = H2ORandomForestEstimator(model_id='RF_D', nfolds=10, fold_assignment="Modulo",
                                               keep_cross_validation_predictions=True)

# Use train() to build the model
RF_default_settings.train(x=predictors, y=target, training_frame=train)

# Let's see the default parameters that RF model utilizes:
RF_default_settings.summary()


hyper_params = {'sample_rate': [0.7, 0.9],
                'col_sample_rate_per_tree': [0.8, 0.9],
                'max_depth': [3, 5, 9],
                'ntrees': [200, 300, 400]
                }

RF_grid_search = H2OGridSearch(
    H2ORandomForestEstimator(nfolds=10, fold_assignment="Modulo", keep_cross_validation_predictions=True,
                             stopping_metric='AUC', stopping_rounds=5), hyper_params=hyper_params,
    grid_id='RF_gridsearch')

# Use train() to start the grid search
RF_grid_search.train(x=predictors, y=target, training_frame=train)

# Sort the grid models
RF_grid_sorted = RF_grid_search.get_grid(sort_by='auc', decreasing=True)
print(RF_grid_sorted)

# Extract the best model from random grid search
Best_RF_model_from_Grid = RF_grid_sorted.model_ids[0]

# Model performance
Best_RF_model_from_Grid = h2o.get_model(Best_RF_model_from_Grid)
print(Best_RF_model_from_Grid)

GBM_default_settings = H2OGradientBoostingEstimator(model_id='GBM_default', nfolds=10, fold_assignment="Modulo",
                                                    keep_cross_validation_predictions=True)

# Use train() to build the model
GBM_default_settings.train(x=predictors, y=target, training_frame=train)

hyper_params = {'learn_rate': [0.001, 0.01, 0.1],
                'sample_rate': [0.8, 0.9],
                'col_sample_rate': [0.2, 0.5, 1],
                'max_depth': [3, 5, 9],
                'ntrees': [100, 200, 300]
                }

GBM_grid_search = H2OGridSearch(
    H2OGradientBoostingEstimator(nfolds=10, fold_assignment="Modulo", keep_cross_validation_predictions=True,
                                 stopping_metric='AUC', stopping_rounds=5),
    hyper_params=hyper_params, grid_id='GBM_Grid')

# Use train() to start the grid search
GBM_grid_search.train(x=predictors, y=target, training_frame=train)

# Sort and show the grid search results
GBM_grid_sorted = GBM_grid_search.get_grid(sort_by='auc', decreasing=True)
print(GBM_grid_sorted)

# Extract the best model from random grid search
Best_GBM_model_from_Grid = GBM_grid_sorted.model_ids[0]

Best_GBM_model_from_Grid = h2o.get_model(Best_GBM_model_from_Grid)
print(Best_GBM_model_from_Grid)

# ### STACKED ENSEMBLE


# list the best models from each grid
all_models = [Best_GLM_model_from_Grid, Best_RF_model_from_Grid, Best_GBM_model_from_Grid]

# Set up Stacked Ensemble
ensemble = H2OStackedEnsembleEstimator(model_id="ensemble", base_models=all_models,
                                       metalearner_algorithm="deeplearning")

# uses GLM as the default metalearner
ensemble.train(y=target, training_frame=train)

# ### Checking model performance of all base learners


# Checking the model performance for all GLM models built

model_perf_GLM_default = GLM_default_settings.model_performance(test)

model_perf_GLM_regularized = GLM_regularized.model_performance(test)

model_perf_Best_GLM_model_from_Grid = Best_GLM_model_from_Grid.model_performance(test)

# Checking the model performance for all RF models built


model_perf_RF_default_settings = RF_default_settings.model_performance(test)

model_perf_Best_RF_model_from_Grid = Best_RF_model_from_Grid.model_performance(test)

# Checking the model performance for all GBM models built

model_perf_GBM_default_settings = GBM_default_settings.model_performance(test)

model_perf_Best_GBM_model_from_Grid = Best_GBM_model_from_Grid.model_performance(test)

# ### Best AUC from the base learners


# Best AUC from the base learner models
best_auc = max(model_perf_GLM_default.auc(), model_perf_GLM_regularized.auc(),
               model_perf_Best_GLM_model_from_Grid.auc(), model_perf_RF_default_settings.auc(),
               model_perf_Best_RF_model_from_Grid.auc(), model_perf_GBM_default_settings.auc(),
               model_perf_Best_GBM_model_from_Grid.auc())

print("Best AUC out of all the models performed: ", format(best_auc))

# ### AUC from the Ensemble Learner


# Eval ensemble performance on the test data
Ensemble_model = ensemble.model_performance(test)
Ensemble_model = Ensemble_model.auc()

print(Ensemble_model)
