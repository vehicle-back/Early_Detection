import numpy as np
import argparse
import torch
import torch.optim as optim
import logging
import os
import pickle
import sys
import gc
import lightgbm as lgb
from lightgbm.callback import early_stopping

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime as dt
from sklearn.model_selection import GridSearchCV

from lgbm_data_loader import get_lgb_dataset
from sklearn.metrics import mean_squared_error

from utils import seed_torch, compute_train_time, save_checkpoint, create_dir, log_train_meta

def masked_rmse(pred, truth, mask):
    assert pred.shape == truth.shape == mask.shape, "Arrays must have the same shape"
    sqaure_error = np.square(pred - truth)
    masked_square_error = sqaure_error[mask==1]
    return np.sqrt(masked_square_error.mean())

def masked_mape(pred, truth, mask):
    assert pred.shape == truth.shape == mask.shape, "Arrays must have the same shape"
    percentage_errors = np.abs((pred - truth) / truth) * 100
    masked_errors = percentage_errors[mask==1]
    return masked_errors.mean()

param_grid = {
    'num_leaves': [20, 31, 50, 70],
    'learning_rate': [0.01, 0.1, 0.3],
    'n_estimators': [100, 200, 500],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0],
    'reg_lambda': [0, 0.1, 1, 10, 100], 
}

def lgbm_step_spd_train_eval(train_X, train_Y, val_X, val_Y, test_X, test_Y, i, args):
    train_inc_mask = train_Y[:, i, 2]
    val_inc_mask = val_Y[:, i, 2]
    test_inc_mask = test_Y[:, i, 2]
    if args.task =='rec':
        w_train = np.where(1-train_inc_mask, 1, 0)
        w_val = np.where(1-val_inc_mask, 1, 0)
    elif args.task =='nonrec':
        w_train = np.where(train_inc_mask, 1, 0)
        w_val = np.where(val_inc_mask, 1, 0)
    elif args.task =='no_fact':
        w_train = np.ones_like(train_inc_mask)
        w_val = np.ones_like(val_inc_mask)
    else:
        raise ValueError('spd prediction task should be no_fact/rec/nonrec')

    if not args.use_laststep_pred:
        lgb_train = lgb.Dataset(train_X, train_Y[:, i, 0], weight=w_train)
        lgb_eval = lgb.Dataset(val_X, val_Y[:, i, 0], reference=lgb_train, weight=w_val)
        params = {'objective': 'regression','metric': 'l2'}
        model = lgb.train(params, lgb_train, valid_sets=[lgb_eval], callbacks=[early_stopping(stopping_rounds=5, verbose=False)]) #, early_stopping_rounds=5, verbose_eval=False)
    
        test_pred = model.predict(test_X, num_iteration=model.best_iteration) 
        print('*****************************************************************')
        test_ground_truth = test_Y[:, i, 0]
        rmse = np.sqrt(mean_squared_error(test_ground_truth, test_pred))
        rec_rmse = masked_rmse(test_pred, test_ground_truth, 1-test_inc_mask)
        non_rec_rmse = masked_rmse(test_pred, test_ground_truth, test_inc_mask)
        print(f'if we do not seperate model, step {i} rmse is: {rmse}, rec rmse is {rec_rmse}, non_rec_rmse is {non_rec_rmse}')
        print('*****************************************************************')
    else:
        lgb_train = lgb.Dataset(train_X, train_Y[:, i, 0], weight=w_train)
        lgb_eval = lgb.Dataset(val_X, val_Y[:, i, 0], reference=lgb_train, weight=w_val)
        params = {'objective': 'regression','metric': 'l2'}
        model = lgb.train(params, lgb_train, valid_sets=[lgb_eval], callbacks=[early_stopping(stopping_rounds=5, verbose=False)]) #, early_stopping_rounds=5, verbose_eval=False)
        train_pred = model.predict(train_X, num_iteration=model.best_iteration)
        val_pred = model.predict(val_X, num_iteration=model.best_iteration)
        test_pred = model.predict(test_X, num_iteration=model.best_iteration)
        print('*****************************************************************')
        test_ground_truth = test_Y[:, i, 0]
        rmse = np.sqrt(mean_squared_error(test_ground_truth, test_pred))
        rec_rmse = masked_rmse(test_pred, test_ground_truth, 1-test_inc_mask)
        non_rec_rmse = masked_rmse(test_pred, test_ground_truth, test_inc_mask)
        print(f'if we do not seperate model, step {i} rmse is: {rmse}, rec rmse is {rec_rmse}, non_rec_rmse is {non_rec_rmse}')
        print('*****************************************************************')
        train_truth = train_Y[:, i, 0]
        random_probs = np.random.rand(*train_truth.shape)
        choice = random_probs < args.teacher_forcing_ratio
        train_pred_truth = np.where(choice, train_truth, train_pred)
        train_concat = train_pred_truth.reshape((-1, 1))
        train_X = np.concatenate((train_concat, train_X), axis=1)
        val_concat = val_pred.reshape((-1, 1))
        val_X = np.concatenate((val_concat, val_X), axis=1)
        test_concat = test_pred.reshape((-1, 1))
        test_X = np.concatenate((test_concat, test_X), axis=1)
    return model, train_X, val_X, test_X

def recall_accuracy_percision_single(pred, truth, threshold):
    truth = np.array(truth)
    mask = np.array(pred)>=threshold
    mask = mask.astype(int)
    TP_mask = mask+truth == 2
    FP_mask = mask-truth == 1
    FN_mask = truth-mask == 1
    TN_mask = truth+mask == 0
    TP_mask = TP_mask.astype(int)
    FP_mask = FP_mask.astype(int)
    FN_mask = FN_mask.astype(int)
    TN_mask = TN_mask.astype(int)
    recall = np.sum(TP_mask)/(np.sum(TP_mask)+np.sum(FN_mask))
    accuracy = (np.sum(TP_mask)+np.sum(TN_mask))/(np.sum(TP_mask)+np.sum(TN_mask)+np.sum(FP_mask)+np.sum(FN_mask))
    percision = np.sum(TP_mask)/(np.sum(TP_mask)+np.sum(FP_mask))
    F1_score = 2*((percision*recall)/(percision+recall))
    return recall, accuracy, percision, F1_score

def find_LR_threshold(pred, truth): # F1 eval
    thresholds = []
    for i in range(0, 1001):
        threshold = round(i * 0.001, 4)
        thresholds.append(threshold)
    best_F1 = 0
    best_threshold = 0
    for threshold in thresholds:
        threshold_specific_F1 = recall_accuracy_percision_single(pred, truth, threshold)[-1] # F1 score is the last value in that function
        if threshold_specific_F1>best_F1:
            best_F1 = threshold_specific_F1
            best_threshold = threshold
    return best_F1, best_threshold

def lgbm_step_inc_train_eval(train_X, train_Y, val_X, val_Y, test_X, test_Y, i, args):

    if not args.use_laststep_pred:
        lgb_train = lgb.Dataset(train_X, train_Y[:, i, 2])
        lgb_eval = lgb.Dataset(val_X, val_Y[:, i, 2], reference=lgb_train)
        params = {'objective': 'binary','metric': 'binary_logloss',#'auc'
        'is_unbalance': True}
        model = lgb.train(params, lgb_train, valid_sets=[lgb_eval], callbacks=[early_stopping(stopping_rounds=5, verbose=False)]) #, early_stopping_rounds=5, verbose_eval=False)
        val_pred = model.predict(val_X, num_iteration=model.best_iteration)
        test_pred = model.predict(test_X, num_iteration=model.best_iteration) 
        print('*****************************************************************')
        val_ground_truth = val_Y[:, i, 2]
        test_ground_truth = test_Y[:, i, 2]
        best_val_F1, best_threshold = find_LR_threshold(val_pred, val_ground_truth)
        print('hightest validation F1 is ', best_val_F1)
        test_recall, test_acc, test_per, test_F1 = recall_accuracy_percision_single(test_pred, test_ground_truth, best_threshold)
        print(f'step {i} F1 is {test_F1}, recall is: {test_recall}, percision is {test_per}, accuracy is {test_acc}')
        print('*****************************************************************')
    else:
        lgb_train = lgb.Dataset(train_X, train_Y[:, i, 2])
        lgb_eval = lgb.Dataset(val_X, val_Y[:, i, 2], reference=lgb_train)
        params = {'objective': 'binary','metric': 'binary_logloss',#'auc'
        'is_unbalance': True}
        model = lgb.train(params, lgb_train, valid_sets=[lgb_eval], callbacks=[early_stopping(stopping_rounds=5, verbose=False)]) #, early_stopping_rounds=5, verbose_eval=False)
        train_pred = model.predict(train_X, num_iteration=model.best_iteration)
        val_pred = model.predict(val_X, num_iteration=model.best_iteration)
        test_pred = model.predict(test_X, num_iteration=model.best_iteration)
        print('*****************************************************************')
        val_ground_truth = val_Y[:, i, 2]
        test_ground_truth = test_Y[:, i, 2]
        best_val_F1, best_threshold = find_LR_threshold(val_pred, val_ground_truth)
        print('hightest validation F1 is ', best_val_F1)
        test_recall, test_acc, test_per, test_F1 = recall_accuracy_percision_single(test_pred, test_ground_truth, best_threshold)
        print(f'step {i} F1 is {test_F1}, recall is: {test_recall}, percision is {test_per}, accuracy is {test_acc}')
        print('*****************************************************************')
        train_truth = train_Y[:, i, 2]
        random_probs = np.random.rand(*train_truth.shape)
        choice = random_probs < args.teacher_forcing_ratio
        train_pred_truth = np.where(choice, train_truth, train_pred)
        train_concat = train_pred_truth.reshape((-1, 1))
        train_X = np.concatenate((train_concat, train_X), axis=1)
        val_concat = val_pred.reshape((-1, 1))
        val_X = np.concatenate((val_concat, val_X), axis=1)
        test_concat = test_pred.reshape((-1, 1))
        test_X = np.concatenate((test_concat, test_X), axis=1)
    return model, train_X, val_X, test_X



def lgbm_step_spd_grid_train_eval(train_X, train_Y, val_X, val_Y, test_X, test_Y, i, args):
    train_inc_mask = train_Y[:, i, 2]
    val_inc_mask = val_Y[:, i, 2]
    test_inc_mask = test_Y[:, i, 2]
    if args.task =='rec':
        w_train = np.where(1-train_inc_mask, 1, 0)
        w_val = np.where(1-val_inc_mask, 1, 0)
    elif args.task =='nonrec':
        w_train = np.where(train_inc_mask, 1, 0)
        w_val = np.where(val_inc_mask, 1, 0)
    elif args.task =='no_fact':
        w_train = np.ones_like(train_inc_mask)
        w_val = np.ones_like(val_inc_mask)
    else:
        raise ValueError('task should be no_fact/rec/nonrec')

    if not args.use_laststep_pred:
        #lgb_train = lgb.Dataset(train_X, train_Y[:, i, 0], weight=w_train)
        #lgb_eval = lgb.Dataset(val_X, val_Y[:, i, 0], reference=lgb_train, weight=w_val)
        #params = {'objective': 'regression','metric': 'l2'}
        #model = lgb.train(params, lgb_train, valid_sets=[lgb_eval], early_stopping_rounds=5, verbose_eval=False)
        gbm = lgb.LGBMRegressor()
        grid_search = GridSearchCV(estimator=gbm, param_grid=param_grid,
                           scoring='neg_mean_squared_error',
                           # cv=3, 
                           n_jobs=-1, verbose=1)
        fit_params = {
            'sample_weight': w_train,
            'eval_set': [(val_X, val_Y[:, i, 0])],
            'eval_sample_weight': [w_val],
            # 'early_stopping_rounds': 5,
            #'verbose': False,
            'callbacks': [early_stopping(stopping_rounds=5, verbose=False)]
            }
        grid_search.fit(train_X, train_Y[:, i, 0], **fit_params)
        model = grid_search.best_estimator_

        test_pred = model.predict(test_X)  # , num_iteration=model.best_iteration
        print('*****************************************************************')
        test_ground_truth = test_Y[:, i, 0]
        rmse = np.sqrt(mean_squared_error(test_ground_truth, test_pred))
        rec_rmse = masked_rmse(test_pred, test_ground_truth, 1-test_inc_mask)
        non_rec_rmse = masked_rmse(test_pred, test_ground_truth, test_inc_mask)
        print(f'step {i} rmse is: {rmse}, rec rmse is {rec_rmse}, non_rec_rmse is {non_rec_rmse}')
        print('*****************************************************************')
    else:
        #lgb_train = lgb.Dataset(train_X, train_Y[:, i, 0], weight=w_train)
        #lgb_eval = lgb.Dataset(val_X, val_Y[:, i, 0], reference=lgb_train, weight=w_val)
        #params = {'objective': 'regression','metric': 'l2'}
        #model = lgb.train(params, lgb_train, valid_sets=[lgb_eval], early_stopping_rounds=5, verbose_eval=False)
        gbm = lgb.LGBMRegressor()
        grid_search = GridSearchCV(estimator=gbm, param_grid=param_grid,
                           scoring='neg_mean_squared_error',
                           # cv=3, 
                           n_jobs=-1, verbose=1)
        fit_params = {
            'sample_weight': w_train,
            'eval_set': [(val_X, val_Y[:, i, 0])],
            'eval_sample_weight': [w_val],
            # 'early_stopping_rounds': 5,
            #'verbose': False
            'callbacks': [early_stopping(stopping_rounds=5, verbose=False)]
            }
        grid_search.fit(train_X,  train_Y[:, i, 0], **fit_params)
        model = grid_search.best_estimator_

        train_pred = model.predict(train_X) #, num_iteration=model.best_iteration
        val_pred = model.predict(val_X)  #, num_iteration=model.best_iteration
        test_pred = model.predict(test_X) # , num_iteration=model.best_iteration
        print('*****************************************************************')
        test_ground_truth = test_Y[:, i, 0]
        rmse = np.sqrt(mean_squared_error(test_ground_truth, test_pred))
        rec_rmse = masked_rmse(test_pred, test_ground_truth, 1-test_inc_mask)
        non_rec_rmse = masked_rmse(test_pred, test_ground_truth, test_inc_mask)
        print(f'step {i} rmse is: {rmse}, rec rmse is {rec_rmse}, non_rec_rmse is {non_rec_rmse}')
        print('*****************************************************************')
        train_truth = train_Y[:, i, 0]
        random_probs = np.random.rand(*train_truth.shape)
        choice = random_probs < args.teacher_forcing_ratio
        train_pred_truth = np.where(choice, train_truth, train_pred)
        train_concat = train_pred_truth.reshape((-1, 1))
        train_X = np.concatenate((train_concat, train_X), axis=1)
        val_concat = val_pred.reshape((-1, 1))
        val_X = np.concatenate((val_concat, val_X), axis=1)
        test_concat = test_pred.reshape((-1, 1))
        test_X = np.concatenate((test_concat, test_X), axis=1)
    return model, train_X, val_X, test_X

def main(args):
    create_dir(f"{args.checkpoint_dir}/{args.county}/{args.link_id}/{args.task}" )
    train_XY, val_XY, test_XY, args.LR_pos_weight = get_lgb_dataset(args=args)
    models = []
    train_X = train_XY[0]
    val_X = val_XY[0]
    test_X = test_XY[0]
    train_Y = train_XY[1]
    val_Y = val_XY[1]
    test_Y = test_XY[1]
    for i in range(train_XY[1].shape[1]):
        if args.task =='no_fact' or args.task =='rec' or args.task =='nonrec':
            if not args.use_grid_search:
                model, train_X, val_X, test_X = lgbm_step_spd_train_eval(train_X, train_Y, val_X, val_Y, test_X, test_Y, i, args)
            else:
                model, train_X, val_X, test_X = lgbm_step_spd_grid_train_eval(train_X, train_Y, val_X, val_Y, test_X, test_Y, i, args)
  

        elif args.task =='LR':
            model, train_X, val_X, test_X = lgbm_step_inc_train_eval(train_X, train_Y, val_X, val_Y, test_X, test_Y, i, args)
   
        else:
            raise ValueError('task should be no_fact/rec/nonrec')
        model.save_model(f'{args.checkpoint_dir}/{args.county}/{args.link_id}/{args.task}/{args.exp_name}_{i}.json')
        models.append(model)
            



def gbm_create_parser():
    """
    Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--teacher_forcing_ratio', type=float, default=0.5, help='threshold of teacher forcing')
    


    # parser.add_argument('--dropout_prob_spd', type=float, default=0.35, help='dropout probability for spd task')

    parser.add_argument('--seq_len_in', type=int, default=7, help='sequence length of input')
    parser.add_argument('--seq_len_out', type=int, default=6, help='sequence length of output')
    parser.add_argument('--freq_out', type=int, default=5, help='frequency of output data') # 

    parser.add_argument('--LR_threshold', type=float, default=0.5, help='threshold of a prediction be considered as an incident') # F1 eval

    parser.add_argument('--inc_threshold', type=float, default=0.5, help='threshold of a prediction be considered as an incident')
    parser.add_argument('--LR_pos_weight', type=float, default=100, help='ratio of positive samples in incident ground truth, automatic, no need to set')


    # Assuming perfect incident status prediction at stage 1 of the 2-stage model (Traffic)
    parser.add_argument('--use_gt_inc', type=int, default=0, help='use ground truth of indicent status as input to second stage')

    # 2. Data Hyper-parameters
    parser.add_argument('--training_granularity', type=int, default=1, help='the minimum data between different slices the training data, to reduce the overfitting effect of the highly overlapped data, for example, if we want 10 min and input data interval (args.freq_out) is 5 min, then this number should be 2')
    parser.add_argument('--time_series_cv_ratio', type=float, default=1, help='should be a number between 0 and 1')
    parser.add_argument('--data_train_ratio', type=float, default=0.7, help='Ratio of training day number versus whole data')
    parser.add_argument('--data_val_ratio', type=float, default=0.2, help='Ratio of validation data versus whole data')
    parser.add_argument('--seed', type=int, default=912)

    parser.add_argument('--dim_in', type=int, default=0, help='dimension of input')
    parser.add_argument('--dim_out', type=int, default=1, help=' dimension of output i.e. number of segments (207 by default)')

    parser.add_argument('--use_spd_all', action="store_true", default=True, help='use speed data of <all vehicles> or not')
    parser.add_argument('--use_spd_truck', action="store_true", default=False, help='use speed data of <truck> or not')
    parser.add_argument('--use_spd_pv', action="store_true", default=False, help='use speed data of <personal vehicles> or not')
    parser.add_argument('--use_slowdown_spd', action="store_true", default=True, help='use slowdown speed or not')
    parser.add_argument('--use_tti', action="store_true", default=True, help='use travel time index or not')
    parser.add_argument('--use_HA', action="store_true", default=False, help='use histroical average or not')
    parser.add_argument('--use_dens', action="store_true", default=False, help='use density features or not')
    parser.add_argument('--use_weather', action="store_true", default=False, help='use weather data or not')
    parser.add_argument('--use_time', action="store_true", default=False, help='use time info or not')    
    parser.add_argument('--use_waze', action="store_true", default=True, help='use waze info or not')
    
    
    # 3. Training Hyper-parameters
    parser.add_argument('--county', type=str, default= "TSMO"
                        , help="Choose one from 'Cranberry', 'TSMO'")
    parser.add_argument('--link_id', type=str, default = '104-04540' #'110N04483' #'110+04484' #'110-04482' #'110+04483' # '110P04483' #
                        , help='the link to be analysed')
    parser.add_argument('--number_of_point_per_day', type=int, default = 186, help='for 05:30-21:00 with 5 min granularity, is 186')
    parser.add_argument('--number_of_business_day', type=int, default =260 #522
                       # 260
                        , help='number of total business day in training/val/test set')
    parser.add_argument('--upstream_range_mile', type=int, default = 2, help='the upstream range mile of the link model')
    parser.add_argument('--downstream_range_mile', type=int, default = 1, help='the downstream range mile of the link model')
    parser.add_argument('--inc_ahead_label_min', type=int, default = 0, help='incident ahead label, force model to report earlier')

    parser.add_argument('--use_laststep_pred', action="store_true", default=False, help='use last step prediction as new input')
    parser.add_argument('--use_grid_search', action="store_true", default=False, help='use last step prediction as new input')
    '''
    TASKs:
        1. "LR": call train_LR() for logistic regression, train encoder and LR_decoder
        2. "rec": call train_rec() for speed prediction, freeze encoder, train rec_decoder
        3. "nonrec": call_train_nonrec() for speed prediction, freeze encoder, train nonrec_decoder
        4. "finetune": call_finetune() for speed prediction, load checkpoint of encoder, LR_decoder, rec_decoder and nonrec_deco    der, and finetune them together
        5. "no_fact": 
            - train a model without Factorization
            - input: no new features
            - output: in 5-min frequency
        6. "naive": naive combination of three encoder-decoder modules (LR, rec, nonrec)
        7. "naive_2enc": naive combination of two-encoder-three-decoder modules (LR, rec, nonrec)
    '''
    parser.add_argument('--task', type=str, default = 'no_fact', help="Choose one from 'LR', 'rec', 'nonrec', 'finetune', 'no_fact', 'naive', 'naive_2enc") 

    parser.add_argument('--batch_size', type=int, default=64, help='Number of sequences in a batch.')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of threads to use for the DataLoader.')
    parser.add_argument('--lr', type=float, default= 0.00025, help='Learning rate (default 0.00025)') # 0.00025


    parser.add_argument('--exp_name', type=str,help='Name of the experiment')

    # 4. Directories and Checkpoint/Sample Iterations
    parser.add_argument('--data_dir', type=str, default='E:/two_stage_model')
    parser.add_argument('--log_dir', type=str, default='./logs')

    parser.add_argument('--checkpoint_dir', type=str, default='./lgbm_checkpoints')
    parser.add_argument('--checkpoint_every', type=int , default=100)

    parser.add_argument('--load_checkpoint', type=str, default='', help='Name of the checkpoint')
    parser.add_argument('--load_checkpoint_epoch', type=int, default=-1, help='Epoch of the checkpoint')

    return parser


if __name__ == '__main__':
    gc.collect()
    torch.cuda.empty_cache()  # empty cached CUDA memory

    # 1. Modify Arguments
    parser = gbm_create_parser()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    # For reproducibility
    seed_torch(args.seed)
    args.exp_name = 'lgbm'
    if args.use_grid_search:
        args.exp_name += '_gs'
    else:
        args.exp_name += '_df' # use_deafult 

    if args.use_laststep_pred:
        args.exp_name += f'_tf_{args.teacher_forcing_ratio}'

    # if args.task == 'LR':
    if args.inc_ahead_label_min > 0:
        args.exp_name += f"_inca_{str(args.inc_ahead_label_min)}"

    args.exp_name += f'_{args.upstream_range_mile}_{args.downstream_range_mile}'
    args.exp_name += f"_{str(args.use_spd_all)[0]}_{str(args.use_spd_truck)[0]}_{str(args.use_spd_pv)[0]}_{str(args.use_slowdown_spd)[0]}_{str(args.use_tti)[0]}_{str(args.use_HA)[0]}_{str(args.use_dens)[0]}_{str(args.use_weather)[0]}_{str(args.use_time)[0]}_{str(args.use_waze)[0]}" 
    args.exp_name += f'_in_{args.seq_len_in}_out_{args.seq_len_out}'
    
    main(args)