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
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from lgbm_data_loader import get_lgb_dataset
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from imblearn.over_sampling import RandomOverSampler


from utils import seed_torch, compute_train_time, save_checkpoint, create_dir, log_train_meta


def SVM(X_train, X_test, y_train, y_test):
    # ros = RandomOverSampler(random_state=42)
    # X_resampled, y_resampled = ros.fit_resample(X_train, y_train)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    svm = SVC(kernel='rbf', class_weight='balanced')
    svm.fit(X_train, y_train)

    y_pred = svm.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(classification_report(y_test, y_pred))


def main(args):
    # create_dir(f"{args.checkpoint_dir}/{args.county}/{args.link_id}/{args.task}" )

    train_XY, _, test_XY, args.LR_pos_weight = get_lgb_dataset(args=args)
    train_X = train_XY[0]
    test_X = test_XY[0]
    train_Y = train_XY[1]
    test_Y = test_XY[1]
    imputer = SimpleImputer(strategy='constant', fill_value=0)
    train_X = imputer.fit_transform(train_X)
    test_X = imputer.fit_transform(test_X)
    train_Y =  (np.sum(train_Y[:, :, 2], axis=1) > 0.5).astype(int)
    test_Y =  (np.sum(test_Y[:, :, 2], axis=1) > 0.5).astype(int)
    
    SVM(train_X, test_X, train_Y, test_Y)


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
    parser.add_argument('--use_waze', action="store_true", default=False, help='use waze info or not')
    
    
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
    if args.county == 'Cranberry':
        args.number_of_business_day = 520

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