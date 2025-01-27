import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
import argparse
import gc
import joblib

from lgbm_data_loader import get_lgb_dataset
from lgbm_spd_data_loader import get_spd_lgb_dataset

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.impute import SimpleImputer
from generate_gan_label import get_data_loader

from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from imblearn.under_sampling import RandomUnderSampler
from lightgbm import LGBMClassifier
import numpy as np
import os
import lightgbm as lgb
from lightgbm.callback import early_stopping

from sklearn.metrics import mean_squared_error

class Generator(nn.Module):
    def __init__(self, nz, model_input_dimension):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(nz,1024),
            nn.ReLU(True),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, model_input_dimension),
            nn.Tanh()
        )
    
    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self, model_input_dimension, dp_rate):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(model_input_dimension, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dp_rate),
            nn.Linear(512, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(64),
            nn.Dropout(dp_rate),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
    
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0)

def train(stage_number, train_dataloader, val_loader, model_input_dimension, nz, epochs, dp_rate,  lr=0.0001, beta1=0.9, beta2=0.99):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    num_train_samples = len(train_dataloader.dataset)
    fixed_noise = torch.randn(num_train_samples, nz, device=device)
    # print(f'Number of samples in the training data loader: {num_train_samples}')

    # 初始化生成器和判别器
    G = Generator(nz, model_input_dimension).to(device) #1
    D = Discriminator(model_input_dimension, dp_rate).to(device) #1
    G.apply(weights_init) #1 
    D.apply(weights_init) #1

    # 初始化权重
    G.apply(weights_init) #1 
    D.apply(weights_init) #1 

    # 定义损失函数和优化器
    optimizerD = optim.Adam(D.parameters(), lr=lr, betas=(beta1, beta2)) #1
    optimizerG = optim.Adam(G.parameters(), lr=lr, betas=(beta1, beta2)) #1

    best_g_loss = float('inf')
    best_generator = None

    for epoch in range(epochs):

        G.train()
        D.train()

        for i, data in enumerate(train_dataloader, 0):
        # for i, data in enumerate(dataloader, 0):
        #for i, (real_data, _) in enumerate(dataloader):
            real_data = data[0].to(device).view(-1, model_input_dimension) # real_data.to(device)
            batch_size = real_data.size(0)

            # 更新判别器
            D.zero_grad() #1
            outputs_real = D(real_data)
            d_loss_real = torch.mean(torch.log(outputs_real))
            
            noise = torch.randn(batch_size, nz).to(device)
            fake_data = G(noise)
            outputs_fake = D(fake_data.detach())
            d_loss_fake = torch.mean(torch.log(1 - outputs_fake))
            
            d_loss = -(d_loss_real + d_loss_fake)
            d_loss.backward()
            optimizerD.step()

            # 更新生成器
            G.zero_grad()
            outputs = D(fake_data)
            # g_loss = torch.mean(torch.log(1- outputs))
            g_loss = -torch.mean(torch.log(outputs))
            g_loss.backward()
            optimizerG.step()

            if i % 10 == 0:
                # print(f'Epoch [{epoch+1}/{epochs}] Loss D: {d_loss.item()}, Loss G: {g_loss.item()}')
            
                G.eval()
        
        G.eval()
        D.eval()
        val_g_loss = 0
        with torch.no_grad():
            for val_real_data, _ in val_loader:
                val_real_data = val_real_data.to(device)
                batch_size = val_real_data.size(0)

                val_noise = torch.randn(batch_size, nz).to(device)
                val_fake_data = G(val_noise)
                val_outputs = D(val_fake_data)
                val_g_loss += -torch.mean(torch.log(val_outputs)) #torch.mean(torch.log(1 - val_outputs))

            val_g_loss /= len(val_loader)
            # print(f'Validation Loss G: {val_g_loss.item()}')

            # 保存最佳生成器模型
            if val_g_loss < best_g_loss:
                best_g_loss = val_g_loss
                best_generator = G.state_dict()
                print(f'best model saved at Epoch {epoch} Validation Loss G: {val_g_loss.item()}')
                torch.save(best_generator, f'{args.data_dir}/model/GAN_baseline/{args.county}/{args.link_id}_{args.upstream_range_mile}_{args.downstream_range_mile}_{args.seq_len_in}_{args.seq_len_out}_{str(args.use_spd_all)[0]}_{str(args.use_spd_truck)[0]}_{str(args.use_spd_pv)[0]}_{str(args.use_slowdown_spd)[0]}_{str(args.use_tti)[0]}_{str(args.use_HA)[0]}_{str(args.use_dens)[0]}_{str(args.use_weather)[0]}_{str(args.use_time)[0]}_{str(args.use_waze)[0]}_{args.inc_ahead_label_min}_best_G_{stage_number}.pth')
                fake_pos = G(fixed_noise).detach().cpu().numpy()


                

    return G, D, fake_pos

def ANO_SVM(args, stage_number, use_fake, X_train, X_test, y_train, y_test):
    ros = RandomUnderSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X_train, y_train)


    
    X_train, y_train = shuffle(X_resampled, y_resampled, random_state=42)
    

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
    print('*'*60)
    if use_fake:
        print(f'for link {args.link_id} at stage {stage_number}, if use GAN')
    else:
        print(f'for link {args.link_id} at stage {stage_number}, not use GAN')
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(classification_report(y_test, y_pred))
    
    if use_fake:
        np.save(f'{args.data_dir}/model/GAN_baseline/{args.county}/{args.link_id}/{stage_number}_svm_gan_ano.npy', y_pred)
        model_path = f'{args.data_dir}/model/GAN_baseline/{args.county}/{args.link_id}/{stage_number}_svm_gan_ano_model.pkl'
        txt_file = f'{args.data_dir}/model/GAN_baseline/{args.county}/{args.link_id}/{stage_number}_svm_gan_ano.txt'
    else:
        np.save(f'{args.data_dir}/model/GAN_baseline/{args.county}/{args.link_id}/{stage_number}_svm_ano.npy', y_pred)
        model_path = f'{args.data_dir}/model/GAN_baseline/{args.county}/{args.link_id}/{stage_number}_svm_ano_model.pkl'
        txt_file = f'{args.data_dir}/model/GAN_baseline/{args.county}/{args.link_id}/{stage_number}_svm_ano.txt'

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(svm, model_path)

    with open(txt_file, "w") as file:
        if use_fake:
            file.write(f'for link {args.link_id} at stage {stage_number}, if use GAN\n')
        else:
            file.write(f'for link {args.link_id} at stage {stage_number}, not use GAN\n')
        file.write(f"Accuracy: {accuracy:.2f}\n")
        file.write(f"Precision: {precision:.2f}\n")
        file.write(f"Recall: {recall:.2f}\n")
        file.write(f"F1 Score: {f1:.2f}\n")



def ANO_LightGBM(args, stage_number, use_fake, X_train, X_test, y_train, y_test):
    # Resampling the dataset to handle imbalance
    # ros = RandomUnderSampler(random_state=42)
    # X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
    
    # Shuffling the data
    # X_train, y_train = shuffle(X_resampled, y_resampled, random_state=42)
    
    # Scaling the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Initializing and training the LightGBM classifier
    lgbm = LGBMClassifier(class_weight='balanced', random_state=42)
    lgbm.fit(X_train, y_train)

    # Making predictions
    y_pred = lgbm.predict(X_test)
    
    # Calculating performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Printing results
    print('*' * 60)
    if use_fake:
        print(f'for link {args.link_id} at stage {stage_number}, if use GAN')
    else:
        print(f'for link {args.link_id} at stage {stage_number}, not use GAN')
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(classification_report(y_test, y_pred))
    
    # Saving the predictions
    if use_fake:
        np.save(f'{args.data_dir}/model/GAN_baseline/{args.county}/{args.link_id}/{stage_number}_lgbm_gan_ano.npy', y_pred)
        model_path = f'{args.data_dir}/model/GAN_baseline/{args.county}/{args.link_id}/{stage_number}_lgbm_gan_ano_model.txt'
        txt_file = f'{args.data_dir}/model/GAN_baseline/{args.county}/{args.link_id}/{stage_number}_lgbm_gan_ano.txt'
    else:
        np.save(f'{args.data_dir}/model/GAN_baseline/{args.county}/{args.link_id}/{stage_number}_lgbm_ano.npy', y_pred)
        model_path = f'{args.data_dir}/model/GAN_baseline/{args.county}/{args.link_id}/{stage_number}_lgbm_ano_model.txt'
        txt_file = f'{args.data_dir}/model/GAN_baseline/{args.county}/{args.link_id}/{stage_number}_lgbm_ano.txt'

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    lgbm.booster_.save_model(model_path)
    
    # Writing results to a text file
    with open(txt_file, "w") as file:
        if use_fake:
            file.write(f'for link {args.link_id} at stage {stage_number}, if use GAN\n')
        else:
            file.write(f'for link {args.link_id} at stage {stage_number}, not use GAN\n')
        file.write(f"Accuracy: {accuracy:.2f}\n")
        file.write(f"Precision: {precision:.2f}\n")
        file.write(f"Recall: {recall:.2f}\n")
        file.write(f"F1 Score: {f1:.2f}\n")

def INC_SVM(args, use_fake, X_train, X_test, y_train):
    ros = RandomUnderSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X_train, y_train)

    X_train, y_train = shuffle(X_resampled, y_resampled, random_state=42)
    

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    svm = SVC(kernel='rbf', class_weight='balanced')
    svm.fit(X_train, y_train)

    y_pred = svm.predict(X_test)

    if use_fake:
        np.save(f'{args.data_dir}/model/GAN_baseline/{args.county}/{args.link_id}/svm_gan_inc.npy', y_pred)
        model_path = f'{args.data_dir}/model/GAN_baseline/{args.county}/{args.link_id}/svm_gan_inc_model.pkl'
    else:
        np.save(f'{args.data_dir}/model/GAN_baseline/{args.county}/{args.link_id}/svm_inc.npy', y_pred)
        model_path = f'{args.data_dir}/model/GAN_baseline/{args.county}/{args.link_id}/svm_inc_model.pkl'
    
    # Create directories if they do not exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Save the trained SVM model using joblib
    joblib.dump(svm, model_path)




def INC_LightGBM(args, use_fake, X_train, X_test, y_train):
    # Undersample the data to handle class imbalance
    # ros = RandomUnderSampler(random_state=42)
    # X_resampled, y_resampled = ros.fit_resample(X_train, y_train)

    # Shuffle the resampled data
    # X_train, y_train = shuffle(X_resampled, y_resampled, random_state=42)

    # Scale the training and test data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # LightGBM classifier
    lgbm = LGBMClassifier(class_weight='balanced', random_state=42)
    lgbm.fit(X_train, y_train)

    # Predict on test data
    y_pred = lgbm.predict(X_test)

    # Save the predictions
    if use_fake:
        np.save(f'{args.data_dir}/model/GAN_baseline/{args.county}/{args.link_id}/lgbm_gan_inc.npy', y_pred)
        model_path = f'{args.data_dir}/model/GAN_baseline/{args.county}/{args.link_id}/lgbm_gan_inc_model.txt'
    else:
        np.save(f'{args.data_dir}/model/GAN_baseline/{args.county}/{args.link_id}/lgbm_inc.npy', y_pred)
        model_path = f'{args.data_dir}/model/GAN_baseline/{args.county}/{args.link_id}/lgbm_inc_model.txt'

    # Create directories if they do not exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Save the trained LightGBM model
    lgbm.booster_.save_model(model_path)

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

def lgbm_step_spd_train_eval(train_X, train_Y, val_X, val_Y, test_X, test_Y, i, args):
    train_inc_mask = train_Y[:, i, 3]
    val_inc_mask = val_Y[:, i, 3]
    test_inc_mask = test_Y[:, i, 3]
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
    
def filter_1(X, Y):
    # 创建布尔掩码，标识 Y 中所有等于 1 的行
    mask = (Y == 1).flatten()
    
    # 使用掩码筛选 X 中对应的行
    filtered_X = X[mask]
    
    return filtered_X

def filter_0(X, Y):
    # 创建布尔掩码，标识 Y 中所有等于 1 的行
    mask = (Y == 0).flatten()
    
    # 使用掩码筛选 X 中对应的行
    filtered_X = X[mask]
    
    return filtered_X


def create_parser():
    """
    Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_type', type=str,  default = 'SVM', help="SVM or LGBM'")
    parser.add_argument('--task', type=str, default = 'incident', help="annomaly or incident")
    parser.add_argument('--use_GAN', action="store_true", default=True, help='use GAN to generate fake data or not')
    
    parser.add_argument('--dropout_prob', type=float, default = 0.3, help='dropout probability')
    

    # 1. Model hyper-parameters
    parser.add_argument('--seq_len_in', type=int, default=7, help='sequence length of input')
    parser.add_argument('--seq_len_out', type=int, default=6, help='sequence length of output')
    parser.add_argument('--freq_out', type=int, default=5, help='frequency of output data')


    parser.add_argument('--inc_threshold', type=float, default=0.5, help='threshold of a prediction be considered as an incident')
    parser.add_argument('--LR_pos_weight', type=float, default=100, help='ratio of positive samples in incident ground truth, automatic, no need to set')
    parser.add_argument('--nonrec_spd_weight', type=float, default=15, help = 'the weight added to non recurrent speed when computing finetune mse')  
    parser.add_argument('--use_expectation', action="store_true", help='use expectation of speed prediction as model output')  

    # Assuming perfect incident status prediction at stage 1 of the 2-stage model (Traffic)
    parser.add_argument('--use_gt_inc', type=int, default=0, help='use ground truth of indicent status as input to second stage')

    # 2. Data Hyper-parameters
    parser.add_argument('--training_granularity', type=int, default=1, help='the minimum data between different slices the training data, '
        'to reduce the overfitting effect of the highly overlapped data, for example, if we want 10 min and input data interval (args.freq_out) is 5 min, then this number should be 2')
    parser.add_argument('--time_series_cv_ratio', type=float, default=1, help='should be a number between 0 and 1')
    parser.add_argument('--data_train_ratio', type=float, default=0.7, help='Ratio of training day number versus whole data')
    parser.add_argument('--data_val_ratio', type=float, default=0.2, help='Ratio of validation data versus whole data')
    parser.add_argument('--seed', type=int, default=3407)

    parser.add_argument('--dim_in', type=int, default=0, help='dimension of input')
    parser.add_argument('--dim_out', type=int, default=1, help=' dimension of output i.e. number of segments (207 by default)')
    parser.add_argument('--num_node', type=int, default=207, help='number of nodes (207 by default) as used in GTrans')

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
    parser.add_argument('--use_laststep_pred', action="store_true", default=False, help='use last step prediction as new input')
    
    
    # 3. Training Hyper-parameters
    parser.add_argument('--county', type=str, default="TSMO", help="Choose one from 'Cranberry', 'TSMO'")
    parser.add_argument('--link_id', type=str, default = '110+04483', help='the link to be analysed')
    parser.add_argument('--number_of_point_per_day', type=int, default = 186, help='for 05:30-21:00 with 5 min granularity, is 186')
    parser.add_argument('--number_of_business_day', type=int, default = 260, help='number of total business day in training/val/test set')
    parser.add_argument('--upstream_range_mile', type=int, default = 2, help='the upstream range mile of the link model')
    parser.add_argument('--downstream_range_mile', type=int, default = 1, help='the downstream range mile of the link model')
    parser.add_argument('--inc_ahead_label_min', type=int, default = 0, help='incident ahead label, force model to report earlier')

     
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=64, help='Number of sequences in a batch.')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of threads to use for the DataLoader.')
    parser.add_argument('--lr', type=float, default= 0.000025, help='Learning rate (default 0.0001)') # 0.00025

    parser.add_argument('--nz', type=int, default= 100, help='nz value')
    parser.add_argument('--beta1', type=float, default= 0.9, help='beta1')
    parser.add_argument('--beta2', type=float, default= 0.99, help='beta2')

    parser.add_argument('--exp_name', type=str,help='Name of the experiment')

    # 4. Directories and Checkpoint/Sample Iterations
    parser.add_argument('--data_dir', type=str, default='E:/two_stage_model')
    parser.add_argument('--log_dir', type=str, default='./logs')

    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--checkpoint_every', type=int , default=100)

    parser.add_argument('--load_checkpoint', type=str, default='', help='Name of the checkpoint')
    parser.add_argument('--load_checkpoint_epoch', type=int, default=-1, help='Epoch of the checkpoint')

    return parser

def dataloader_to_numpy(dataloader):
    # 初始化空列表用于存储数据和标签
    X_list = []
    y_list = []

    # 遍历DataLoader
    for data, target in dataloader:
        # 将数据和标签转换为NumPy数组，并添加到列表中
        X_list.append(data.numpy())
        y_list.append(target.numpy())

    # 将列表中的数据合并成单个NumPy数组
    X_array = np.concatenate(X_list, axis=0)
    y_array = np.concatenate(y_list, axis=0)

    return X_array, y_array

def main(args, inc_or_ano, use_fake):
    

    if inc_or_ano=='annomaly':
        

        for stage_number in range(0, 6):
            train_XY, _, test_XY = get_lgb_dataset(args, stage_number, inc_or_ano)
            gan_pos_train_dloader, _, gan_pos_val_dloader, _, _, _, _, _, _, _ = get_data_loader(args, stage_number, inc_or_ano)
            train_X = train_XY[0]
            test_X = test_XY[0]
            train_Y = train_XY[1]
            test_Y = test_XY[1]
            imputer = SimpleImputer(strategy='constant', fill_value=0)
            train_X_all = imputer.fit_transform(train_X)
        
            test_X_all = imputer.fit_transform(test_X)
            train_annomaly = train_Y[:, stage_number, 4]
            test_annomaly = test_Y[:, stage_number, 4]
            SVM_input_dimension = train_X[0].shape[0]
        

            if use_fake:
                _, _, fake_train_pos = train(stage_number+1, gan_pos_train_dloader, gan_pos_val_dloader, SVM_input_dimension, nz=args.nz, epochs=args.num_epochs, dp_rate = args.dropout_prob,  lr=args.lr, beta1=args.beta1, beta2=args.beta2)
                train_X_all = np.concatenate((train_X_all, fake_train_pos), axis=0)
                fake_labels =  np.ones(fake_train_pos.shape[0])
                train_annomaly = np.concatenate((train_annomaly, fake_labels), axis=0)
            
            if args.model_type == 'SVM':
                ANO_SVM(args, stage_number, use_fake, train_X_all, test_X_all, train_annomaly, test_annomaly)
            else:
                ANO_LightGBM(args, stage_number, use_fake, train_X_all, test_X_all, train_annomaly, test_annomaly)

    elif inc_or_ano=='incident':
        stage_number = 1
        train_XY, _, test_XY = get_lgb_dataset(args, stage_number, inc_or_ano)
        gan_pos_train_dloader, _, gan_pos_val_dloader, _, _, _, _, _, _, _ = get_data_loader(args, stage_number, inc_or_ano)
        train_X = train_XY[0]
        test_X = test_XY[0]
        train_Y = train_XY[1]
        test_Y = test_XY[1]
        imputer = SimpleImputer(strategy='constant', fill_value=0)
        train_X_all = imputer.fit_transform(train_X)
        test_X_all = imputer.fit_transform(test_X)
        train_inc = train_Y[:, stage_number, 2]
        # test_inc = test_Y[:, stage_number, 2]
        SVM_input_dimension = train_X[0].shape[0]
        if use_fake:
            _, _, fake_train_pos = train(stage_number+1, gan_pos_train_dloader, gan_pos_val_dloader, SVM_input_dimension, nz=args.nz, epochs=args.num_epochs, dp_rate = args.dropout_prob,  lr=args.lr, beta1=args.beta1, beta2=args.beta2)
            train_X_all = np.concatenate((train_X_all, fake_train_pos), axis=0)
            fake_labels =  np.ones(fake_train_pos.shape[0])
            train_inc = np.concatenate((train_inc, fake_labels), axis=0)
 
        


        
        # X_test, Y_test = dataloader_to_numpy(test_dloader)
        # print('X_test, Y_test', np.shape(X_test), np.shape(Y_test))

        # X_pos_train, Y_pos_train = dataloader_to_numpy(gan_pos_train_dloader)
        # print('X_pos_train, Y_pos_train', np.shape(X_pos_train), np.shape(Y_pos_train))
        if args.model_type == 'SVM':
            INC_SVM(args, use_fake, train_X_all, test_X_all, train_inc)
        else:
            INC_LightGBM(args, use_fake, train_X_all, test_X_all, train_inc)
    elif inc_or_ano=='no_fact':
        train_XY, val_XY, test_XY, _ = get_spd_lgb_dataset(args=args)
        train_X = train_XY[0]
        val_X = val_XY[0]
        test_X = test_XY[0]
        train_Y = train_XY[1]
        val_Y = val_XY[1]
        test_Y = test_XY[1]
        for i in range(0, 6):
            model, train_X, val_X, test_X = lgbm_step_spd_train_eval(train_X, train_Y, val_X, val_Y, test_X, test_Y, i, args)
            model.save_model(f'{args.data_dir}/model/GAN_baseline/{args.county}/{args.link_id}/LightGBM_spd_{i}.json')
        
    else:
        raise ValueError("inc_or_ano should be incidnet or annomaly")
        for stage_number in range(0, 1):
            train_XY, _, test_XY = get_lgb_dataset(args, stage_number, 'annomaly')
            gan_pos_train_dloader, _, gan_pos_val_dloader, _, _, _, _, _, _, _ = get_data_loader(args, stage_number, 'annomaly')
            train_X = train_XY[0]
            test_X = test_XY[0]
            train_Y = train_XY[1]
            test_Y = test_XY[1]
            imputer = SimpleImputer(strategy='constant', fill_value=0)
            train_X_all = imputer.fit_transform(train_X)
            test_X_all = imputer.fit_transform(test_X)
            train_annomaly = train_Y[:, stage_number, 4]
            test_annomaly = test_Y[:, stage_number, 4]
            SVM_input_dimension = train_X[0].shape[0]
            print('SVM_input_dimension', SVM_input_dimension)

            if use_fake:
                _, _, fake_train_pos = train(stage_number+1, gan_pos_train_dloader, gan_pos_val_dloader, SVM_input_dimension, nz=args.nz, epochs=args.num_epochs, dp_rate = args.dropout_prob,  lr=args.lr, beta1=args.beta1, beta2=args.beta2)
                train_X_all = np.concatenate((train_X_all, fake_train_pos), axis=0)
                fake_labels =  np.ones(fake_train_pos.shape[0])
                train_annomaly = np.concatenate((train_annomaly, fake_labels), axis=0)
 
            ANO_SVM(args, stage_number, use_fake, train_X_all, test_X_all, train_annomaly, test_annomaly)            
    
# train(dataloader, model_input_dimension, nz, epochs, dp_rate,  lr=0.0001, beta1=0.9, beta2=0.99)

if __name__ == '__main__':
    gc.collect()
    torch.cuda.empty_cache()  # empty cached CUDA memory

    # 1. Modify Arguments
    parser = create_parser()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    main(args, args.task, args.use_GAN)
    torch.cuda.empty_cache()
           