import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
import argparse
import gc
from generate_gan_label import get_data_loader
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils import shuffle

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

            # 创建标签
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

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
    # ros = RandomOverSampler(random_state=42)
    # X_resampled, y_resampled = ros.fit_resample(X_train, y_train)


    
    X_train, y_train = shuffle(X_train,  y_train, random_state=42)
    

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
        print(f'for link {args.link_id} at stage {stage_number-1}, if use GAN')
    else:
        print(f'for link {args.link_id} at stage {stage_number-1}, not use GAN')
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(classification_report(y_test, y_pred))
    
    if use_fake:
        np.save(f'{args.data_dir}/model/GAN_baseline/{args.county}/{args.link_id}_{stage_number-1}_svm_gan_ano.npy', y_pred)
        txt_file = f'{args.data_dir}/model/GAN_baseline/{args.county}/{args.link_id}_{stage_number-1}_svm_gan_ano.txt'
    else:
        np.save(f'{args.data_dir}/model/GAN_baseline/{args.county}/{args.link_id}_{stage_number-1}_svm_ano.npy', y_pred)
        txt_file = f'{args.data_dir}/model/GAN_baseline/{args.county}/{args.link_id}_{stage_number-1}_svm_ano.txt'


    with open(txt_file, "w") as file:
        if use_fake:
            file.write(f'for link {args.link_id} at stage {stage_number-1}, if use GAN\n')
        else:
            file.write(f'for link {args.link_id} at stage {stage_number-1}, not use GAN\n')
        file.write(f"Accuracy: {accuracy:.2f}\n")
        file.write(f"Precision: {precision:.2f}\n")
        file.write(f"Recall: {recall:.2f}\n")
        file.write(f"F1 Score: {f1:.2f}\n")


def INC_SVM(args, use_fake, X_train, X_test, y_train):
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X_train, y_train)

    scaler = StandardScaler()
    X_train_resampled = scaler.fit_transform(X_resampled)
    X_test = scaler.transform(X_test)

    svm = SVC(kernel='rbf', class_weight='balanced')
    svm.fit(X_train_resampled, y_resampled)

    y_pred = svm.predict(X_test)
    if use_fake:
        np.save(f'{args.data_dir}/model/GAN_baseline/{args.county}/{args.link_id}_svm_gan_inc.npy', y_pred)
    else:
        np.save(f'{args.data_dir}/model/GAN_baseline/{args.county}/{args.link_id}_svm_inc.npy', y_pred)

    



def create_parser():
    """
    Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_type', type=str, help="Choose one from 'Seq2Seq', 'Trans', 'GTrans'")
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
    
    
    # 3. Training Hyper-parameters
    parser.add_argument('--county', type=str, default="TSMO", help="Choose one from 'Cranberry', 'TSMO'")
    parser.add_argument('--link_id', type=str, default = '110+04483', help='the link to be analysed')
    parser.add_argument('--number_of_point_per_day', type=int, default = 186, help='for 05:30-21:00 with 5 min granularity, is 186')
    parser.add_argument('--number_of_business_day', type=int, default = 260, help='number of total business day in training/val/test set')
    parser.add_argument('--upstream_range_mile', type=int, default = 2, help='the upstream range mile of the link model')
    parser.add_argument('--downstream_range_mile', type=int, default = 1, help='the downstream range mile of the link model')
    parser.add_argument('--inc_ahead_label_min', type=int, default = 0, help='incident ahead label, force model to report earlier')

    parser.add_argument('--task', type=str, help="Choose one from 'LR', 'rec', 'nonrec', 'finetune', 'no_fact', 'naive', 'naive_2enc")  
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
        for stage_number in range(1, 2):
            gan_pos_train_dloader, gan_neg_train_dloader, gan_pos_val_dloader, gan_neg_val_dataloader, gan_pos_test_dloader, gan_neg_test_dataloader, train_dloader, val_dloader, test_dloader, model_input_dimension = get_data_loader(args, stage_number, inc_or_ano)
            SVM_input_dimension =  model_input_dimension*args.seq_len_in
            true_pos_train_, _ = dataloader_to_numpy(gan_pos_train_dloader)
            true_pos_train = true_pos_train_.reshape(true_pos_train_.shape[0], SVM_input_dimension)
            if use_fake:
                _, _, fake_train_pos = train(stage_number, gan_pos_train_dloader, gan_pos_val_dloader, SVM_input_dimension, nz=args.nz, epochs=args.num_epochs, dp_rate = args.dropout_prob,  lr=args.lr, beta1=args.beta1, beta2=args.beta2)
                pos_train = np.concatenate((true_pos_train, fake_train_pos), axis=0) 
            else:
                pos_train = true_pos_train

            neg_train_, _ = dataloader_to_numpy(gan_neg_train_dloader)
            neg_train = neg_train_.reshape(neg_train_.shape[0], SVM_input_dimension)
            print('pos_train', np.shape(pos_train), 'neg_train', np.shape(neg_train))



            trian_X = np.concatenate((pos_train, neg_train), axis=0)
            pos_labels =  np.ones(pos_train.shape[0])
            neg_labels = np.zeros(neg_train.shape[0])
            train_Y =  np.concatenate((pos_labels, neg_labels), axis=0)
   

            test_X_, test_Y_ = dataloader_to_numpy(test_dloader)
            test_X = test_X_.reshape(test_X_.shape[0], SVM_input_dimension)
            test_Y = test_Y_[:, stage_number, 0]
      

            ANO_SVM(args, stage_number, use_fake, trian_X, test_X, train_Y, test_Y)

    elif inc_or_ano=='incident':
        stage_number = 1
        gan_pos_train_dloader, gan_neg_train_dloader, gan_pos_val_dloader, gan_neg_val_dataloader, gan_pos_test_dloader, gan_neg_test_dataloader, train_dloader, val_dloader, test_dloader, model_input_dimension = get_data_loader(args, stage_number, inc_or_ano)
        SVM_input_dimension =  model_input_dimension*args.seq_len_in
        true_pos_train_, _ = dataloader_to_numpy(gan_pos_train_dloader)
        true_pos_train = true_pos_train_.reshape(true_pos_train_.shape[0], SVM_input_dimension)
        if use_fake:
            _, _, fake_train_pos = train(stage_number, gan_pos_train_dloader, gan_pos_val_dloader, SVM_input_dimension, nz=args.nz, epochs=args.num_epochs, dp_rate = args.dropout_prob,  lr=args.lr, beta1=args.beta1, beta2=args.beta2)
            pos_train = np.concatenate((true_pos_train, fake_train_pos), axis=0)        
        else:
            pos_train = true_pos_train

        neg_train_, _ = dataloader_to_numpy(gan_neg_train_dloader)
        neg_train = neg_train_.reshape(neg_train_.shape[0], SVM_input_dimension)

        
        trian_X = np.concatenate((pos_train, neg_train), axis=0)
        pos_labels =  np.ones(pos_train.shape[0])
        neg_labels = np.zeros(neg_train.shape[0])
        train_Y =  np.concatenate((pos_labels, neg_labels), axis=0)
   

        test_X_, test_Y_ = dataloader_to_numpy(test_dloader)
        test_X = test_X_.reshape(test_X_.shape[0], SVM_input_dimension)
        test_Y = test_Y_[:, 1, 2]

        
        # X_test, Y_test = dataloader_to_numpy(test_dloader)
        # print('X_test, Y_test', np.shape(X_test), np.shape(Y_test))

        # X_pos_train, Y_pos_train = dataloader_to_numpy(gan_pos_train_dloader)
        # print('X_pos_train, Y_pos_train', np.shape(X_pos_train), np.shape(Y_pos_train))
        INC_SVM(args, use_fake, trian_X, test_X, train_Y)
    else:
        raise ValueError("inc_or_ano should be incidnet or annomaly")
    
# train(dataloader, model_input_dimension, nz, epochs, dp_rate,  lr=0.0001, beta1=0.9, beta2=0.99)

if __name__ == '__main__':
    gc.collect()
    torch.cuda.empty_cache()  # empty cached CUDA memory

    # 1. Modify Arguments
    parser = create_parser()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    
    for link_id in ['110+04483']:#, '110-04482', '110+04424', '110-04423', '110+04522']: # 110-04482, 110+04424, 110-04423, 110+04522, 
        args.link_id = link_id
        for inc_ano in ['annomaly']:
            main(args, inc_ano, True)
            torch.cuda.empty_cache()
            args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
            main(args, inc_ano, False)
            torch.cuda.empty_cache()
            args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    
    args.county =  'Cranberry'
    args.number_of_business_day = 522

    # 104-04444, 104+04445, 104+04542, 104-04540, 104N04741
    for link_id in []:#'104-04444', '104+04445', '104+04542', '104-04540', '104N04741']:
        args.link_id = link_id
        for inc_ano in ['annomaly']: # 
            main(args, inc_ano, True)
            torch.cuda.empty_cache()
            args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
            main(args, inc_ano, False)
            torch.cuda.empty_cache()
            args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')