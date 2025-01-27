import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
import argparse
import gc
from generate_gan_label import get_data_loader


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

def evaluate_discriminator(netD, pos_dataloader, neg_dataloader, model_input_dimension, device):
    netD.eval()  
    total, correct = 0, 0
    with torch.no_grad():
        for data in pos_dataloader:
            real_data = data[0].to(device).view(-1, model_input_dimension)
            output = netD(real_data).view(-1)
            predicted = (output > 0.5).float()
            total += real_data.size(0)
            correct += (predicted == 1).sum().item()
        # print(correct, total)
        for data in neg_dataloader:
            real_data = data[0].to(device).view(-1, model_input_dimension)
            output = netD(real_data).view(-1)
            predicted = (output < 0.5).float()
            total += real_data.size(0)
            correct += (predicted == 1).sum().item()
    netD.train() 
    return correct/total


def evaluate_test(netD, pos_dataloader, neg_dataloader, model_input_dimension, device):
    netD.eval()  
    total, correct = 0, 0
    pos_total, pos_correct = 0, 0
    pos_incorrect = 0
    with torch.no_grad():
        for data in pos_dataloader:
            real_data = data[0].to(device).view(-1, model_input_dimension)
            output = netD(real_data).view(-1)
            predicted = (output > 0.5).float()
            total += real_data.size(0)
            correct += (predicted == 1).sum().item()
            pos_total += real_data.size(0)
            pos_correct += (predicted == 1).sum().item()
        # print(correct, total)
        for data in neg_dataloader:
            real_data = data[0].to(device).view(-1, model_input_dimension)
            output = netD(real_data).view(-1)
            predicted = (output > 0.5).float()
            total += real_data.size(0)
            pos_incorrect += (predicted == 1).sum().item()
            # neg_total += real_data.size(0)
    DR = pos_correct/pos_total
    FAR = pos_incorrect/(pos_incorrect+pos_correct)
    netD.train() 
    return DR, FAR

def train(train_loader, pos_val_loader, neg_val_loader, pos_test_loader, neg_test_loader, model_input_dimension, dp_rate, num_epochs=50, batch_size=64, nz=100, lr=0.0001, beta1=0.9, beta2=0.99):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    netG = Generator(nz, model_input_dimension).to(device)
    netD = Discriminator(model_input_dimension, dp_rate).to(device)
    netG.apply(weights_init)
    netD.apply(weights_init)

    criterion_D = nn.BCELoss()
    criterion_G = nn.MSELoss()

    fixed_noise = torch.randn(batch_size, nz, device=device)
    real_label = 1.
    fake_label = 0.

    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, beta2))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, beta2))

    best_validation_accuracy = 0
  
    for epoch in range(num_epochs):
        for i, data in enumerate(train_loader, 0):
            # train discriminator
            netD.zero_grad()
            real_data = data[0].to(device).view(-1, model_input_dimension)  # 假设dataloader返回(batch_size, 1)的张量
            b_size = real_data.size(0)
            label = torch.full((b_size,), real_label, device=device)
            output = netD(real_data).view(-1)
            errD_real = criterion_D(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            noise = torch.randn(b_size, nz, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach()).view(-1)
            errD_fake = criterion_D(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            netG.zero_grad()
            label.fill_(real_label)
            output = netD(fake).view(-1)
            errG = criterion_G(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            if i % 10 == 0:
                print(f'[{epoch}/{num_epochs}][{i}/{len(train_loader)}] '
                      f'Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} '
                      f'D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}')
                
        validation_accuracy = evaluate_discriminator(netD, pos_val_loader, neg_val_loader, model_input_dimension, device)
        print(f'Epoch {epoch}, Validation Accuracy: {validation_accuracy:.4f}')
        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            test_DR, test_FAR = evaluate_test(netD, pos_test_loader, neg_test_loader, model_input_dimension, device)
            print('*'*60)
            print(f'best mdoel saved')
            print(f'test DR {test_DR}, test FAR {test_FAR}')
            print('*'*60)
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu().numpy()
                np.save(f'{args.data_dir}/model/link_in_out/GAN_generated/{args.county}/test_{args.link_id}_{args.upstream_range_mile}_{args.downstream_range_mile}_{args.seq_len_in}_{args.seq_len_out}_{str(args.use_spd_all)[0]}_{str(args.use_spd_truck)[0]}_{str(args.use_spd_pv)[0]}_{str(args.use_slowdown_spd)[0]}_{str(args.use_tti)[0]}_{str(args.use_HA)[0]}_{str(args.use_dens)[0]}_{str(args.use_weather)[0]}_{str(args.use_time)[0]}_{str(args.use_waze)[0]}_{args.inc_ahead_label_min}_gan_fake_x.npy', 
                    fake)

    print("GAN training is done")


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
    parser.add_argument('--num_epochs', type=int, default=300, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=64, help='Number of sequences in a batch.')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of threads to use for the DataLoader.')
    parser.add_argument('--lr', type=float, default= 0.00025, help='Learning rate (default 0.0001)') # 0.00025

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


def main(args):
    gan_train_dloader, gan_pos_val_dloader, gan_neg_val_dataloader, gan_pos_test_dloader, gan_neg_test_dataloader, train_dloader, val_dloader, test_dloader, model_input_dimension = get_data_loader(args=args)
    # args.model_input_dimension = model_input_dimension
    train(gan_train_dloader, gan_pos_val_dloader, gan_neg_val_dataloader, gan_pos_test_dloader, gan_neg_test_dataloader, model_input_dimension*args.seq_len_in, args.dropout_prob, num_epochs=args.num_epochs, batch_size=args.batch_size, nz=args.nz, lr=args.lr, beta1=args.beta1, beta2=args.beta2)


if __name__ == '__main__':
    gc.collect()
    torch.cuda.empty_cache()  # empty cached CUDA memory

    # 1. Modify Arguments
    parser = create_parser()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    main(args)