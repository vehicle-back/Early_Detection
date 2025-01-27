import numpy as np
import argparse
import torch
import torch.optim as optim
import logging
import os
import pickle
import sys
import gc

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime as dt

from models import *
from data_loader import get_data_loader
from GTrans_data_loader import get_GTrans_data_loader
from utils import seed_torch, compute_train_time, save_checkpoint, create_dir, log_train_meta
# from torchmetrics.functional.regression import mean_absolute_percentage_error
def print_hook(module, input, output):
    print(f"{module.__class__.__name__} output contains NaN: {torch.isnan(output).any()}")


########################################
#       TRAINING/TESTING FUNCTIONS     #
########################################


def train(train_dataloader, model, opt, epoch, args, writer):

    model.train()
    step = epoch*len(train_dataloader)
    num_sample = 0
    if args.task in ["LR", "no_fact", "rec", "nonrec"]:
        epoch_loss = [0.0]
    else:
        epoch_loss = [0.0, 0.0]

    
    rec_nonrec_number_count = 0

    for i, batch in enumerate(train_dataloader):
        x, target = batch
        x = x.to(args.device)  # (batch_size, in_seq_len, dim_in)
        target = target.to(args.device)  # (batch_size, out_seq_len + 1, dim_out, 2 or 4) for TrafficModel, (batch_size, out_seq_len + 1, dim_out) for TrafficSeq2Seq

        # Forward Pass
        # The first element retruned is speed prediction (or inc prediction in "LR" task)
        # The second element returned is incident predictions (logits) in finetune task, or hidden tensor in other tasks
        # The third element returned is attention weights, which won't be used here but will be visualized during inference.
        pred, inc_pred, _ = model(x, target, mode="train")

        # Compute Loss
        criterion_spd = torch.nn.MSELoss()
        criterion_inc = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(args.LR_pos_weight)) # combine nn.Sigmoid() with nn.BCELoss() but more numerically stable; don't forget to set pos_weight 
        # criterion_inc = torch.nn.BCELoss(weight=torch.tensor(args.LR_pos_weight))
        if args.task == "LR":
            loss_per_sample = [criterion_inc(pred, target[:, 1:, :, 3])]  # avg loss per time step per sample
            epoch_loss = [epoch_loss[i] + loss_per_sample[i].detach()*x.size(0)  for i in range(len(epoch_loss))]
        elif args.task in ["no_fact", "rec", "nonrec"]: 
            # register hook to only consider certain segments for the computation of loss
            if args.task == "rec":
                h = pred.register_hook(lambda grad: grad * (target[:, 1:, :, 3] < 0.5).float())
                raw_loss_matrix = torch.square(pred-target[:, 1:, :, 0])
                rec_mask = target[:, 1:, :, 3] < 0.5
                loss_per_sample = [torch.tensor(0)] # in case of not declaring loss_per_sample if not run the following
                if rec_mask.any():
                    masked_elements = torch.masked_select(raw_loss_matrix, rec_mask)
                    loss = torch.mean(masked_elements)
                    loss_per_sample = [loss]
                    # loss_per_sample = [criterion_spd(pred, target[:, 1:, :, 0])]  # avg loss per time step per sample
                    epoch_loss = [epoch_loss[i] + loss_per_sample[i].detach()*masked_elements.size(0)  for i in range(len(epoch_loss))]
                    rec_nonrec_number_count += masked_elements.size(0)

            elif args.task == "nonrec":
                h = pred.register_hook(lambda grad: grad * (target[:, 1:, :, 3] >= 0.5).float())
                raw_loss_matrix = torch.square(pred-target[:, 1:, :, 0])
                nonrec_mask = target[:, 1:, :, 3] >= 0.5
                loss_per_sample = [torch.tensor(0)] # in case of not declaring loss_per_sample if not run the following
                if nonrec_mask.any():
                    masked_elements = torch.masked_select(raw_loss_matrix, nonrec_mask)
                    loss = torch.mean(masked_elements)
                    loss_per_sample = [loss]
                    # loss_per_sample = [criterion_spd(pred, target[:, 1:, :, 0])]  # avg loss per time step per sample
                    epoch_loss = [epoch_loss[i] + loss_per_sample[i].detach()*masked_elements.size(0)  for i in range(len(epoch_loss))]
                    rec_nonrec_number_count += masked_elements.size(0)

            else:
                loss_per_sample = [criterion_spd(pred, target[:, 1:, :, 0])]
                epoch_loss = [epoch_loss[i] + loss_per_sample[i].detach()*x.size(0)  for i in range(len(epoch_loss))]
                rec_nonrec_number_count += x.size(0)
        else:
            rec_mask = target[:, 1:, :, 3] < 0.5
            nonrec_mask = target[:, 1:, :, 3] >= 0.5
            raw_loss_matrix = torch.square(pred-target[:, 1:, :, 0])
            rec_masked_elements = torch.masked_select(raw_loss_matrix, rec_mask)
            nonrec_masked_elements = torch.masked_select(raw_loss_matrix, nonrec_mask)*args.nonrec_spd_weight
            spd_loss_list = torch.cat((rec_masked_elements, nonrec_masked_elements))
            spd_loss = torch.mean(spd_loss_list)
            loss_per_sample = [spd_loss, criterion_inc(inc_pred, target[:, 1:, :, 3])]  # avg loss per time step per sample
            # loss_per_sample = [criterion_spd(pred, target[:, 1:, :, 0]), criterion_inc(inc_pred, target[:, 1:, :, 3])]  # avg loss per time step per sample
            epoch_loss = [epoch_loss[i] + loss_per_sample[i].detach()*x.size(0)  for i in range(len(epoch_loss))]
        
        

        # epoch_loss += loss_per_sample*x.size(0)
        num_sample += x.size(0)

        # Backward and Optimize
        opt.zero_grad()
        # loss_per_sample.backward()
        sum(loss_per_sample).backward()
        opt.step()

        # remove hook if needed
        if args.task in {"rec", "nonrec"}:
            h.remove()

        # Logging
        if args.task in ["LR", "no_fact", "rec", "nonrec"]:
            writer.add_scalar("train_loss", sum(loss_per_sample).item(), step+i)
        else:
            writer.add_scalar("train_loss", sum(loss_per_sample).item(), step+i)
    
    
    if args.task in ['rec', 'nonrec']: # rescale 
        avg_epoch_loss = [e/rec_nonrec_number_count for e in epoch_loss]
    else:
        avg_epoch_loss = [e/num_sample for e in epoch_loss]
    
    # Decay Learning Rate
    # scheduler.step()
    
    # return epoch_loss/len(train_dataloader)
    # return epoch_loss/num_sample  # avg loss per time step per sample
    return avg_epoch_loss

def Compute_F1(pred, truth, threshold):  # F1 eval
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
    # accuracy = (np.sum(TP_mask)+np.sum(TN_mask))/(np.sum(TP_mask)+np.sum(TN_mask)+np.sum(FP_mask)+np.sum(FN_mask))
    percision = np.sum(TP_mask)/(np.sum(TP_mask)+np.sum(FP_mask))
    F1_score = 2*((percision*recall)/(percision+recall))
    return F1_score

def find_LR_threshold(pred, truth): # F1 eval
    thresholds = []
    for i in range(0, 1001):
        threshold = round(i * 0.001, 4)
        thresholds.append(threshold)
    best_F1 = 0
    best_threshold = 0
    for threshold in thresholds:
        threshold_specific_F1 = Compute_F1(pred, truth, threshold)
        if threshold_specific_F1>best_F1:
            best_F1 = threshold_specific_F1
            best_threshold = threshold
    return best_F1, best_threshold

def eval(eval_dataloader, model, epoch, args, writer, eval_for_validation):
    
    model.eval() # deactivate dropout, and adjust for batch normalization
    step = epoch*len(eval_dataloader)
    num_sample = 0
    rec_nonrec_number_count = 0
    if args.task in ["LR", "no_fact", "rec", "nonrec"]:
        epoch_loss = [0.0]
    else:
        epoch_loss = [0.0, 0.0]
    
    inc_pred_all = torch.Tensor() # F1 eval
    inc_target_all = torch.Tensor() # F1 eval

    for i, batch in enumerate(eval_dataloader):
        #if i == 0 and not eval_for_validation:
            #print(batch)
        x, target = batch
        x = x.to(args.device)  # (batch_size, in_seq_len, dim_in)
        target = target.to(args.device)  # (batch_size, out_seq_len + 1, dim_out, 2) for TrafficModel, (batch_size, out_seq_len + 1, dim_out) for TrafficSeq2Seq

        with torch.no_grad():
            # Forward Pass
            # The first element retruned is speed prediction (or inc prediction in "LR" task)
            # The second element returned is incident predictions (logits) in finetune task, or hidden tensor in other tasks
            # The third element returned is attention weights, which won't be used here but will be visualized during inference.
            pred, inc_pred, _ = model(x, target, mode="eval")

            # Compute Loss
            criterion_spd = torch.nn.MSELoss()
            criterion_inc = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(args.LR_pos_weight)) # combine nn.Sigmoid() with nn.BCELoss() but more numerically stable; don't forget to set pos_weight 

            if args.task == "LR":
                loss_per_sample = [criterion_inc(pred, target[:, 1:, :, 3])]  # avg loss per time step per sample
                epoch_loss = [epoch_loss[i] + loss_per_sample[i].detach()*x.size(0)  for i in range(len(epoch_loss))]
      
                if inc_pred_all.nelement() == 0: # F1 eval
                    inc_pred_all = pred.detach() # F1 eval
                else: # F1 eval
                    inc_pred_all = torch.cat((inc_pred_all, pred.detach()), dim=0) # F1 eval
                if inc_target_all.nelement() == 0: # F1 eval
                    inc_target_all = target[:, 1:, :, 3].detach() # F1 eval
                else: # F1 eval
                    inc_target_all = torch.cat((inc_target_all,  target[:, 1:, :, 3].detach()), dim=0) # F1 eval
                
                

            elif args.task in ["no_fact", "rec", "nonrec"]: 
                if args.task == "rec":
                    raw_loss_matrix = torch.square(pred-target[:, 1:, :, 0])
                    rec_mask = target[:, 1:, :, 3] < 0.5
                    loss_per_sample = [torch.tensor(0)] # in case of not declaring loss_per_sample if not run the following
                    if rec_mask.any():
                        masked_elements = torch.masked_select(raw_loss_matrix, rec_mask)
                        loss = torch.mean(masked_elements)
                        loss_per_sample = [loss]
                        epoch_loss = [epoch_loss[i] + loss_per_sample[i].detach()*masked_elements.size(0)  for i in range(len(epoch_loss))]
                        rec_nonrec_number_count += masked_elements.size(0)
                elif args.task == "nonrec":
                    raw_loss_matrix = torch.square(pred-target[:, 1:, :, 0])
                    nonrec_mask = target[:, 1:, :, 3] >= 0.5
                    loss_per_sample = [torch.tensor(0)] # in case of not declaring loss_per_sample if not run the following
                    if nonrec_mask.any():
                        masked_elements = torch.masked_select(raw_loss_matrix, nonrec_mask)
                        loss = torch.mean(masked_elements)
                        loss_per_sample = [loss]
                        epoch_loss = [epoch_loss[i] + loss_per_sample[i].detach()*masked_elements.size(0)  for i in range(len(epoch_loss))]
                        rec_nonrec_number_count += masked_elements.size(0)
                else:
                    loss_per_sample = [criterion_spd(pred, target[:, 1:, :, 0])]
                    epoch_loss = [epoch_loss[i] + loss_per_sample[i].detach()*x.size(0)  for i in range(len(epoch_loss))]
                    rec_nonrec_number_count += x.size(0)

                
            else:
                rec_mask = target[:, 1:, :, 3] < 0.5
                nonrec_mask = target[:, 1:, :, 3] >= 0.5
                raw_loss_matrix = torch.square(pred-target[:, 1:, :, 0])
                rec_masked_elements = torch.masked_select(raw_loss_matrix, rec_mask)
                nonrec_masked_elements = torch.masked_select(raw_loss_matrix, nonrec_mask)*args.nonrec_spd_weight
                spd_loss_list = torch.cat((rec_masked_elements, nonrec_masked_elements))
                spd_loss = torch.mean(spd_loss_list)
                loss_per_sample = [spd_loss, criterion_inc(inc_pred, target[:, 1:, :, 3])]  # avg loss per time step per sampl
                epoch_loss = [epoch_loss[i] + loss_per_sample[i].detach()*x.size(0)  for i in range(len(epoch_loss))]

                if inc_pred_all.nelement() == 0: # F1 eval
                    inc_pred_all = pred.detach() # F1 eval
                else: # F1 eval
                    inc_pred_all = torch.cat((inc_pred_all, pred.detach()), dim=0) # F1 eval

                if inc_target_all.nelement() == 0: # F1 eval
                    inc_target_all = target[:, 1:, :, 3].detach() # F1 eval
                else: # F1 eval
                    inc_target_all = torch.cat((inc_target_all,  target[:, 1:, :, 3].detach()), dim=0) # F1 eval
     
            num_sample += x.size(0)


        if args.task in ["LR", "no_fact", "rec", "nonrec"]:
            if eval_for_validation:
                writer.add_scalar("val_loss", sum(loss_per_sample).item(), step+i)
            else:
                writer.add_scalar("test_loss", sum(loss_per_sample).item(), step+i)
        else:
            if eval_for_validation:
                writer.add_scalar("train_loss", sum(loss_per_sample).item(), step+i)
            else:
                writer.add_scalar("train_loss", sum(loss_per_sample).item(), step+i)
    

    if args.task in ['rec', 'nonrec', "no_fact"]: # rescale 
        avg_epoch_loss = [e/rec_nonrec_number_count for e in epoch_loss]
    else:
        avg_epoch_loss = [e/num_sample for e in epoch_loss]
        if eval_for_validation: # F1, add case
            LR_pred = torch.sigmoid(inc_pred_all).cpu()
            LR_truth = inc_target_all.cpu()
            half_number = int(LR_pred.size(0)/2)
            val_1_F1, best_threshold = find_LR_threshold(LR_pred[:half_number, :, :], LR_truth[:half_number, :, :])  # F1 eval
            args.LR_threshold = best_threshold
            val_2_F1 = Compute_F1(LR_pred[half_number:, :, :], LR_truth[half_number:, :, :], args.LR_threshold)
            # print('Vali, epoch ', epoch, ' F1 score is', val_1_F1, val_2_F1, args.LR_threshold) # F1 eval
            avg_epoch_loss[-1] = -val_2_F1 # F1 score is higher is better
        else:
            LR_pred = torch.sigmoid(inc_pred_all).cpu()
            LR_truth = inc_target_all.cpu()
            test_F1 = Compute_F1(LR_pred, LR_truth, args.LR_threshold)
            # print('Test, epoch ', epoch, ' F1 score is', test_F1, test_F1, args.LR_threshold) # F1 eval
            avg_epoch_loss[-1] = -test_F1
    
    return avg_epoch_loss


########################################
#           TRAINING PIPELINE          #
########################################
def main(args):
    """
    train model, evaluate on test data, and save checkpoints
    """
    # 1. Create Directories
    create_dir(args.checkpoint_dir)
    create_dir(args.log_dir)

    # 2. Set up Logger for Tensorboard and Logging
    writer = SummaryWriter('{}/{}'.format(args.log_dir,args.exp_name))
    if args.load_checkpoint_epoch > 0:
        logging.basicConfig(filename=f"{args.log_dir}/{args.exp_name}/training.log", filemode="a", format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.DEBUG) 
    else:
        logging.basicConfig(filename=f"{args.log_dir}/{args.exp_name}/training.log", filemode="w", format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.DEBUG) 
        # log meta info of training experiment
        log_train_meta(args)
    
    # 7. Load Data for Training & Testing
    # if args.model_type == "GTrans":
    #    args.edge_idx = get_edge_idx(args).to(args.device)
        
    if args.model_type == "GTrans":
        train_dataloader, val_dataloader, test_dataloader, args.dim_in, args.LR_pos_weight, adj_matrix = get_GTrans_data_loader(args=args)
        args.num_node = adj_matrix.shape[0]

        rows, cols = np.where(adj_matrix != 0)
        args.edge_idx = torch.tensor([rows, cols], dtype=torch.long).to(args.device)
        assert args.edge_idx.min() >= 0
        assert args.edge_idx.max() < args.num_node  
        assert args.edge_idx.dim() == 2 and args.edge_idx.size(0) == 2
    else:
        train_dataloader, val_dataloader, test_dataloader, args.dim_in, args.LR_pos_weight = get_data_loader(args=args)
        _, _, _, _, _, _ = get_GTrans_data_loader(args=args)
    
    print('link_id', args.link_id, 'dim_in', args.dim_in, 'LR_pos_weight', args.LR_pos_weight)
    logging.info(f"successfully loaded data \n")

    # 3. Initialize Model
    if args.task == "no_fact":
        if args.model_type == "Seq2Seq":
            model = Seq2SeqNoFact(args).to(args.device)
        elif args.model_type == "Trans":
            model = TransNoFact(args).to(args.device)
        else:
            model = GTransNoFact(args).to(args.device)
    elif args.task == "naive":
        if args.model_type == "Seq2Seq":
            model = Seq2SeqFactNaive(args).to(args.device)
        elif args.model_type == "Trans":
            model = TransFactNaive(args).to(args.device)
        else:
            pass
    elif args.task == "naive_2enc":
        if args.model_type == "Seq2Seq":
            model = Seq2SeqFactNaive_2enc(args).to(args.device)
        else:
            pass
    else:
        if args.model_type == "Seq2Seq":
            model = Seq2SeqFact(args).to(args.device)
        elif args.model_type == "Trans":
            model = TransFact(args).to(args.device)
        else:
            print("GTrans Model Intialized")
            model = GTransFact(args).to(args.device)
            # hook_1 = model.encoder.st_blocks[0].spatial_module.register_forward_hook(print_hook) # check
            
    

    # 4. Set up Optimizer and LR Scheduler
    # opt = optim.Adam(model.parameters(), args.lr, betas=(0.9, 0.999))
    if args.task == "rec":
        opt = optim.AdamW(params=model.parameters(), lr=args.lr_rec, betas=(0.9, 0.999), weight_decay=0.01)  # performs better than Adam
    elif args.task == "nonrec":
        opt = optim.AdamW(params=model.parameters(), lr=args.lr_nonrec, betas=(0.9, 0.999), weight_decay=0.01)  # performs better than Ada
    elif args.task =='finetune':
        opt = optim.AdamW(params=model.parameters(), lr=args.lr_finetune, betas=(0.9, 0.999), weight_decay=0.01)  # performs better than Ada
    else:
        opt = optim.AdamW(params=model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)  # performs better than Ada
    # scheduler = optim.lr_scheduler.StepLR(opt, step_size=args.lr_decay_freq, gamma=args.lr_decay_rate)


    logging.info('{:*^100}'.format(" LOADING PROGRESS "))

    # 5. Load Checkpoint 
    if args.load_checkpoint:   
        checkpoint_path = "{}/{}.pt".format(args.checkpoint_dir, args.load_checkpoint)
        with open(checkpoint_path, 'rb') as f:
            state_dict = torch.load(f, map_location=args.device)
            model.load_state_dict(state_dict["model"])
            opt.load_state_dict(state_dict["optimizer"])
        logging.info(f"successfully loaded checkpoint from {checkpoint_path}")
    else:
        # in "rec" and "nonrec" tasks, if checkpoint is not specified, 
        # then we need to initialize the model with best checkpoint from "LR"
        if (args.task == "rec" or args.task == "nonrec" ): # check
            checkpoint_path = "/".join(args.checkpoint_dir.split("/")[:-1]) + "/LR/best_{}.pt".format(args.LR_exp_name)
            with open(checkpoint_path, 'rb') as f:
                state_dict = torch.load(f, map_location=args.device)
                model.load_state_dict(state_dict["model"])
            logging.info(f"successfully loaded checkpoint from {checkpoint_path}")

        # in "finetune" task, if checkpoint is not specified,
        # then we need to initialize the model with best checkpoint from "LR" (encoder & LR_decoder), "rec" (rec_decoder) and "nonrec" (nonrec_decoder)
        elif args.task == "finetune":

            LR_model_path = "/".join(args.checkpoint_dir.split("/")[:-1]) + "/LR/best_{}.pt".format("_".join(args.LR_exp_name.split("_")))#[:-1]))  
            rec_model_path = "/".join(args.checkpoint_dir.split("/")[:-1]) + "/rec/best_{}.pt".format("_".join(args.rec_exp_name.split("_")))#[:-1]))
            nonrec_model_path = "/".join(args.checkpoint_dir.split("/")[:-1]) + "/nonrec/best_{}.pt".format("_".join(args.nonrec_exp_name.split("_")))#[:-1]))

            # load state dict partially to initialize corresponding modules of TrafficModel
            with open(LR_model_path, 'rb') as f_LR, open(rec_model_path, 'rb') as f_rec, open(nonrec_model_path, 'rb') as f_nonrec:
                
                # load state dict 
                state_dict_LR = torch.load(f_LR, map_location=args.device)
                state_dict_rec = torch.load(f_rec, map_location=args.device)
                state_dict_nonrec = torch.load(f_nonrec, map_location=args.device)

                # retain corresponding modules only 
                enc_dec_lr_state_d = {k:v for k, v in state_dict_LR["model"].items() if "LR" in k or "encoder" in k}
                dec_rec_state_d = {k:v for k, v in state_dict_rec["model"].items() if "rec" in k and "nonrec" not in k}
                dec_nonrec_state_d = {k:v for k, v in state_dict_nonrec["model"].items() if "nonrec" in k}

                # load state dict of corresponding modules 
                model.load_state_dict(enc_dec_lr_state_d, strict=False)
                model.load_state_dict(dec_rec_state_d, strict=False)
                model.load_state_dict(dec_nonrec_state_d, strict = False)

            logging.info(f"successfully loaded checkpoint from \
                    {LR_model_path} \n\
                    {rec_model_path} \n\
                    {nonrec_model_path} ")

    # 6. Freeze Module 
    if args.task == "LR":
        # in "LR" task, freeze decoders for recurrent and nonrecurrent prediction
        model.rec_decoder.requires_grad_(False)
        model.nonrec_decoder.requires_grad_(False)
    elif args.task == "rec":
        # in "rec" task, freeze everything except recurrent decoder
        #model.encoder.requires_grad_(False)
        model.LR_decoder.requires_grad_(False)
        model.nonrec_decoder.requires_grad_(False)
    elif args.task == "nonrec":
        # in "nonrec" task, freeze everythign except nonrecurrent decoder
        model.encoder.requires_grad_(False)
        model.LR_decoder.requires_grad_(False)
        model.rec_decoder.requires_grad_(False)

    # 8. Train, Test & Save Checkpoints
    # Logging
    if args.load_checkpoint_epoch > 0:
        logging.info('{:=^100}'.format(" Training Resumes from Epoch {} ".format(args.load_checkpoint_epoch)))
    else:
        logging.info('{:=^100}'.format(" Training Starts "))
        logging.info("please check tensorboard for plots of experiment {}/{}".format(args.log_dir, args.exp_name))
        logging.info("please check logging messages at {}/{}/training.log".format(args.log_dir, args.exp_name))
    logging.info(f"Average losses are calculated per time step per sample.")
    
    start_time = dt.now()
    best_val_loss = [float("inf")]
    best_test_loss = [float("inf")]
    if args.load_checkpoint_epoch > 0:
        checkpoint_path = "{}/{}.pt".format(args.checkpoint_dir, args.load_checkpoint)
        with open(checkpoint_path, 'rb') as f:
            state_dict = torch.load(f, map_location=args.device)
            best_val_loss = state_dict["losses"]["best_val_epoch_loss_hitherto"]
    best_epoch = -1  # evaluated based on validation loss

    for epoch in range(max(0, args.load_checkpoint_epoch+1), args.num_epochs):
        # Train
        train_epoch_loss = train(train_dataloader, model, opt, epoch, args, writer)   
        val_epoch_loss = eval(val_dataloader, model, epoch, args, writer, True)
        test_epoch_loss = eval(test_dataloader, model, epoch, args, writer, False)

        # hook_1.remove() # check

        if len(train_epoch_loss) == 1:
            logging.info("epoch: {}   train loss: {:.4f}   val loss: {:.4f}   test loss: {:.4f}".format(epoch, train_epoch_loss[0], val_epoch_loss[0], test_epoch_loss[0]))
        else:
            logging.info("epoch: {}   train loss: {:.4f} + {:.4f}   val loss: {:.4f} + {:.4f}   test loss: {:.4f} + {:.4f}".format(epoch, train_epoch_loss[0], train_epoch_loss[1], val_epoch_loss[0], val_epoch_loss[1], test_epoch_loss[0], test_epoch_loss[1]))
        
        # Save Model Checkpoint Regularly
        if epoch % args.checkpoint_every == 0:
            logging.info("checkpoint saved at epoch {}".format(epoch))
            save_checkpoint(epoch=epoch, model=model, opt=opt, args=args, train_epoch_loss=train_epoch_loss, val_epoch_loss=val_epoch_loss, test_epoch_loss=test_epoch_loss, best_val_epoch_loss_hitherto=best_val_loss, best=False)

        # Save Best Model Checkpoint
        # if (val_epoch_loss <= best_val_loss):
        #     best_val_loss = val_epoch_loss
        #     best_test_loss = test_epoch_loss
        #     best_epoch = epoch
        #     logging.info("best model saved at epoch {}".format(epoch))
        #     save_checkpoint(epoch=epoch, model=model, opt=opt, args=args, train_epoch_loss=train_epoch_loss, val_epoch_loss=val_epoch_loss, test_epoch_loss=test_epoch_loss, best_val_epoch_loss_hitherto=best_val_loss, best=True)
        if (val_epoch_loss[0] < best_val_loss[0]):
            best_val_loss = val_epoch_loss
            best_test_loss = test_epoch_loss
            best_epoch = epoch
            logging.info("best model saved at epoch {}".format(epoch))
            save_checkpoint(epoch=epoch, model=model, opt=opt, args=args, train_epoch_loss=train_epoch_loss, val_epoch_loss= val_epoch_loss, test_epoch_loss= test_epoch_loss, best_val_epoch_loss_hitherto=best_val_loss, best=True)

    end_time = dt.now()
    training_time = compute_train_time(start_time, end_time)
    if len(best_val_loss) == 1:
        logging.info('{:=^100}'.format(" Training completes after {} hr {} min {} sec ({} epochs trained, best epoch at {} with val loss = {:.4f} and test loss = {:.4f}) ".format(training_time["hours"], training_time["minutes"], training_time["seconds"], args.num_epochs, best_epoch, best_val_loss[0], best_test_loss[0])))
    else:
        logging.info('{:=^100}'.format(" Training completes after {} hr {} min {} sec ({} epochs trained, best epoch at {} with val loss = {:.4f} + {:.4f} and test loss = {:.4f} + {:.4f}) ".format(training_time["hours"], training_time["minutes"], training_time["seconds"], args.num_epochs, best_epoch, best_val_loss[0], best_val_loss[1], best_test_loss[0], best_test_loss[1])))

def create_parser():
    """
    Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()
    

    # 1. Model hyper-parameters
    parser.add_argument('--model_type', type=str, help="Choose one from 'Seq2Seq', 'Trans', 'GTrans'")

    parser.add_argument('--dim_hidden', type=int, default=64, help='Hidden dimension in encoder and decoder')
    parser.add_argument('--teacher_forcing_ratio', type=float, default=0.5, help='threshold of teacher forcing')
    parser.add_argument('--dropout_prob', type=float, default = 0.2, help='dropout probability')
    parser.add_argument('--dropout_rec', type=float, default = 0.2, help='dropout probability, for rec')
    parser.add_argument('--dropout_nonrec', type=float, default=0.35, help='dropout probability, for nonrec')
    parser.add_argument('--dropout_finetune', type=float, default=0.1, help='dropout probability, for fintuning')
    # parser.add_argument('--dropout_prob_spd', type=float, default=0.35, help='dropout probability for spd task')

    # Seq2Seq
    parser.add_argument('--num_layer_GRU', type=int, default=2, help='Number of stacked GRUs in encoder and decoder')

    # Trans & GTrans
    parser.add_argument('--num_head', type=int, default=8, help='Number of heads in a transformer encoder/decoder layer')
    parser.add_argument('--num_layer_EncTrans', type=int, default=2, help='Number of transformer encoder layers')
    parser.add_argument('--num_layer_DecTrans', type=int, default=2, help='Number of transformer decoder layers')
    parser.add_argument('--num_STBlock', type=int, default=1, help='Number of spatial-temporal blocks')

    parser.add_argument('--seq_len_in', type=int, default=7, help='sequence length of input')
    parser.add_argument('--seq_len_out', type=int, default=6, help='sequence length of output')
    parser.add_argument('--freq_out', type=int, default=5, help='frequency of output data')

    parser.add_argument('--LR_threshold', type=float, default=0.5, help='threshold of a prediction be considered as an incident') # F1 eval

    parser.add_argument('--inc_threshold', type=float, default=0.5, help='threshold of a prediction be considered as an incident')
    parser.add_argument('--LR_pos_weight', type=float, default=100, help='ratio of positive samples in incident ground truth, automatic, no need to set')
    parser.add_argument('--nonrec_spd_weight', type=float, default=15, help = 'the weight added to non recurrent speed when computing finetune mse')  
    parser.add_argument('--use_expectation', action="store_true", help='use expectation of speed prediction as model output')  

    # Assuming perfect incident status prediction at stage 1 of the 2-stage model (Traffic)
    parser.add_argument('--use_gt_inc', type=int, default=0, help='use ground truth of indicent status as input to second stage')

    # 2. Data Hyper-parameters
    parser.add_argument('--training_granularity', type=int, default=1, help='the minimum data between different slices the training data, to reduce the overfitting effect of the highly overlapped data, for example, if we want 10 min and input data interval (args.freq_out) is 5 min, then this number should be 2')
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
    parser.add_argument('--use_waze', action="store_true", default=True, help='use waze info or not')
    
    
    # 3. Training Hyper-parameters
    parser.add_argument('--county', type=str, default="TSMO", help="Choose one from 'Cranberry', 'TSMO'")
    parser.add_argument('--link_id', type=str, default = '110+04483', help='the link to be analysed')
    parser.add_argument('--number_of_point_per_day', type=int, default = 186, help='for 05:30-21:00 with 5 min granularity, is 186')
    parser.add_argument('--number_of_business_day', type=int, default = 260, help='number of total business day in training/val/test set')
    parser.add_argument('--upstream_range_mile', type=int, default = 2, help='the upstream range mile of the link model')
    parser.add_argument('--downstream_range_mile', type=int, default = 1, help='the downstream range mile of the link model')
    parser.add_argument('--inc_ahead_label_min', type=int, default = 0, help='incident ahead label, force model to report earlier')
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
    parser.add_argument('--task', type=str, help="Choose one from 'LR', 'rec', 'nonrec', 'finetune', 'no_fact', 'naive', 'naive_2enc")  

    parser.add_argument('--num_epochs', type=int, default=901, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=64, help='Number of sequences in a batch.')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of threads to use for the DataLoader.')
    parser.add_argument('--lr', type=float, default= 0.00025, help='Learning rate (default 0.00025)') # 0.00025
    parser.add_argument('--lr_rec', type=float, default= 0.00025, help='Learning rate (default 0.00025)')
    parser.add_argument('--lr_nonrec', type=float, default= 0.00025, help='Learning rate (default 0.00025)')
    parser.add_argument('--lr_finetune', type=float, default= 0.00025, help='Learning rate (default 0.00025)')

    parser.add_argument('--exp_name', type=str,help='Name of the experiment')
    parser.add_argument('--LR_exp_name', type=str,help='Name of the LR experiment, for checkpoint loading in rec')
    parser.add_argument('--rec_exp_name', type=str,help='Name of the experiment')
    parser.add_argument('--nonrec_exp_name', type=str,help='Name of the experiment')

    # 4. Directories and Checkpoint/Sample Iterations
    parser.add_argument('--data_dir', type=str, default='E:/two_stage_model')
    parser.add_argument('--log_dir', type=str, default='./logs')

    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--checkpoint_every', type=int , default=100)

    parser.add_argument('--load_checkpoint', type=str, default='', help='Name of the checkpoint')
    parser.add_argument('--load_checkpoint_epoch', type=int, default=-1, help='Epoch of the checkpoint')

    return parser


if __name__ == '__main__':
    gc.collect()
    torch.cuda.empty_cache()  # empty cached CUDA memory

    # 1. Modify Arguments
    parser = create_parser()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    # For reproducibility
    seed_torch(args.seed)

    # Modify Data & Model Hyper-parameters If Needed for Different Counties
    if args.county == "TSMO":
        args.number_of_business_day = 260
        if args.model_type == "GTrans":
            args.dim_hidden = 32
    else:
        args.number_of_business_day = 522

    # Task specific directories
    args.log_dir += f"/{args.county}/{args.link_id}/{args.task}"
    args.checkpoint_dir += f"/{args.county}/{args.link_id}/{args.task}" 

    args.exp_name = args.model_type
    if args.model_type == 'Seq2Seq':
        args.exp_name += f"_G{args.num_layer_GRU}"
        args.exp_name += f"_{args.upstream_range_mile}_{args.downstream_range_mile}_{str(args.use_spd_all)[0]}_{str(args.use_spd_truck)[0]}_{str(args.use_spd_pv)[0]}_{str(args.use_slowdown_spd)[0]}_{str(args.use_tti)[0]}_{str(args.use_HA)[0]}_{str(args.use_dens)[0]}_{str(args.use_weather)[0]}_{str(args.use_time)[0]}_{str(args.use_waze)[0]}_in_{args.seq_len_in}_out_{args.seq_len_out}_tg_{args.training_granularity}_cv_{args.time_series_cv_ratio}_dp_{args.dropout_prob}_hidden_{args.dim_hidden}_batch_{args.batch_size}_lr_{args.lr}_tf_{args.teacher_forcing_ratio}" 
    else: # for Transformer and G-Transformer
        args.exp_name += f"_H{args.num_head}_E{args.num_layer_EncTrans}_D{args.num_layer_DecTrans}"
        args.exp_name += f"_{args.upstream_range_mile}_{args.downstream_range_mile}_{str(args.use_spd_all)[0]}_{str(args.use_spd_truck)[0]}_{str(args.use_spd_pv)[0]}_{str(args.use_slowdown_spd)[0]}_{str(args.use_tti)[0]}_{str(args.use_HA)[0]}_{str(args.use_dens)[0]}_{str(args.use_weather)[0]}_{str(args.use_time)[0]}_{str(args.use_waze)[0]}_in_{args.seq_len_in}_out_{args.seq_len_out}_tg_{args.training_granularity}_cv_{args.time_series_cv_ratio}_dp_{args.dropout_prob}_hidden_{args.dim_hidden}_batch_{args.batch_size}_lr_{args.lr}_tf_{args.teacher_forcing_ratio}" 

    if args.inc_ahead_label_min > 0:
        args.exp_name += f"_inca_{str(args.inc_ahead_label_min)}"
    
    # for loading checkpoints purpose
    args.LR_exp_name = args.exp_name
    args.rec_exp_name = args.LR_exp_name + f'_{args.lr_rec}_{args.dropout_rec}'
    args.nonrec_exp_name = args.LR_exp_name + f'_{args.lr_nonrec}_{args.dropout_nonrec}'

    
    if args.task == "finetune":
        args.exp_name += f"_{str(args.nonrec_spd_weight)}"
        args.exp_name += f"_uE_{str(args.use_expectation)[0]}"
        args.exp_name +=  f'_rec_{args.lr_rec}_{args.dropout_rec}_nrec_{args.lr_nonrec}_{args.dropout_nonrec}_ft_{args.lr_finetune}_{args.dropout_finetune}'
        args.dropout_prob = args.dropout_finetune
        if not args.use_expectation:
            args.exp_name += f"_{str(args.inc_threshold)}"
    elif args.task == 'rec':
        args.exp_name += f'_{args.lr_rec}_{args.dropout_rec}'
        args.dropout_prob = args.dropout_rec
    elif args.task == 'nonrec':
        args.exp_name += f'_{args.lr_nonrec}_{args.dropout_nonrec}'
        args.dropout_prob = args.dropout_nonrec 

    if args.load_checkpoint_epoch > 0:
        args.load_checkpoint = f"epoch_{args.load_checkpoint_epoch}_{args.exp_name}"
    
    # 2. Execute Training Pipeline
    main(args)