import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import TransformerEncoder, TransformerEncoderLayer

from modules import *

##########################
#     1. Seq-to-seq      #
##########################
# 1.1 Sequence-to-sequence Model without Factorization
class Seq2SeqNoFact(nn.Module):
    '''
    Proposed seq2seq model in "Learning to Recommend Signal Plans under Incidents with Real-Time Traffic Prediction"
    We use it here as a baseline method
    '''
    def __init__(self, args):
        super(Seq2SeqNoFact, self).__init__()
        self.args = args
        self.encoder = EncoderRNN(args=args)
        self.decoder = AttnDecoderRNN(args=args)
    
    def forward(self, x, target, mode):
        '''
        INPUTs
            x: input, (batch_size, seq_len_in, dim_in)
            target: (batch_size, seq_len_out+1, dim_out, 5), the last dimension refers to 1~3: speed (all, truck, pv); 4: incident status; 5: Waze incident status
            mode: string of value "train" or "eval", denoting the mode to control decoder

        OUTPUTs
            dec_out: (batch_size, seq_len_out, dim_out)  
            dec_hidden: (num_layer, batch_size, dim_hidden)
            attn_weights: (batch_size, seq_len_out, seq_len_in)
        '''
        batch_size = x.size(0)

        # pass into encoder
        ini_hidden = self.encoder.initHidden(batch_size = batch_size)
        enc_out, enc_hidden = self.encoder(x, ini_hidden)

        # pass into decoder
        dec_out, dec_hidden, attn_weights = self.decoder(target[..., 0], enc_hidden ,enc_out, mode)

        return dec_out, dec_hidden, attn_weights

# 1.2 Sequence-to-sequence Model with Factorization
class Seq2SeqFact(nn.Module):
    
    def __init__(self, args):
        super(Seq2SeqFact, self).__init__()
        self.args = args
        self.encoder = EncoderRNN(args=args)
        self.inc_activation = nn.Sigmoid()

        # For weight computation in final speed prediction
        # Option 2.1 - Learnable rec_weight and nonrec_weight
        self.sum_weight = nn.Parameter(data=torch.tensor(1.0))

        # Option 2.2 - MLP (doesn't perform as well as Option 1)
        # self.out_processing = OutputProcessing(dim_list=[2,16,16,32,32,16,16,1]) 

        # Option 3 - Theshold of Incident Prediction
        self.threshold = torch.nn.Threshold(threshold=-0.5, value=-1.0)

        self.LR_decoder = AttnDecoderRNN(args=args)  # decoder for incident prediction (logistic regression)
        self.rec_decoder = AttnDecoderRNN(args=args)  # decoder for speed prediction in recurrent scenario
        self.nonrec_decoder = AttnDecoderRNN(args=args)  # decoder for speed prediction in non-recurrent scenario

        self.args = args
    
    def forward(self, x, target, mode):
        '''
        INPUTs
            x: input, (batch_size, seq_len_in, num_feat_dim + incident_feat_dim)
            target: (batch_size, seq_len_out+1, dim_out, 5), the last dimension refers to 1~3: speed (all, truck, pv); 4: incident status; 5: Waze incident status 
            mode: string of value "train" or "eval", denoting the mode to control decoder

        OUTPUTs
            LR_out: incident status prediction, (batch_size, seq_len_out, dim_out)
            speed_pred: speed prediction, (batch_size, seq_len_out, dim_out or 3*dim_out)
            xxx_attn_weights: (batch_size, seq_len_out, seq_len_in)
        '''
        batch_size = x.size(0)

        # pass into encoder
        ini_hidden = self.encoder.initHidden(batch_size=batch_size)
        enc_out, enc_hidden = self.encoder(x, ini_hidden)

        # pass into decoder
        '''
        xxx_dec_out: (batch_size, seq_len_out, dim_out)  
        xxx_dec_hidden: (num_layer, batch_size, dim_hidden)
        xxx_attn_weights: (batch_size, seq_len_out, seq_len_in)
        '''
        if self.args.task == "LR":
            return self.LR_decoder(target[..., 3], enc_hidden, enc_out, mode)  # Be careful! LR_out hasn't gone through nn.Sigmoid and doesn't denote the probability
        elif self.args.task == "rec":
            return self.rec_decoder(target[..., 0], enc_hidden, enc_out, mode)
        elif self.args.task == "nonrec":
            return self.nonrec_decoder(target[..., 0], enc_hidden, enc_out, mode)
        else:
            LR_out, LR_hidden, LR_attn_weights = self.LR_decoder(target[..., 3], enc_hidden, enc_out, mode)
            inc_pred = self.inc_activation(LR_out)  # convert logits to probabilities
            rec_out, rec_hidden, rec_attn_weights = self.rec_decoder(target[..., 0], enc_hidden, enc_out, mode)
            nonrec_out, nonrec_hidden, nonrec_attn_weights = self.nonrec_decoder(target[..., 0], enc_hidden, enc_out, mode)

            # generate final speed prediction
            if self.args.use_gt_inc:
                # Option 1 - Use Threshold for Incident Ground Truth (for inference ONLY)
                rec_weight = target[:, 1:, :, 3] < 0.5
                nonrec_weight = target[:, 1:, :, 3] >= 0.5
            else:
                # Option 2 - Use Expectation
                if self.args.use_expectation:
                    # Option 2.1 - Learnable rec_weight and nonrec_weight
                    nonrec_weight = torch.clamp(self.sum_weight * inc_pred, 0, 1) # torch.clamp ensures weights are within [0,1]
                    rec_weight = torch.ones(LR_out.size()).to(self.args.device) - nonrec_weight

                    # For Option 2.2 - MLP
                    # nonrec_weight = inc_pred
                    # rec_weight = torch.ones(LR_out.size()).to(self.args.device) - nonrec_weight
                
                # Option 3 - Use Threshold for Incident Prediction
                else:
                    # The following 2 lines of code are not differentiable, therefore failing to update inc_pred through MSE loss of speed_pred.
                    rec_weight = inc_pred < self.args.inc_threshold 
                    nonrec_weight = inc_pred >= self.args.inc_threshold

                    # The following 2 lines of commented code are differentiable, but in fact generates the exact same result as above two lines of code do.
                    # rec_weight = torch.clamp(self.threshold(-inc_pred)+1, max=0.5)*2   # generate 0-1 mask as what "inc_pred < self.args.inc_threshold" does
                    # nonrec_weight = torch.ones(LR_out.size()).to(self.args.device) - rec_weight  # generate 0-1 mask as what "inc_pred >= self.args.inc_threshold" does


            speed_pred = rec_out * rec_weight + nonrec_out * nonrec_weight

            # For Expectation Option 2 - MLP
            # intermediate_spd_pred = torch.stack(tensors=[rec_out * rec_weight, nonrec_out * nonrec_weight], dim=-1)
            # speed_pred = self.out_processing(intermediate_spd_pred).view(intermediate_spd_pred.shape[:3])
            
            return speed_pred, LR_out, [LR_attn_weights, rec_attn_weights, nonrec_attn_weights]

# 1.3 Sequence-to-sequence Model of Naive Combination
class Seq2SeqFactNaive(nn.Module):
    def __init__(self, args):
        super(Seq2SeqFactNaive, self).__init__()
        self.args = args
        self.inc_activation = nn.Sigmoid()

        self.LR_encoder = EncoderRNN(args=args)
        self.rec_encoder = EncoderRNN(args=args)
        self.nonrec_encoder = EncoderRNN(args=args)

        self.LR_decoder = AttnDecoderRNN(args=args)  # decoder for incident prediction (logistic regression)
        self.rec_decoder = AttnDecoderRNN(args=args)  # decoder for speed prediction in recurrent scenario
        self.nonrec_decoder = AttnDecoderRNN(args=args)  # decoder for speed prediction in non-recurrent scenario
    
    def forward(self, x, target, mode):
        '''
        INPUTs
            x: input, (batch_size, seq_len_in, num_feat_dim + incident_feat_dim)
            target: (batch_size, seq_len_out+1, dim_out, 5), the last dimension refers to 1~3: speed (all, truck, pv); 4: incident status; 5: Waze incident status
            mode: string of value "train" or "eval", denoting the mode to control decoder

        OUTPUTs
            LR_out: incident status prediction, (batch_size, seq_len_out, dim_out)
            speed_pred: speed prediction, (batch_size, seq_len_out, dim_out or 3*dim_out)
            xxx_attn_weights: (batch_size, seq_len_out, seq_len_in)
        '''
        batch_size = x.size(0)

        # pass into encoder
        LR_ini_hidden = self.LR_encoder.initHidden(batch_size=batch_size)
        LR_enc_out, LR_enc_hidden = self.LR_encoder(x, LR_ini_hidden)
        rec_ini_hidden = self.rec_encoder.initHidden(batch_size=batch_size)
        rec_enc_out, rec_enc_hidden = self.rec_encoder(x, rec_ini_hidden)
        nonrec_ini_hidden = self.nonrec_encoder.initHidden(batch_size=batch_size)
        nonrec_enc_out, nonrec_enc_hidden = self.nonrec_encoder(x, nonrec_ini_hidden)

        # pass into decoder
        '''
        xxx_dec_out: (batch_size, seq_len_out, dim_out)  
        xxx_dec_hidden: (num_layer, batch_size, dim_hidden)
        xxx_attn_weights: (batch_size, seq_len_out, seq_len_in)
        '''
        LR_out, LR_hidden, LR_attn_weights = self.LR_decoder(target[..., 3], LR_enc_hidden, LR_enc_out, mode)
        inc_pred = self.inc_activation(LR_out)
        rec_out, rec_hidden, rec_attn_weights = self.rec_decoder(target[..., 0], rec_enc_hidden, rec_enc_out, mode)
        nonrec_out, nonrec_hidden, nonrec_attn_weights = self.nonrec_decoder(target[..., 0], nonrec_enc_hidden, nonrec_enc_out, mode)

        # generate final speed prediction
        if self.args.use_gt_inc:
            rec_weight = target[:, 1:, :, 3] < 0.5
            nonrec_weight = target[:, 1:, :, 3] >= 0.5
        else:
            if self.args.use_expectation:
                rec_weight = torch.ones(LR_out.size()).to(self.args.device) - inc_pred
                nonrec_weight = torch.ones(LR_out.size()).to(self.args.device) - rec_weight
            else:
                rec_weight = inc_pred < self.args.inc_threshold
                nonrec_weight = inc_pred >= self.args.inc_threshold

        speed_pred = rec_out * rec_weight + nonrec_out * nonrec_weight
        
        return speed_pred, LR_out, [LR_attn_weights, rec_attn_weights, nonrec_attn_weights]

class Seq2SeqFactNaive_2enc(nn.Module):
    def __init__(self, args):
        super(Seq2SeqFactNaive_2enc, self).__init__()
        self.args = args
        self.inc_activation = nn.Sigmoid()
        self.LR_encoder = EncoderRNN(args=args)
        self.Non_LR_encoder = EncoderRNN(args=args)

        self.LR_decoder = AttnDecoderRNN(args=args)  # decoder for incident prediction (logistic regression)
        self.rec_decoder = AttnDecoderRNN(args=args)  # decoder for speed prediction in recurrent scenario
        self.nonrec_decoder = AttnDecoderRNN(args=args)  # decoder for speed prediction in non-recurrent scenario
    
    def forward(self, x, target, mode):
        '''
        INPUTs
            x: input, (batch_size, seq_len_in, num_feat_dim + incident_feat_dim)
            target: (batch_size, seq_len_out+1, dim_out, 5), the last dimension refers to 1~3: speed (all, truck, pv); 4: incident status; 5: Waze incident status
            mode: string of value "train" or "eval", denoting the mode to control decoder

        OUTPUTs
            LR_out: incident status prediction, (batch_size, seq_len_out, dim_out)
            speed_pred: speed prediction, (batch_size, seq_len_out, dim_out or 3*dim_out)
            xxx_attn_weights: (batch_size, seq_len_out, seq_len_in)
        '''
        batch_size = x.size(0)

        # pass into encoder
        LR_ini_hidden = self.LR_encoder.initHidden(batch_size=batch_size)
        LR_enc_out, LR_enc_hidden = self.LR_encoder(x, LR_ini_hidden)
        Non_LR_ini_hidden = self.Non_LR_encoder.initHidden(batch_size=batch_size)
        Non_LR_enc_out, Non_LR_enc_hidden = self.Non_LR_encoder(x, Non_LR_ini_hidden)

        # pass into decoder
        '''
        xxx_dec_out: (batch_size, seq_len_out, dim_out)  
        xxx_dec_hidden: (num_layer, batch_size, dim_hidden)
        xxx_attn_weights: (batch_size, seq_len_out, seq_len_in)
        '''
        LR_out, LR_hidden, LR_attn_weights = self.LR_decoder(target[..., 3], LR_enc_hidden, LR_enc_out, mode)
        inc_pred = self.inc_activation(LR_out)
        rec_out, rec_hidden, rec_attn_weights = self.rec_decoder(target[..., 0], Non_LR_enc_hidden, Non_LR_enc_out, mode)
        nonrec_out, nonrec_hidden, nonrec_attn_weights = self.nonrec_decoder(target[..., 0], Non_LR_enc_hidden, Non_LR_enc_out, mode)

        # generate final speed prediction
        if self.args.use_gt_inc:
            rec_weight = target[:, 1:, :, 3] < 0.5
            nonrec_weight = target[:, 1:, :, 3] >= 0.5
        else:
            if self.args.use_expectation:
                rec_weight = torch.ones(LR_out.size()).to(self.args.device) - inc_pred
                nonrec_weight = torch.ones(LR_out.size()).to(self.args.device) - rec_weight
            else:
                rec_weight = inc_pred < self.args.inc_threshold
                nonrec_weight = inc_pred >= self.args.inc_threshold

        speed_pred = rec_out * rec_weight + nonrec_out * nonrec_weight
        
        return speed_pred, LR_out, [LR_attn_weights, rec_attn_weights, nonrec_attn_weights]
    
##########################
#     2. Transformer     #
##########################
# 2.1 Transformer Model without Factorization
class TransNoFact(nn.Module):
    def __init__(self, args):
        super(TransNoFact, self).__init__()
        self.encoder = EncoderTrans(args)
        self.decoder = DecoderTrans(args)
        self.args = args
    
    def forward(self, x, target, mode):
        '''
        INPUTs
            x: input, (batch_size, seq_len_in, dim_in)
            target: (batch_size, seq_len_out+1, dim_out, 5), the last dimension refers to 1~3: speed (all, truck, pv); 4: incident status; 5: Waze incident status
            mode: string of value "train" or "eval", denoting the mode to control decoder

        OUTPUTs
            dec_out: (batch_size, seq_len_out, dim_out or 3*dim_out)  
        '''
        enc_out = self.encoder(x)  # (batch_size, seq_len_in, dim_hidden)
        dec_out = self.decoder(target[..., 0], enc_out, mode)

        return dec_out, 0, 0


# 2.2 Transformer Model with Factorization
class TransFact(nn.Module):
    def __init__(self, args):
        super(TransFact, self).__init__()
        self.args = args
        self.encoder = EncoderTrans(args=args)
        self.inc_activation = nn.Sigmoid()

        # For weight computation in final speed prediction
        # Option 2.1 - Learnable rec_weight and nonrec_weight
        self.sum_weight = nn.Parameter(data=torch.tensor(1.0))

        # Option 2.2 - MLP (doesn't perform as well as Option 1)
        # self.out_processing = OutputProcessing(dim_list=[2,16,16,32,32,16,16,1]) 

        # Option 3 - Theshold of Incident Prediction
        self.threshold = torch.nn.Threshold(threshold=-0.5, value=-1.0)

        self.LR_decoder = DecoderTrans(args=args)  # decoder for incident prediction (logistic regression)
        self.rec_decoder = DecoderTrans(args=args)  # decoder for speed prediction in recurrent scenario
        self.nonrec_decoder = DecoderTrans(args=args)  # decoder for speed prediction in non-recurrent scenario

        self.args = args
    
    def forward(self, x, target, mode):
        '''
        INPUTs
            x: input, (batch_size, seq_len_in, dim_in)
            target: (batch_size, seq_len_out+1, dim_out, 5), the last dimension refers to 1~3: speed (all, truck, pv); 4: incident status; 5: Waze incident status
            mode: string of value "train" or "eval", denoting the mode to control decoder

        OUTPUTs
            LR_out: incident status prediction, (batch_size, seq_len_out, dim_out)
            speed_pred: speed prediction, (batch_size, seq_len_out, dim_out or 3*dim_out)
        '''

        # pass into encoder
        enc_out = self.encoder(x)  # (batch_size, seq_len_in, dim_in)

        # pass into decoder
        '''
        xxx_dec_out: (batch_size, seq_len_out, dim_out)  
        '''
        if self.args.task == "LR":
            return self.LR_decoder(target[..., 3], enc_out, mode), 0, 0 # Be careful! LR_out hasn't gone through nn.Sigmoid and doesn't denote the probability
            
        elif self.args.task == "rec":
            return self.rec_decoder(target[..., 0], enc_out, mode), 0, 0
        elif self.args.task == "nonrec":
            return self.nonrec_decoder(target[..., 0], enc_out, mode), 0, 0
        else:
            LR_out = self.LR_decoder(target[..., 3], enc_out, mode)
            inc_pred = self.inc_activation(LR_out) # convert logits to probabilities
            rec_out = self.rec_decoder(target[..., 0], enc_out, mode)
            nonrec_out = self.nonrec_decoder(target[..., 0], enc_out, mode)

            # generate final speed prediction
            if self.args.use_gt_inc:
                # Option 1 - Use Threshold for Incident Ground Truth (for inference ONLY)
                rec_weight = target[:, 1:, :, 3] < 0.5
                nonrec_weight = target[:, 1:, :, 3] >= 0.5
            else:
                # Option 2 - Use Expectation
                if self.args.use_expectation:
                    # Option 2.1 - Learnable rec_weight and nonrec_weight
                    nonrec_weight = torch.clamp(self.sum_weight * inc_pred, 0, 1) # torch.clamp ensures weights are within [0,1]
                    rec_weight = torch.ones(LR_out.size()).to(self.args.device) - nonrec_weight

                    # For Option 2.2 - MLP
                    # nonrec_weight = inc_pred
                    # rec_weight = torch.ones(LR_out.size()).to(self.args.device) - nonrec_weight
                
                # Option 3 - Use Threshold for Incident Prediction
                else:
                    # The following 2 lines of code are not differentiable, therefore failing to update inc_pred through MSE loss of speed_pred.
                    rec_weight = inc_pred < self.args.inc_threshold 
                    nonrec_weight = inc_pred >= self.args.inc_threshold

                    # The following 2 lines of commented code are differentiable, but in fact generates the exact same result as above two lines of code do.
                    # rec_weight = torch.clamp(self.threshold(-inc_pred)+1, max=0.5)*2   # generate 0-1 mask as what "inc_pred < self.args.inc_threshold" does
                    # nonrec_weight = torch.ones(LR_out.size()).to(self.args.device) - rec_weight  # generate 0-1 mask as what "inc_pred >= self.args.inc_threshold" does

            speed_pred = rec_out * rec_weight + nonrec_out * nonrec_weight

            # For Expectation Option 2 - MLP
            # intermediate_spd_pred = torch.stack(tensors=[rec_out * rec_weight, nonrec_out * nonrec_weight], dim=-1)
            # speed_pred = self.out_processing(intermediate_spd_pred).view(intermediate_spd_pred.shape[:3])
            
            return speed_pred, LR_out, 0

# 2.3 Transformer of Naive Combination
class TransFactNaive(nn.Module):
    def __init__(self, args):
        super(TransFactNaive, self).__init__()
        self.args = args
        self.inc_activation = nn.Sigmoid()
        self.LR_encoder = EncoderTrans(args=args)
        self.rec_encoder = EncoderTrans(args=args)
        self.nonrec_encoder = EncoderTrans(args=args)

        self.LR_decoder = DecoderTrans(args=args)  # decoder for incident prediction (logistic regression)
        self.rec_decoder = DecoderTrans(args=args)  # decoder for speed prediction in recurrent scenario
        self.nonrec_decoder = DecoderTrans(args=args)  # decoder for speed prediction in non-recurrent scenario
    
    def forward(self, x, target, mode):
        '''
        INPUTs
            x: input, (batch_size, seq_len_in, dim_in)
            target: (batch_size, seq_len_out+1, dim_out, 5), the last dimension refers to 1~3: speed (all, truck, pv); 4: incident status; 5: Waze incident status
            mode: string of value "train" or "eval", denoting the mode to control decoder

        OUTPUTs
            LR_out: incident status prediction, (batch_size, seq_len_out, dim_out)
            speed_pred: speed prediction, (batch_size, seq_len_out, dim_out or 3*dim_out)
        '''

        # pass into encoder
        LR_enc_out = self.LR_encoder(x)  # (batch_size, seq_len_in, dim_in)
        rec_enc_out = self.rec_encoder(x)  # (batch_size, seq_len_in, dim_in)
        nonrec_enc_out = self.nonrec_encoder(x)  # (batch_size, seq_len_in, dim_in)

        # pass into decoder
        '''
        xxx_out: (batch_size, seq_len_out, dim_out)  
        '''
        LR_out = self.LR_decoder(target[..., 3], LR_enc_out, mode)
        inc_pred = self.inc_activation(LR_out)
        rec_out = self.rec_decoder(target[..., 0], rec_enc_out, mode)
        nonrec_out = self.nonrec_decoder(target[..., 0], nonrec_enc_out, mode)

        # generate final speed prediction
        if self.args.use_gt_inc:
            rec_weight = target[:, 1:, :, 3] < 0.5
            nonrec_weight = target[:, 1:, :, 3] >= 0.5
        else:
            if self.args.use_expectation:
                rec_weight = torch.ones(LR_out.size()).to(self.args.device) - inc_pred
                nonrec_weight = torch.ones(LR_out.size()).to(self.args.device) - rec_weight
            else:
                rec_weight = inc_pred < self.args.inc_threshold
                nonrec_weight = inc_pred >= self.args.inc_threshold

        speed_pred = rec_out * rec_weight + nonrec_out * nonrec_weight
        
        return speed_pred


###############################################
#     3. GTrans (GCN-Transformer) Modules     #
###############################################
# 3.1 GCN-Transformer Model without Factorization
class GTransNoFact(nn.Module):
    def __init__(self, args):
        super(GTransNoFact, self).__init__()
        self.encoder = EncoderGTrans(args)
        # self.decoder = DecoderGTrans(args)
        self.decoder = DecoderTrans(args)
        self.args = args
    
    def forward(self, x, target, mode):
        '''
        INPUTs
            x: input, (batch_size, seq_len_in, num_node, dim_in)
            target: (batch_size, seq_len_out+1, num_node, 5), the last dimension refers to 1~3: speed (all, truck, pv); 4: incident status; 5: Waze incident status
            mode: string of value "train" or "eval", denoting the mode to control decoder

        OUTPUTs
            dec_out: (batch_size, seq_len_out, dim_out or 3*dim_out)  
        '''
        enc_out = self.encoder(x)  # (batch_size, seq_len_in, dim_hidden)
        dec_out = self.decoder(target[..., 0], enc_out, mode)

        return dec_out, 0, 0


# 3.2 GCN-Transformer Model with Factorization
class GTransFact(nn.Module):
    def __init__(self, args):
        super(GTransFact, self).__init__()
        self.args = args
        self.inc_activation = nn.Sigmoid()
        self.encoder = EncoderGTrans(args=args)

        self.LR_decoder = DecoderTrans(args=args)
        self.rec_decoder = DecoderTrans(args=args)
        self.nonrec_decoder = DecoderTrans(args=args)

        #self.LR_decoder = DecoderGTrans(args=args)  # decoder for incident prediction (logistic regression)
        #self.rec_decoder = DecoderGTrans(args=args)  # decoder for speed prediction in recurrent scenario
        #self.nonrec_decoder = DecoderGTrans(args=args)  # decoder for speed prediction in non-recurrent scenario
    
    def forward(self, x, target, mode):
        '''
        INPUTs
            x: input, (batch_size, seq_len_in, num_node, dim_in)
            target: (batch_size, seq_len_out+1, num_node, 5), the last dimension refers to 1~3: speed (all, truck, pv); 4: incident status; 5: Waze incident status
            mode: string of value "train" or "eval", denoting the mode to control decoder

        OUTPUTs
            LR_out: incident status prediction, (batch_size, seq_len_out, dim_out)
            speed_pred: speed prediction, (batch_size, seq_len_out, dim_out or 3*dim_out)
        '''

        # pass into encoder
        enc_out = self.encoder(x)
        batch_size, seq_len_in, num_node, dim_in = enc_out.shape  # (batch_size, seq_len_in, num_node, dim_in)
        enc_out = enc_out.reshape(batch_size, seq_len_in, num_node*dim_in)

        # pass into decoder
        '''
        xxx_dec_out: (batch_size, seq_len_out, dim_out)  
        '''
        if self.args.task == "LR":
            return self.LR_decoder(target[..., 3   ], enc_out, mode), 0, 0
        elif self.args.task == "rec":
            return self.rec_decoder(target[..., 0], enc_out, mode), 0, 0
        elif self.args.task == "nonrec":
            return self.nonrec_decoder(target[..., 0], enc_out, mode), 0, 0
        else:
            LR_out = self.LR_decoder(target[..., 3], enc_out, mode)
            inc_pred = self.inc_activation(LR_out)
            rec_out = self.rec_decoder(target[..., 0], enc_out, mode)
            nonrec_out = self.nonrec_decoder(target[..., 0], enc_out, mode)

            # generate final speed prediction
            if self.args.use_gt_inc:
                rec_weight = target[:, 1:, :, 3] < 0.5
                nonrec_weight = target[:, 1:, :, 3] >= 0.5
            else:
                if self.args.use_expectation:
                    rec_weight = torch.ones(LR_out.size()).to(self.args.device) - inc_pred
                    nonrec_weight = torch.ones(LR_out.size()).to(self.args.device) - rec_weight
                else:
                    rec_weight = inc_pred < self.args.inc_threshold
                    nonrec_weight = inc_pred >= self.args.inc_threshold

            speed_pred = rec_out * rec_weight + nonrec_out * nonrec_weight
            
            return speed_pred, LR_out, 0