import numpy as np
import pandas as pd
import math
import random
import argparse
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import os
import logging
import time as Time
from utility import pad_history,calculate_hit,extract_axis_1
from collections import Counter
from Modules_ori import *

from consistency_models import * 
from utils import *

logging.getLogger().setLevel(logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser(description="Run supervised GRU.")

    parser.add_argument('--data', nargs='?', default='yc',
                        help='yc, ks, zhihu')
    parser.add_argument('--random_seed', type=int, default=100,
                        help='random seed')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--layers', type=int, default=1,
                        help='gru_layers')
    parser.add_argument('--hidden_factor', type=int, default=64,
                        help='Number of hidden factors, i.e., embedding size.')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='Learning rate.')
    parser.add_argument('--l2_decay', type=float, default=0,
                        help='l2 loss reg coef.')
    parser.add_argument('--cuda', type=int, default=2,
                        help='cuda device.')
    parser.add_argument('--dropout_rate', type=float, default=0.1,
                        help='dropout ')
    parser.add_argument('--report_epoch', type=bool, default=True,
                        help='report frequency')
    parser.add_argument('--diffuser_type', type=str, default='mlp1',
                        help='type of diffuser.')
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='type of optimizer.')
    parser.add_argument('--descri', type=str, default='',
                        help='description of the work.')
    parser.add_argument('--sigma_min', type=float, default=0.002,
                        help='Minimum standard deviation of the noise.')
    parser.add_argument('--sigma_max', type=float, default=80.0,
                        help='Maximum standard deviation of the noise.')
    parser.add_argument('--rho', type=float, default=7.0,
                        help=' Schedule hyper-parameter.')
    parser.add_argument('--sigma_data', type=float, default=0.5,
                        help='Standard deviation of the data.')
    parser.add_argument('--initial_timesteps', type=int, default=10,
                        help='Schedule timesteps at the start of training.')
    parser.add_argument('--final_timesteps', type=int, default=1280,
                        help='Schedule timesteps at the end of training.')
    parser.add_argument('--loss_type', type=str, default='l2',
                        help='loss type.')    
    parser.add_argument('--total_training_step', type=int, default=1000,
                        help='total training step.')
    return parser.parse_args()

args = parse_args()

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

setup_seed(args.random_seed)
torch.pi = math.pi


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def linear_beta_schedule(timesteps, beta_start, beta_end):
    beta_start = beta_start
    beta_end = beta_end
    return torch.linspace(beta_start, beta_end, timesteps)

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def exp_beta_schedule(timesteps, beta_min=0.1, beta_max=10):
    x = torch.linspace(1, 2 * timesteps + 1, timesteps)
    betas = 1 - torch.exp(- beta_min / timesteps - x * 0.5 * (beta_max - beta_min) / (timesteps * timesteps))
    return betas

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

def generate_sigma_list(start_value, num, decay_rate, decay_style='linear'):
    sigma_list = []
    for i in range(num):
        if decay_style == 'linear':
            value = start_value - decay_rate * i
        elif decay_style == 'exp':
            value = start_value * math.exp(-decay_rate * i)
        sigma_list.append(value)
    return sigma_list


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class NoiseLevelEmbedding(nn.Module):
    def __init__(self, channels: int, scale: float = 16.0) -> None:
        super().__init__()

        self.W = nn.Parameter(torch.randn(channels // 2) * scale, requires_grad=False)

        self.projection = nn.Sequential(
            nn.Linear(channels, 4 * channels),
            nn.SiLU(),
            nn.Linear(4 * channels, channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        h = x[:, None] * self.W[None, :] * 2 * torch.pi
        h = torch.cat([torch.sin(h), torch.cos(h)], dim=-1)

        return self.projection(h)
        
class Tenc(nn.Module):
    def __init__(self, hidden_size, item_num, state_size, dropout, diffuser_type, device, num_heads=1):
        super(Tenc, self).__init__()
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.item_num = int(item_num)
        self.dropout = nn.Dropout(dropout)
        self.diffuser_type = diffuser_type
        self.device = device
        self.item_embeddings = nn.Embedding(
            num_embeddings=item_num + 1,
            embedding_dim=hidden_size,
        )
        nn.init.normal_(self.item_embeddings.weight, 0, 1)
        self.none_embedding = nn.Embedding(
            num_embeddings=1,
            embedding_dim=self.hidden_size,
        )
        nn.init.normal_(self.none_embedding.weight, 0, 1)
        self.positional_embeddings = nn.Embedding(
            num_embeddings=state_size,
            embedding_dim=hidden_size
        )
        # emb_dropout is added
        self.emb_dropout = nn.Dropout(dropout)
        self.ln_1 = nn.LayerNorm(hidden_size)
        self.ln_2 = nn.LayerNorm(hidden_size)
        self.ln_3 = nn.LayerNorm(hidden_size)
        self.mh_attn = MultiHeadAttention(hidden_size, hidden_size, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(hidden_size, hidden_size, dropout)
        self.s_fc = nn.Linear(hidden_size, item_num)


        self.step_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size*2),
            nn.GELU(),
            nn.Linear(self.hidden_size*2, self.hidden_size),
        )

        self.noise_mlp = NoiseLevelEmbedding(self.hidden_size)

        self.emb_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.hidden_size*2)
        )

        self.diff_mlp = nn.Sequential(
            nn.Linear(self.hidden_size * 3, self.hidden_size*2),
            nn.GELU(),
            nn.Linear(self.hidden_size*2, self.hidden_size),
        )

        if self.diffuser_type =='mlp1':
            self.diffuser = nn.Sequential(
                nn.Linear(self.hidden_size*3, self.hidden_size)
            )
        elif self.diffuser_type =='mlp2':
            self.diffuser = nn.Sequential(
            nn.Linear(self.hidden_size * 3, self.hidden_size*2),
            nn.GELU(),
            nn.Linear(self.hidden_size*2, self.hidden_size)
        )

    def forward(self, x, sigma, h):
        sigma = self.noise_mlp(sigma)
        if self.diffuser_type == 'mlp1':
            res = self.diffuser(torch.cat((x, h, sigma), dim=1))
        elif self.diffuser_type == 'mlp2':
            res = self.diffuser(torch.cat((x, h, sigma), dim=1))
        return res

    def forward_uncon(self, x, step):
        h = self.none_embedding(torch.tensor([0]).to(self.device))
        h = torch.cat([h.view(1, 64)]*x.shape[0], dim=0)

        t = self.step_mlp(step)

        if self.diffuser_type == 'mlp1':
            res = self.diffuser(torch.cat((x, h, t), dim=1))
        elif self.diffuser_type == 'mlp2':
            res = self.diffuser(torch.cat((x, h, t), dim=1))
            
        return res

        # return x

    def cacu_x(self, x):
        x = self.item_embeddings(x)

        return x

    def cacu_h(self, states, len_states, p=0.1):
        #hidden
        inputs_emb = self.item_embeddings(states)
        inputs_emb += self.positional_embeddings(torch.arange(self.state_size).to(self.device))
        seq = self.emb_dropout(inputs_emb)
        mask = torch.ne(states, self.item_num).float().unsqueeze(-1).to(self.device)
        seq *= mask
        seq_normalized = self.ln_1(seq)
        mh_attn_out = self.mh_attn(seq_normalized, seq)
        ff_out = self.feed_forward(self.ln_2(mh_attn_out))
        ff_out *= mask
        ff_out = self.ln_3(ff_out)
        state_hidden = extract_axis_1(ff_out, len_states - 1)
        h = state_hidden.squeeze()

        # B, D = h.shape[0], h.shape[1]
        # mask1d = (torch.sign(torch.rand(B) - p) + 1) / 2
        # maske1d = mask1d.view(B, 1)
        # mask = torch.cat([maske1d] * D, dim=1)
        # mask = mask.to(self.device)

        # print(h.device, self.none_embedding(torch.tensor([0]).to(self.device)).device, mask.device)
        # h = h * mask + self.none_embedding(torch.tensor([0]).to(self.device)) * (1-mask)


        return h  
    
    def predict(self, states, len_states, consistency_sampler, sigma_style='linear', sigma_num=10):
        #hidden
        inputs_emb = self.item_embeddings(states)
        inputs_emb += self.positional_embeddings(torch.arange(self.state_size).to(self.device))
        seq = self.emb_dropout(inputs_emb)
        mask = torch.ne(states, self.item_num).float().unsqueeze(-1).to(self.device)
        seq *= mask
        seq_normalized = self.ln_1(seq)
        mh_attn_out = self.mh_attn(seq_normalized, seq)
        ff_out = self.feed_forward(self.ln_2(mh_attn_out))
        ff_out *= mask
        ff_out = self.ln_3(ff_out)
        state_hidden = extract_axis_1(ff_out, len_states - 1)
        h = state_hidden.squeeze()

        # x = diff.sample(self.forward, self.forward_uncon, h)
        x = self.sample(consistency_sampler, h, sigmas_style=sigma_style, sigma_num=sigma_num)
        
        test_item_emb = self.item_embeddings.weight
        scores = torch.matmul(x, test_item_emb.transpose(0, 1))

        return scores

    @torch.no_grad()
    def sample(self, consistency_sampler, h, sigmas_style='linear', sigma_num=10):
        # if sigmas_style == 'linear':
        #     sigma_list = generate_sigma_list(start_value=args.sigma_max, num=sigma_num, decay_rate=(args.sigma_max-args.sigma_min)/sigma_num, decay_style='linear')
        # elif sigmas_style == 'exp':
        #     sigma_list = generate_sigma_list(start_value=args.sigma_max, num=sigma_num, decay_rate=0.2, decay_style='exp')
        num_timesteps = improved_timesteps_schedule(
            current_training_step,
            args.total_training_step,
            args.initial_timesteps,
            args.final_timesteps,
        )
        sigmas = karras_schedule(
            num_timesteps, self.sigma_min, self.sigma_max, self.rho, h.device
        )[:sigma_num]
        print(sigmas)
        samples = consistency_sampler(
            model=self,
            x_initial=torch.randn_like(h) * args.sigma_max,
            sigmas=sigmas,
            clip_denoised=False,
            sequence_rep=h
        )
        return samples

def evaluate(model, test_data, device, consistency_sampler, sigma_style='linear', sigma_num=10):
    eval_data=pd.read_pickle(os.path.join(data_directory, test_data))

    batch_size = 100
    evaluated=0
    total_clicks=1.0
    total_purchase = 0.0
    total_reward = [0, 0, 0, 0]
    hit_clicks=[0,0,0,0]
    ndcg_clicks=[0,0,0,0]
    hit_purchase=[0,0,0,0]
    ndcg_purchase=[0,0,0,0]

    seq, len_seq, target = list(eval_data['seq'].values), list(eval_data['len_seq'].values), list(eval_data['next'].values)


    num_total = len(seq)

    for i in range(num_total // batch_size):
        seq_b, len_seq_b, target_b = seq[i * batch_size: (i + 1)* batch_size], len_seq[i * batch_size: (i + 1)* batch_size], target[i * batch_size: (i + 1)* batch_size]
        states = np.array(seq_b)
        states = torch.LongTensor(states)
        states = states.to(device)

        prediction = model.predict(states, np.array(len_seq_b), consistency_sampler, sigma_style, sigma_num)
        _, topK = prediction.topk(100, dim=1, largest=True, sorted=True)
        topK = topK.cpu().detach().numpy()
        sorted_list2=np.flip(topK,axis=1)
        sorted_list2 = sorted_list2
        calculate_hit(sorted_list2,topk,target_b,hit_purchase,ndcg_purchase)

        total_purchase+=batch_size
 

    hr_list = []
    ndcg_list = []
    print('sigma style: ', sigma_style, 'sigma num: ', sigma_num)
    print('{:<10s} {:<10s} {:<10s} {:<10s} {:<10s} {:<10s}'.format('HR@'+str(topk[0]), 'NDCG@'+str(topk[0]), 'HR@'+str(topk[1]), 'NDCG@'+str(topk[1]), 'HR@'+str(topk[2]), 'NDCG@'+str(topk[2])))
    for i in range(len(topk)):
        hr_purchase=hit_purchase[i]/total_purchase
        ng_purchase=ndcg_purchase[i]/total_purchase

        hr_list.append(hr_purchase)
        ndcg_list.append(ng_purchase[0,0])

        if i == 1:
            hr_20 = hr_purchase
            ndcg_20 = ng_purchase[0,0]

    print('{:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f}'.format(hr_list[0], (ndcg_list[0]), hr_list[1], (ndcg_list[1]), hr_list[2], (ndcg_list[2])))

    return hr_20, ndcg_20


if __name__ == '__main__':

    # args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    data_directory = './data/' + args.data
    data_statis = pd.read_pickle(
        os.path.join(data_directory, 'data_statis.df'))  # read data statistics, includeing seq_size and item_num
    seq_size = data_statis['seq_size'][0]  # the length of history to define the seq
    item_num = data_statis['item_num'][0]  # total number of items
    topk=[10, 20, 50]

    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    model = Tenc(args.hidden_factor,item_num, seq_size, args.dropout_rate, args.diffuser_type, device)

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-8, weight_decay=args.l2_decay)
    elif args.optimizer =='adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, eps=1e-8, weight_decay=args.l2_decay)
    elif args.optimizer =='adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr, eps=1e-8, weight_decay=args.l2_decay)
    elif args.optimizer =='rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, eps=1e-8, weight_decay=args.l2_decay)

    if args.loss_type == 'l1':
        loss_fn = F.l1_loss
    elif args.loss_type == 'l2':
        loss_fn = F.mse_loss
    elif args.loss_type == "huber":
        loss_fn = F.smooth_l1_loss
    else:
        raise NotImplementedError()


    # scheduler = lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1, total_iters=20)
    
    model.to(device)
    # teacher_model = Tenc(args.hidden_factor,item_num, seq_size, args.dropout_rate, args.diffuser_type, device)
    # teacher_model.to(device)())
    # for param in teacher_model.parameters():
    #     param.requires_grad = False
    # teacher_model.load_state_dict(model.state_dict
    # teacher_model = teacher_model.eval()

    train_data = pd.read_pickle(os.path.join(data_directory, 'train_data.df'))

    num_rows=train_data.shape[0]
    num_batches=int(num_rows/args.batch_size)

    improved_consistency_training = ImprovedConsistencyTraining(
        sigma_min = args.sigma_min,
        sigma_max = args.sigma_max,
        rho = args.rho,
        sigma_data= args.sigma_data,
        initial_timesteps = args.initial_timesteps,
        final_timesteps = args.final_timesteps,
    )

    best_epoch = 0
    best_hr_20, best_ndcg_20 = 0., 0.
    counter = 0  # counter for stopping early
    
    for current_training_step in range(args.total_training_step):
        step_loss = 0.0
        start_time = Time.time()
        for j in range(num_batches):
            batch = train_data.sample(n=args.batch_size).to_dict()
            seq = list(batch['seq'].values())
            len_seq = list(batch['len_seq'].values())
            target=list(batch['next'].values())

            optimizer.zero_grad()
            seq = torch.LongTensor(seq)
            len_seq = torch.LongTensor(len_seq)
            target = torch.LongTensor(target)

            seq = seq.to(device)
            target = target.to(device)
            len_seq = len_seq.to(device)


            x_start = model.cacu_x(target)  # e_n^0

            h = model.cacu_h(seq, len_seq) # c_{n-1}
            output = improved_consistency_training(
                model=model, 
                x=x_start, 
                current_training_step=current_training_step, 
                total_training_steps=args.total_training_step,
                h=h
            )
            loss = (pseudo_huber_loss(output.predicted, output.target) * output.loss_weights).mean()

            loss.backward()
            optimizer.step()
            step_loss += loss.item()

        
        if args.report_epoch:
            if current_training_step % 1 == 0:
                print("Epoch {:03d}; ".format(current_training_step) + 'Train loss: {:.4f}; '.format(step_loss) + "Time cost: " + Time.strftime(
                        "%H: %M: %S", Time.gmtime(Time.time()-start_time)))

            if (current_training_step + 1) % 5 == 0:
                consistency_sampler = ConsistencySamplingAndEditing(
                                        sigma_min=args.sigma_min,
                                        sigma_data=args.sigma_data,
                                    )
                eval_start = Time.time()
                print('-------------------------- VAL PHRASE --------------------------')
                _ = evaluate(model, 'val_data.df', device, consistency_sampler)
                print('-------------------------- TEST PHRASE -------------------------')
                for sigma_num in [1,2,5, 10]:
                    hr_20, ndcg_20 = evaluate(model, 'test_data.df', device, consistency_sampler, sigma_style='linear', sigma_num=sigma_num)
                    if hr_20 > best_hr_20: 
                        counter = 0
                        best_hr_20 = hr_20
                        best_ndcg_20 = ndcg_20
                        best_epoch = current_training_step
                    hr_20, ndcg_20 = evaluate(model, 'test_data.df', device, consistency_sampler, sigma_style='exp', sigma_num=sigma_num)
                    if hr_20 > best_hr_20:
                        counter = 0
                        best_hr_20 = hr_20
                        best_ndcg_20 = ndcg_20
                        best_epoch = current_training_step
                print("Evalution cost: " + Time.strftime("%H: %M: %S", Time.gmtime(Time.time()-eval_start)))
                print('----------------------------------------------------------------')


    print('Best epoch: ', best_epoch, 'Best HR@20: ', best_hr_20, 'Best NDCG@20: ', best_ndcg_20)



                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     

