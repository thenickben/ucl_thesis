import sys, shutil, os
import numpy as np
import torch
import random
import copy
import json, yaml
import logging
import time, datetime
import seaborn as sns
import pandas as pd
from abc import ABC
import IPython
from pathlib import Path as P
import bs4 as bs
from pprint import pprint
from pprint import pprint
from torch import nn
from torch import multiprocessing as mp
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.optim as optim


from transformers import BertForSequenceClassification, AdamW, BertConfig, BertTokenizer
from core.policy.vector.vector_multiwoz import MultiWozVector

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=UserWarning) 
pd.set_option("display.max_rows", None, "display.max_columns", None)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MultiDiscretePolicy(nn.Module):
    def __init__(self, s_dim, h_dim, a_dim):
        super(MultiDiscretePolicy, self).__init__()
        
        self.net = nn.Sequential(nn.Linear(s_dim, h_dim),
                                 nn.ReLU(),
                                 nn.Linear(h_dim, h_dim),
                                 nn.ReLU(),
                                 nn.Linear(h_dim, a_dim))

    def forward(self, s):
        # [b, s_dim] => [b, a_dim]
        a_weights = self.net(s)

        return a_weights
    
    def select_action(self, s, sample=True):
        """
        :param s: [s_dim]
        :return: [a_dim]
        """
        # forward to get action probs
        # [s_dim] => [a_dim]
        a_weights = self.forward(s)
        a_probs = torch.sigmoid(a_weights)
        
        # [a_dim] => [a_dim, 2]
        a_probs = a_probs.unsqueeze(1)
        a_probs = torch.cat([1-a_probs, a_probs], 1)
        a_probs = torch.clamp(a_probs, 1e-10, 1 - 1e-10)
        
        # [a_dim, 2] => [a_dim]
        a = a_probs.multinomial(1).squeeze(1) if sample else a_probs.argmax(1)
        
        return a
    
    def get_log_prob(self, s, a):
        """
        :param s: [b, s_dim]
        :param a: [b, a_dim]
        :return: [b, 1]
        """
        # forward to get action probs
        # [b, s_dim] => [b, a_dim]
        a_weights = self.forward(s)
        a_probs = torch.sigmoid(a_weights)
        
        # [b, a_dim] => [b, a_dim, 2]
        a_probs = a_probs.unsqueeze(-1)
        a_probs = torch.cat([1-a_probs, a_probs], -1)
        
        # [b, a_dim, 2] => [b, a_dim]
        trg_a_probs = a_probs.gather(-1, a.unsqueeze(-1).long()).squeeze(-1)
        log_prob = torch.log(trg_a_probs)
        
        return log_prob.sum(-1, keepdim=True)


class Value(nn.Module):
    def __init__(self, s_dim, hv_dim):
        super(Value, self).__init__()

        self.net = nn.Sequential(nn.Linear(s_dim, hv_dim),
                                 nn.ReLU(),
                                 nn.Linear(hv_dim, hv_dim),
                                 nn.ReLU(),
                                 nn.Linear(hv_dim, 1))

    def forward(self, s):
        """
        :param s: [b, s_dim]
        :return:  [b, 1]
        """
        value = self.net(s)

        return value

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'mask'))

class Memory(object):

    def __init__(self):
        self.memory = []

    def push(self, *args):
        """Saves a transition."""
        self.memory.append(Transition(*args))

    def get_batch(self, batch_size=None):
        if batch_size is None:
            return Transition(*zip(*self.memory))
        else:
            random_batch = random.sample(self.memory, batch_size)
            return Transition(*zip(*random_batch))

    def append(self, new_memory):
        self.memory += new_memory.memory

    def __len__(self):
        return len(self.memory)

# abstract class for vectoriser
class Vector():

    def __init__(self):
        pass

    def generate_dict(self):
        """init the dict for mapping state/action into vector"""

    def state_vectorize(self, state):
        """vectorize a state
        Args:
            state (tuple):
                Dialog state
        Returns:
            state_vec (np.array):
                Dialog state vector
        """
        raise NotImplementedError

    def action_devectorize(self, action_vec):
        """recover an action
        
        Args:
            action_vec (np.array):
                Dialog act vector
        Returns:
            action (tuple):
                Dialog act
        """
        raise NotImplementedError

# abstract module class -->policy
class Module(ABC):

    def train(self, *args, **kwargs):
        """Model training entry point"""
        pass

    def test(self, *args, **kwargs):
        """Model testing entry point"""
        pass

    def from_cache(self, *args, **kwargs):
        """restore internal state for multi-turn dialog"""
        return None

    def to_cache(self, *args, **kwargs):
        """save internal state for multi-turn dialog"""
        return None

    def init_session(self):
        """Init the class variables for a new session."""
        pass


class Policy(Module):
    """Policy module interface."""

    def predict(self, state):
        """Predict the next agent action given dialog state.
        Args:
            state (dict or list of list):
                when the policy takes dialogue state as input, the type is dict.
                else when the policy takes dialogue act as input, the type is list of list.
        Returns:
            action (list of list or str):
                when the policy outputs dialogue act, the type is list of list.
                else when the policy outputs utterance directly, the type is str.
        """
        return []

class PPO(Policy):

    def __init__(self):

        self.update_round = update_round
        self.optim_batchsz = optim_batchsz
        self.gamma = gamma
        self.epsilon = surrogate_clip
        self.tau = tau
        self.policy_lr = policy_lr
        self.value_lr = value_lr

        # featurizer
        voc_file = os.path.join(path_convlab, 'data/multiwoz/sys_da_voc.txt')
        voc_opp_file = os.path.join(path_convlab, 'data/multiwoz/usr_da_voc.txt')
        self.vector = MultiWozVector(voc_file, voc_opp_file)

        # construct policy and value network
        self.policy = MultiDiscretePolicy(self.vector.state_dim, h_dim, self.vector.da_dim).to(device=DEVICE)
        self.value = Value(self.vector.state_dim, hv_dim).to(device=DEVICE)

        # optimizers
        self.policy_optim = optim.AdamW(self.policy.parameters(), lr=self.policy_lr)
        self.value_optim = optim.AdamW(self.value.parameters(), lr=self.value_lr)

        # load pre-trained policy net
        load_mle(self.policy)    

    def predict(self, state):
        """
        Predict an system action given state.
        Args:
            state (dict): Dialog state. Please refer to util/state.py
        Returns:
            action : System act, with the form of (act_type, {slot_name_1: value_1, slot_name_2, value_2, ...})
        """
        s_vec = torch.Tensor(self.vector.state_vectorize(state))
        a = self.policy.select_action(s_vec.to(device=DEVICE), False).cpu()
        action = self.vector.action_devectorize(a.numpy())
        state['system_action'] = action
        return action

    def init_session(self):
        """
        Restore after one session
        """
        pass
    
    def est_adv(self, r, v, mask):
        """
        we save a trajectory in continuous space and it reaches the ending of current trajectory when mask=0.
        :param r: reward, Tensor, [b]
        :param v: estimated value, Tensor, [b]
        :param mask: indicates ending for 0 otherwise 1, Tensor, [b]
        :return: A(s, a), V-target(s), both Tensor
        """
        batchsz = v.size(0)

        # v_target is worked out by Bellman equation.
        v_target = torch.Tensor(batchsz).to(device=DEVICE)
        delta = torch.Tensor(batchsz).to(device=DEVICE)
        A_sa = torch.Tensor(batchsz).to(device=DEVICE)

        prev_v_target = 0
        prev_v = 0
        prev_A_sa = 0
        for t in reversed(range(batchsz)):
            # mask here indicates a end of trajectory
            # this value will be treated as the target value of value network.
            # mask = 0 means the immediate reward is the real V(s) since it's end of trajectory.
            # formula: V(s_t) = r_t + gamma * V(s_t+1)
            v_target[t] = r[t] + self.gamma * prev_v_target * mask[t]

            # please refer to : https://arxiv.org/abs/1506.02438
            # for generalized adavantage estimation
            # formula: delta(s_t) = r_t + gamma * V(s_t+1) - V(s_t)
            delta[t] = r[t] + self.gamma * prev_v * mask[t] - v[t]

            # formula: A(s, a) = delta(s_t) + gamma * lamda * A(s_t+1, a_t+1)
            # here use symbol tau as lambda, but original paper uses symbol lambda.
            A_sa[t] = delta[t] + self.gamma * self.tau * prev_A_sa * mask[t]

            # update previous
            prev_v_target = v_target[t]
            prev_v = v[t]
            prev_A_sa = A_sa[t]

        # normalize A_sa
        A_sa = (A_sa - A_sa.mean()) / A_sa.std()

        return A_sa, v_target
    
    def update(self, epoch, batchsz, s, a, r, mask):
        # get estimated V(s) and PI_old(s, a)
        # actually, PI_old(s, a) can be saved when interacting with env, so as to save the time of one forward elapsed
        # v: [b, 1] => [b]
        v = self.value(s).squeeze(-1).detach()
        log_pi_old_sa = self.policy.get_log_prob(s, a).detach()
        
        # estimate advantage and v_target according to GAE and Bellman Equation
        A_sa, v_target = self.est_adv(r, v, mask)
        
        for i in range(self.update_round):

            # 1. shuffle current batch
            perm = torch.randperm(batchsz)
            # shuffle the variable for mutliple optimize
            v_target_shuf, A_sa_shuf, s_shuf, a_shuf, log_pi_old_sa_shuf = v_target[perm], A_sa[perm], s[perm], a[perm], \
                                                                           log_pi_old_sa[perm]

            # 2. get mini-batch for optimizing
            optim_chunk_num = int(np.ceil(batchsz / self.optim_batchsz))
            # chunk the optim_batch for total batch
            v_target_shuf, A_sa_shuf, s_shuf, a_shuf, log_pi_old_sa_shuf = torch.chunk(v_target_shuf, optim_chunk_num), \
                                                                           torch.chunk(A_sa_shuf, optim_chunk_num), \
                                                                           torch.chunk(s_shuf, optim_chunk_num), \
                                                                           torch.chunk(a_shuf, optim_chunk_num), \
                                                                           torch.chunk(log_pi_old_sa_shuf,
                                                                                       optim_chunk_num)
            # 3. iterate all mini-batch to optimize
            policy_loss, value_loss = 0., 0.
            for v_target_b, A_sa_b, s_b, a_b, log_pi_old_sa_b in zip(v_target_shuf, A_sa_shuf, s_shuf, a_shuf,
                                                                     log_pi_old_sa_shuf):
                # print('optim:', batchsz, v_target_b.size(), A_sa_b.size(), s_b.size(), a_b.size(), log_pi_old_sa_b.size())
                # 1. update value network
                self.value_optim.zero_grad()
                v_b = self.value(s_b).squeeze(-1)
                loss = (v_b - v_target_b).pow(2).mean()
                value_loss += loss.item()
                
                # backprop
                loss.backward()
                # nn.utils.clip_grad_norm(self.value.parameters(), 4)
                self.value_optim.step()

                # 2. update policy network by clipping
                self.policy_optim.zero_grad()
                # [b, 1]
                log_pi_sa = self.policy.get_log_prob(s_b, a_b)
                # ratio = exp(log_Pi(a|s) - log_Pi_old(a|s)) = Pi(a|s) / Pi_old(a|s)
                # we use log_pi for stability of numerical operation
                # [b, 1] => [b]
                ratio = (log_pi_sa - log_pi_old_sa_b).exp().squeeze(-1)
                # because the joint action prob is the multiplication of the prob of each da
                # it may become extremely small
                # and the ratio may be inf in this case, which causes the gradient to be nan
                # clamp in case of the inf ratio, which causes the gradient to be nan
                ratio = torch.clamp(ratio, 0, 10)
                surrogate1 = ratio * A_sa_b
                surrogate2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * A_sa_b
                # this is element-wise comparing.
                # we add negative symbol to convert gradient ascent to gradient descent
                surrogate = - torch.min(surrogate1, surrogate2).mean()
                policy_loss += surrogate.item()

                # backprop
                surrogate.backward()
                # although the ratio is clamped, the grad may still contain nan due to 0 * inf
                # set the inf in the gradient to 0
                for p in self.policy.parameters():
                    p.grad[p.grad != p.grad] = 0.0
                # gradient clipping, for stability
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), grad_clip)
                # optim step
                self.policy_optim.step()
            
            value_loss /= optim_chunk_num
            policy_loss /= optim_chunk_num

            

#######################################################################################################

# ICM from utterances
class ICM_UTT(nn.Module):
  def __init__(self):
        super(ICM_UTT, self).__init__()
        
        self.user_nlg = user_nlg
        self.sys_nlg = sys_nlg
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.inv_h_dim = inv_h_dim
        self.fwd_h_dim = fwd_h_dim
        self.s_enc_out_dim = s_enc_out_dim
        self.max_len = max_len
        self.classifier_only = classifier_only
        self.icm_beta = icm_beta

        # Multiwoz vector
        voc_file = os.path.join(path_convlab, 'data/multiwoz/sys_da_voc.txt')
        voc_opp_file = os.path.join(path_convlab, 'data/multiwoz/usr_da_voc.txt')
        self.vector = MultiWozVector(voc_file, voc_opp_file)

        # tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # --- state encoder ---
        self.state_encoder = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels = self.s_enc_out_dim).to(device=DEVICE)
        
        # inverse and forward models heads
        self.inv_head = nn.Sequential(nn.Linear(2 * self.s_enc_out_dim, self.inv_h_dim),
                                 nn.ReLU(),
                                 nn.Linear(self.inv_h_dim, self.inv_h_dim),
                                 nn.ReLU(),
                                 nn.Linear(self.inv_h_dim, self.a_dim)).to(device=DEVICE)
        
        self.fwd_head = nn.Sequential(nn.Linear(self.max_len + self.s_enc_out_dim, self.fwd_h_dim),
                                 nn.ReLU(),
                                 nn.Linear(self.fwd_h_dim, self.fwd_h_dim),
                                 nn.ReLU(),
                                 nn.Linear(self.fwd_h_dim, self.s_enc_out_dim)).to(device=DEVICE)

        # freeze state encoder but head
        if classifier_only:
          for name, param in self.state_encoder.named_parameters():
            if 'classifier' not in name: # classifier layer
              param.requires_grad = False

  def state_utt_tensorize(self, state_utt):
    
    state_tokens = self.tokenizer.tokenize(state_utt) 
    
    # to cover all corner cases
    if len(state_tokens) >= self.max_len:
      tokens = ["[CLS]"] + state_tokens[:(self.max_len-2)] + ["[SEP]"] + ["[PAD]"]*(self.max_len - len(state_tokens) - 2)
    elif len(state_tokens) == self.max_len - 1 :
      tokens = ["[CLS]"] + state_tokens[:(self.max_len-2)] + ["[SEP]"]
    else:
      tokens = ["[CLS]"] + state_tokens + ["[SEP]"] + ["[PAD]"] * (self.max_len - len(state_tokens) -2)

    indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)
    
    # to cover all corner cases
    if len(state_tokens) >= self.max_len:
      mask_ids = [0]*(len(state_tokens[:(self.max_len-2)])+2) + [1] *  (self.max_len - len(state_tokens) - 2)
    elif len(state_tokens) == self.max_len - 1 :
      mask_ids = [0]*(len(state_tokens)) + [1]
    else:
      mask_ids = [0]*(len(state_tokens)+2) + [1] *  (self.max_len - len(state_tokens) - 2)

    state_tokens_tensor = torch.tensor([indexed_tokens]).to(device=DEVICE)
    state_mask_tensor = torch.tensor([mask_ids]).to(device=DEVICE)

    return state_tokens_tensor, state_mask_tensor

  def action_utt_tensorize(self, action_utt):

    action_tokens = self.tokenizer.tokenize(action_utt)   
    # to cover all corner cases
    if len(action_tokens) >= self.max_len:
      tokens = ["[CLS]"] + action_tokens[:(self.max_len-2)] + ["[SEP]"] + ["[PAD]"]*(self.max_len - len(action_tokens) - 2)
    elif len(action_tokens) == self.max_len - 1 :
      tokens = ["[CLS]"] + action_tokens[:(self.max_len-2)] + ["[SEP]"]
    else:
      tokens = ["[CLS]"] + action_tokens + ["[SEP]"] + ["[PAD]"] * (self.max_len - len(action_tokens) -2)

    indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)
    
    # to cover all corner cases
    if len(action_tokens) >= self.max_len:
      mask_ids = [0]*(len(action_tokens[:(self.max_len-2)])+2) + [1] *  (self.max_len - len(action_tokens) - 2)
    elif len(action_tokens) == self.max_len - 1 :
      mask_ids = [0]*(len(action_tokens)) + [1]
    else:
      mask_ids = [0]*(len(action_tokens)+2) + [1] *  (self.max_len - len(action_tokens) - 2)

    action_tokens_tensor = torch.tensor([indexed_tokens]).to(device=DEVICE)
    action_mask_tensor = torch.tensor([mask_ids]).to(device=DEVICE)

    return action_tokens_tensor, action_mask_tensor
  
  def forward(self, state_utt, action_da, next_state_utt):
        
        # utterize (?) action
        action_utt = self.user_nlg.generate(action_da)

        # vectorize action
        action_vec = self.vector.action_vectorize(action_da)

        # tensorize
        state_tokens_tensor, state_mask_tensor = self.state_utt_tensorize(state_utt)
        action_tokens_tensor, action_mask_tensor = self.action_utt_tensorize(action_utt)
        next_state_tokens_tensor, next_state_mask_tensor = self.state_utt_tensorize(next_state_utt)

        # get encodings
        phi_state = self.state_encoder(state_tokens_tensor, state_mask_tensor)[0].squeeze(0)
        phi_next_state = self.state_encoder(next_state_tokens_tensor, next_state_mask_tensor)[0].squeeze(0)

        # --- Inverse model pass ---
        # concat both encodings
        phi_concat = torch.cat((phi_state,phi_next_state),0)

        # pass thru inverse model
        action_vec_est = torch.sigmoid(self.inv_head(phi_concat))

        # --- Forward model pass ---
        # concat state encoding with action token tensors
        phi_s_a_concat = torch.cat((phi_state, action_tokens_tensor.squeeze(0)),0)

        # pass thru forward model
        phi_next_state_est = self.fwd_head(phi_s_a_concat)

        return action_vec, action_vec_est, phi_next_state, phi_next_state_est

  def get_intrinsic_rewards(self, user_das, sys_das, mask, eta = 0.01):
    
    intrinsic_rewards = []

    # iterate over batch and form tuples of (state, action, next_state)
    for i in range(len(user_das)-1):
      
      # not count if end of dialogue
      if mask[i] == 0.0:
        intrinsic_rewards.append(0.0)
      
      # get tuple
      state_utt = self.user_nlg.generate(user_das[i])
      
      action_da = sys_das[i+1]

      next_state_utt = self.user_nlg.generate(user_das[i+1])

      # get estimates from inverse and forward models
      _, _, phi_next_state, phi_next_state_est = self.forward(state_utt, action_da, next_state_utt)

      # extrinsic reward
      intrinsic_reward = (eta / 2.0) * torch.mean((phi_next_state_est - phi_next_state) ** 2)

      # append to list
      intrinsic_rewards.append(intrinsic_reward.item())

    return intrinsic_rewards

  def compute_loss(self, user_das, sys_das, mask):

    loss = torch.tensor(0.).to(device = DEVICE)

    # iterate over batch and form tuples of (state, action, next_state)
    for i in range(len(user_das)-1):
      
      # not count if end of dialogue
      if mask[i] == 0.0:
        continue
      
      # get tuple
      state_utt = self.user_nlg.generate(user_das[i])
      
      action_da = sys_das[i+1]

      next_state_utt = self.user_nlg.generate(user_das[i+1])

      # get estimates from inverse and forward models
      action_vec, action_vec_est, phi_next_state, phi_next_state_est = self.forward(state_utt, action_da, next_state_utt)

      # inverse model loss
      loss_inv = torch.mean((torch.Tensor(action_vec).to(device = DEVICE) - action_vec_est) ** 2)

      # forward model loss
      loss_fwd = torch.mean((phi_next_state_est - phi_next_state) ** 2)

      # total ICM loss weighted by icm_beta
      loss_step = (1 - self.icm_beta) * loss_inv + self.icm_beta * loss_fwd

      loss += loss_step
    
    return loss / len(user_das)

#######################################################################################################

# ICM from DAs
class IC_DA(nn.Module):
  def __init__(self):
        super(IC_DA, self).__init__()
        
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.inv_h_dim = inv_h_dim
        self.fwd_h_dim = fwd_h_dim
        self.s_enc_out_dim = s_enc_out_dim
        self.icm_beta = icm_beta

        # Multiwoz vector
        voc_file = os.path.join(path_convlab, 'data/multiwoz/sys_da_voc.txt')
        voc_opp_file = os.path.join(path_convlab, 'data/multiwoz/usr_da_voc.txt')
        self.vector = MultiWozVector(voc_file, voc_opp_file)

        # encoder
        self.state_encoder = nn.Sequential(nn.Linear(self.a_dim, self.s_enc_out_dim),
                                 nn.ReLU(),
                                 nn.Linear(self.s_enc_out_dim, self.s_enc_out_dim)).to(device=DEVICE)

        # inverse and forward models heads
        self.inv_head = nn.Sequential(nn.Linear(2 * self.s_enc_out_dim, self.inv_h_dim),
                                 nn.ReLU(),
                                 nn.Linear(self.inv_h_dim, self.inv_h_dim),
                                 nn.ReLU(),
                                 nn.Linear(self.inv_h_dim, self.a_dim)).to(device=DEVICE)
        
        self.fwd_head = nn.Sequential(nn.Linear(self.a_dim + self.s_enc_out_dim, self.fwd_h_dim),
                                 nn.ReLU(),
                                 nn.Linear(self.fwd_h_dim, self.fwd_h_dim),
                                 nn.ReLU(),
                                 nn.Linear(self.fwd_h_dim, self.s_enc_out_dim)).to(device=DEVICE)


  def da_tensorize(self, da):
        da_vec = self.vector.action_vectorize(da)
        da_tensor = torch.tensor(da_vec).float().to(device=DEVICE)
        return da_tensor
  
  def forward(self, state_da, action, next_state_da):
        
        action_vec = self.vector.action_vectorize(action)

        # tensorize inputs
        state_tensor = self.da_tensorize(state_da)
        action_tensor = self.da_tensorize(action)
        next_state_tensor = self.da_tensorize(next_state_da)

        # get encodings
        phi_state = self.state_encoder(state_tensor)
        phi_next_state = self.state_encoder(next_state_tensor)

        # --- Inverse model pass ---
        # concat both encodings
        phi_concat = torch.cat((phi_state, phi_next_state),0)

        # pass thru inverse model
        action_vec_est = torch.sigmoid(self.inv_head(phi_concat))

        # --- Forward model pass ---
        # concat state encoding with action token tensors
        phi_s_a_concat = torch.cat((phi_state, action_tensor),0)

        # pass thru forward model
        phi_next_state_est = self.fwd_head(phi_s_a_concat)

        return action_vec, action_vec_est, phi_next_state, phi_next_state_est
  
  def get_intrinsic_rewards(self, user_das, sys_das, mask, eta = 0.01):
    
    intrinsic_rewards = []
    
    # iterate over batch and form tuples of (state, action, next_state)
    for i in range(len(user_das)-1):
      
      # not count if end of dialogue
      if mask[i].item() == 0.0:
        continue
      
      # get tuple
      state_da = user_das[i]
      
      action = sys_das[i+1]

      next_state_da = user_das[i+1]

      # get estimates from inverse and forward models
      _, _, phi_next_state, phi_next_state_est = self.forward(state_da, action, next_state_da)

      # extrinsic reward
      intrinsic_reward = (eta / 2.0) * torch.mean((phi_next_state_est - phi_next_state) ** 2)

      # append to list
      intrinsic_rewards.append(intrinsic_reward.item())

    return intrinsic_rewards

  def compute_loss(self, user_das, sys_das, mask):
    
    loss = torch.tensor(0.).to(device = DEVICE)

    # iterate over batch and form tuples of (state, action, next_state)
    for i in range(len(user_das)-1):
      
      # not count if end of dialogue
      if mask[i].item() == 0.0:
        continue
      
      # get tuple
      state_da = user_das[i]
      #state_utt = user_nlg.generate(user_das[i])
      
      action = sys_das[i+1]
      #action_utt = sys_nlg.generate(sys_das[i+1])

      next_state_da = user_das[i+1]
      #next_state_utt = user_nlg.generate(user_das[i+1])

      # get estimates from inverse and forward models
      action_vec, action_vec_est, phi_next_state, phi_next_state_est = self.forward(state_da, action, next_state_da)

      # inverse model loss
      loss_inv = torch.mean((torch.Tensor(action_vec).to(device = DEVICE) - action_vec_est) ** 2)

      # forward model loss
      loss_fwd = torch.mean((phi_next_state_est - phi_next_state) ** 2)

      # total ICM loss weighted by icm_beta
      loss_step = (1 - self.icm_beta) * loss_inv + self.icm_beta * loss_fwd

      loss += loss_step
    
    return loss / len(user_das)

# RND DA model
class RND_DA(nn.Module):
  def __init__(self, a_dim, rnd_h_dim = 524, out_dim = 340):
        super(RND_DA, self).__init__()
        
        # Multiwoz vector
        voc_file = os.path.join(path_convlab, 'data/multiwoz/sys_da_voc.txt')
        voc_opp_file = os.path.join(path_convlab, 'data/multiwoz/usr_da_voc.txt')
        self.vector = MultiWozVector(voc_file, voc_opp_file)

        # net
        self.net = nn.Sequential(nn.Linear(2 * a_dim, rnd_h_dim),
                                 nn.ReLU(),
                                 nn.Linear(rnd_h_dim, rnd_h_dim),
                                 nn.ReLU(),
                                 nn.Linear(rnd_h_dim, out_dim))
        self.net = self.net.float().to(device=DEVICE)

  def da_tensorize(self, user_da, sys_da):
        user_da_vec = self.vector.action_vectorize(user_da)
        sys_da_vec = self.vector.action_vectorize(sys_da)

        user_da_tensor = torch.tensor(user_da_vec).float().to(device=DEVICE)
        sys_da_tensor = torch.tensor(sys_da_vec).float().to(device=DEVICE)

        # concat das
        das_tensor = torch.cat((user_da_tensor, sys_da_tensor), 0)

        return das_tensor

  
  def forward(self, user_da, sys_da):
        
        # vectorize das and concatenate
        das_tensor = self.da_tensorize(user_da, sys_da)

        # pass thru net
        x = self.net(das_tensor)

        return x