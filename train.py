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
from tqdm import tqdm, trange
import IPython
from pathlib import Path as P
import bs4 as bs
import argparse
from pprint import pprint
import matplotlib.pyplot as plt
from pprint import pprint
from google.colab import output
from torch import nn
from torch import multiprocessing as mp
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.optim as optim
from utils import *
from models import *

from transformers import BertForSequenceClassification, AdamW, BertConfig, BertTokenizer

# core modules were taken/modified from convlab-2 for commits before bdc9dba (inclusive)
# see https://github.com/thu-coai/ConvLab-2
from core.policy.rlmodule import MultiDiscretePolicy
from core.policy.vector.vector_multiwoz import MultiWozVector
from core.util.train_util import init_logging_handler
from core.dialog_agent.agent import PipelineAgent
from core.dialog_agent.env import Environment
from core.dst.rule.multiwoz import RuleDST
from core.policy.rule.multiwoz import RulePolicy
from core.policy.rlmodule import Memory, Transition
from core.nlg.template.multiwoz import TemplateNLG
from core.evaluator.multiwoz_eval import MultiWozEvaluator
from core.dialog_agent.session import BiSession
from core.util.analysis_tool.helper import Reporter
from core.utils.analyzer_wrapper import Analyzer

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=UserWarning) 
pd.set_option("display.max_rows", None, "display.max_columns", None)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Arguments for DS training')
parser.add_argument('--cfg', default = 'ppo')
parser.add_argument('--model_name', default = 'myppo')
parser.add_argument('--folder_name', default = 'myppo_results')

args = parser.parse_args()


# set instance names
folder_name = args.folder_name

model_name = args.model_name

cfg_file = args.cfg

# read config file
with open(os.path.join('cfg', cfg_file + '.yaml')) as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

# --- experiment hyperparameters ---
batchsz = cfg['batchsz']
n_steps = cfg['n_steps  # in steps']
log_freq = cfg['log_freq    # in epochs ']
update_round = cfg['update_round  # in steps']
n_eval = cfg['n_eval']

# PPO
s_dim = cfg['s_dim']
h_dim = cfg['h_dim']
a_dim = cfg['a_dim']
hv_dim = cfg['hv_dim']
surrogate_clip = cfg['surrogate_clip']     
optim_batchsz = cfg['optim_batchsz']
gamma = cfg['gamma']
tau = cfg['tau']
policy_lr = cfg['policy_lr']
value_lr =  cfg['value_lr']
grad_clip = cfg['grad_clip']
seed = cfg['seed']

# IC
if cfg_file == 'ic_das' or cfg_file == 'ic_utt':
	icm_lr = cfg['icm_lr']

	# icm pre-training hyperparameters
	pretrain_steps = cfg['pretrain_steps']
	update_round_icm = cfg['update_round_icm']
	show_freq = cfg['show_freq']

	# intrinsic reward models hyperparameters
	use_das = cfg['use_das']
	classifier_only = cfg['classifier_only']
	inv_h_dim = cfg['inv_h_dim']
	fwd_h_dim = cfg['fwd_h_dim']
	s_enc_out_dim = cfg['s_enc_out_dim']
	max_len = cfg['max_len']
	icm_beta = cfg['icm_beta']

	max_steps_intrinsic_reward = cfg['max_steps_intrinsic_reward']
	icm_eta = cfg['icm_eta']

# RND
if cfg_file == 'rnd_das' or cfg_file == 'rnd_utt':
	use_das = True
	classifier_only = True
	lr_rnd = 0.001
	m = 2
	warmup_episodes = 100
	r_mult = 1.0
	steps_update_rnd = 5
	max_annealing_step = 20000
	rnd_grad_clip = 10


# fix seeds
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# --- DS instance ---

# simple rule DST
dst_sys = RuleDST()
sys_nlg = TemplateNLG(is_user = False, mode='manual')

dst_usr = None
policy_usr = RulePolicy(character='usr')
user_agent = PipelineAgent(None, None, policy_usr, None, 'user')
user_nlg = TemplateNLG(is_user= True, mode='manual')

policy_sys = PPO()
if cfg_file == 'ic_das':
	icm = IC_DAS()
elif cfg_file == 'ic_utt':
	icm = IC_UTT()

# assemble
simulator = PipelineAgent(None, None, policy_usr, None, 'user')

evaluator = MultiWozEvaluator()
env = Environment(None, simulator, None, dst_sys, evaluator)



# ICM pre-training
if cfg_file == 'ic_das' or cfg_file == 'ic_utt':
	steps = 0
	i = 0 #episode

	# optimizer: outside any 'update' class method to allow
	icm_optim = optim.AdamW(icm.parameters(), lr= icm_lr)

	# ---------- pre-training loop ----------------
	while True:
	 
	  # get episode
	  sampled_episode, user_das, sys_das = sample(env, policy_sys, batchsz)

	  # unpack
	  _, _, _, _, mask = sampled_episode
	  
	  batchsz_real = len(mask)
	  
	  # update ICM
	  for j in range(update_round_icm):
	    # optim zero grad
	    icm_optim.zero_grad()
	    # compute icm loss
	    icm_loss = icm.compute_loss(user_das, sys_das, mask)
	    # backprop
	    icm_loss.backward()   
	    #clip
	    torch.nn.utils.clip_grad_norm_(icm.parameters(), 10)   
	    #optim step
	    icm_optim.step()

	  if i % show_freq == 0:
	    print('\r Steps: {} \tICM Loss: {}'.format(steps, icm_loss.item()), end="\n")

	  # finish if max steps reached
	  if steps > pretrain_steps:
	    break

	  steps += batchsz_real
	  i += 1

# RND pre-train
if cfg_file == 'rnd_das' or cfg_file == 'rnd_utt':

	out_dim = s_dim
	# --- RND using DAs ---
	if use_das:
	  model = RND_DA(a_dim)
	  target = RND_DA(a_dim)

	  # freeze target
	  for name, param in target.named_parameters():
	      param.requires_grad = False

	# --- RND using utterances ---
	else:
	  model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels = out_dim).to(device = DEVICE)
	  target = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels = out_dim).to(device = DEVICE)

	  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

	  # freeze target
	  for name, param in target.named_parameters():
	      param.requires_grad = False

	  # freeze model but head
	  if classifier_only:
	    for name, param in model.named_parameters():
	      if 'classifier' not in name: # classifier layer
	        param.requires_grad = False

	# instantiate rnd optimizer
	optimizer_rnd = AdamW(model.parameters(), lr = lr_rnd)

	# *********** RND warmup  ****************
	r_i_vec = []
	loss_vec_rnd = []
	r_i_normalised_vec = []

	# loss running average warmup
	# sys agent only used during RND warmup
	sys_agent = PipelineAgent(None, dst_sys, policy_sys, None, name="sys")

	steps = 0
	for warmup_episode in range(warmup_episodes):
	  sess = BiSession(sys_agent=sys_agent, user_agent=user_agent, kb_query=None, evaluator=evaluator)
	  sys_response = []
	  sess.init_session()
	  for step in range(50):
	    # sample dialog
	    sys_response, user_response, session_over, reward = sess.next_turn(sys_response)
	    user_da = sess.user_agent.get_out_da()
	    user_utt = user_nlg.generate(user_da)
	    sys_da = sess.sys_agent.get_out_da()
	    sys_utt = sys_nlg.generate(sys_da)
	    # get loss
	    if use_das:
	      pred_out = model(user_da, sys_da)
	      target_out = target(user_da, sys_da)
	    else:
	      tokens_tensor, mask_tensors = utt_tensorize(user_utt, sys_utt)
	      pred_out = model(tokens_tensor, mask_tensors)[0]
	      target_out = target(tokens_tensor, mask_tensors)[0]
	    loss_rnd = torch.mean((pred_out - target_out) ** 2)
	    loss_vec_rnd.append(loss_rnd.item())
	    if len(loss_vec_rnd) > m:
	      # compute running mean
	      loss_running_mean = np.array(loss_vec_rnd[-m:]).mean()
	      # compute intrinsic reward as the difference between the running mean and the turn loss
	      r_i = (loss_rnd.item() - loss_running_mean)
	      r_i_vec.append(r_i)
	      r_intrinsic = (r_i/(np.std(r_i_vec[-m:])+ 1e-3)) * r_mult
	      r_i_normalised_vec.append(r_intrinsic)
	      steps += 1
	    if session_over is True:
	      break
	  
	  if warmup_episode % 5 == 0:
	    print("Episode:", warmup_episode, "\tIntrinsic reward:", r_intrinsic)


# --- training loop ---
# init timer
t0 = time.time()

steps_vec = []
success_vec = []
req_rate_vec = []
book_rate_vec = []
turns_vec = []
rewards_vec = []
steps = 0
i = 0 # episodes

# log file
f = open(os.path.join(path_logs, model_name + '_log.txt'), "w+")

print("Date:", datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
print()
print("*" * 100)
print()
print("Training...")
print()

# ---------- training loop ----------------
while True:
  print('\rTotal Steps {}'.format(steps), end="")
  
  # get episode
  sampled_episode, user_das, sys_das = sample(env, policy_sys, batchsz)

  # unpack
  state, action, reward_ext, next_state, mask = sampled_episode
  
  if cfg_file == 'ic_das' or cfg_file == 'ic_utt':
	  if steps < max_steps_intrinsic_reward:
	    # get intrinsic rewards
	    reward_int = icm.get_intrinsic_rewards(user_das, sys_das, mask, eta = icm_eta)

	    # total reward = intrinsic + extrinsic
	    reward = tuple(list(reward_ext) + reward_int)
  elif cfg_file == 'rnd_das' or cfg_file == 'rnd_utt'
  	  # get intrinsic rewards + update model
	  reward_int = update_rnd(user_das, sys_das, reward_ext, steps)

	  # total reward = intrinsic + extrinsic
	  reward = tuple(list(reward_ext) + reward_int)
  else:
  	  reward_int = [0.]

  s = torch.from_numpy(np.stack(state)).to(device=DEVICE)
  a = torch.from_numpy(np.stack(action)).to(device=DEVICE)
  r = torch.from_numpy(np.stack(reward)).to(device=DEVICE)
  mask = torch.Tensor(np.stack(mask)).to(device=DEVICE)
  batchsz_real = s.size(0)
  
  # update policy with sampled transition batch
  policy_sys.update(i, batchsz_real, s, a, r, mask)

  if i % log_freq == 0:
    # compute metrics
    avg_success, avg_req_rate, avg_book_rate, avg_turns, avg_rewards = policy_evaluate_analyzer(policy_sys, env, evaluator, n_eval = n_eval)
    steps_vec.append(steps)
    success_vec.append(avg_success)
    req_rate_vec.append(avg_req_rate)
    book_rate_vec.append(avg_book_rate)
    turns_vec.append(avg_turns)
    rewards_vec.append(avg_rewards)
    print()
    print("-"*50)
    print('\rTime elapsed: {}'.format(format_time(time.time() - t0)), end="\n")
    print("Success rate:", avg_success)
    print("Request rate:",avg_req_rate)
    print("Book rate:",avg_book_rate)
    print("Average turns:",avg_turns)
    print("Average reward:",avg_rewards)
    print("-"*50)
    print()
    #log
    f.write("\n")
    f.write("Total Steps " + str(steps) +"\n")
    f.write("Success rate: " + str(avg_success) +"\n")
    f.write("Request rate: " + str(avg_req_rate) +"\n")
    f.write("Book rate: " + str(avg_book_rate) +"\n")
    f.write("Average turns: " + str(avg_turns) +"\n")
    f.write("Average reward: " + str(avg_rewards) +"\n")
    # save learning curve --> deprecated, not used
    save_learning_curve(steps_vec, [float(e) for e in success_vec])
     
  # finish if max steps reached
  if steps > n_steps:
    break

  steps += batchsz_real
  i += 1

# save trained policy
torch.save(policy_sys.policy.state_dict(), os.path.join(path_models, model_name + '_last_policy.pt'))

print()
print()
print("*" * 100)
print()
print("Training has finished. Complete training took {}".format(format_time(time.time() - t0)))
print("Best Success rate:", best_sr)
print()
print("*" * 100)

f.close()