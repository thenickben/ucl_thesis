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
from coreutils.analyzer_wrapper import Analyzer

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=UserWarning) 
pd.set_option("display.max_rows", None, "display.max_columns", None)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# load best policy
def load_best_policy(policy):
  policy.policy.load_state_dict(torch.load(os.path.join(path_models, model_name + '_policy.pt')))

def load_convlab_policy(policy):
  policy.policy.load_state_dict(torch.load(os.path.join(path_models,'best_ppo.pol.mdl')))


if __name__ == "__main__":

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
	icm_lr = cfg['icm_lr']
	seed = cfg['seed']

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

	grad_clip = cfg['grad_clip']

	# fix seeds
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)

	# --- DS instance ---
	policy_usr = RulePolicy(character='usr')
	dst_sys = RuleDST()

	# load best policy
	policy_sys = PPO()
	load_best_policy(policy_sys)

	sys_agent = PipelineAgent(None, dst_sys, policy_sys, None, name="sys")
	user_agent = PipelineAgent(None, None, policy_usr, None, 'user')

	user_nlg = TemplateNLG(is_user= True, mode='manual')
	sys_nlg = TemplateNLG(is_user = False, mode='manual')

	# instantiate analyzer
	analyzer = Analyzer(user_agent, path_convlab)

	# get analyzer results
	analyzer.analyze(sys_agent = sys_agent, total_dialog = 10000)

	# sample dialogue
	analyzer.sample_dialog(sys_agent, show_das = False, sys_nlg = sys_nlg, user_nlg = user_nlg)