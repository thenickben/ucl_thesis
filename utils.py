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

from models import MultiDiscretePolicy
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


class Analyzer:
    def __init__(self, user_agent, path_convlab, dataset='multiwoz'):
        self.user_agent = user_agent
        self.dataset = dataset
        self.path_convlab = path_convlab

    def build_sess(self, sys_agent):
        if self.dataset == 'multiwoz':
            evaluator = MultiWozEvaluator()
        else:
            evaluator = None

        if evaluator is None:
            self.sess = None
        else:
            self.sess = BiSession(sys_agent=sys_agent, user_agent=self.user_agent, kb_query=None, evaluator=evaluator)
        return self.sess

    def sample_dialog(self, sys_agent, show_das = False, sys_nlg = None, user_nlg = None):
        sess = self.build_sess(sys_agent)
        sys_response = '' if self.user_agent.nlu else []
        sess.init_session()
        print('init goal:')
        pprint(sess.evaluator.goal)
        print('-'*50)
        for i in range(40):
            sys_response, user_response, session_over, reward = sess.next_turn(sys_response)
            # Show utterances
            if show_das:          
              print('user in da:', sess.user_agent.get_in_da())
              print('user out da:', sess.user_agent.get_out_da())
              print()
              print('sys in da:', sess.sys_agent.get_in_da())
              print('sys out da:', sess.sys_agent.get_out_da())
            else:
              print('user:', user_nlg.generate(sess.user_agent.get_out_da()))
              print('sys:',  sys_nlg.generate(sess.sys_agent.get_out_da()))
            print()
            if session_over is True:
                break
        print('task complete:', sess.user_agent.policy.policy.goal.task_complete())
        print('task success:', sess.evaluator.task_success())
        print('book rate:', sess.evaluator.book_rate())
        print('inform precision/recall/f1:', sess.evaluator.inform_F1())
        print('-' * 50)
        print('final goal:')
        pprint(sess.evaluator.goal)
        print('=' * 100)

    def comprehensive_analyze(self, sys_agent, model_name, total_dialog=100):
        sess = self.build_sess(sys_agent)

        goal_seeds = [random.randint(1,100000) for _ in range(total_dialog)]
        precision = []
        recall = []
        f1 = []
        match = []
        suc_num = 0
        complete_num = 0
        turn_num = 0
        turn_suc_num = 0

        reporter = Reporter(model_name)

        if not os.path.exists('results'):
            os.mkdir('results')
        output_dir = os.path.join('results', model_name)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        for j in range(total_dialog):
            sys_response = '' if self.user_agent.nlu else []
            random.seed(goal_seeds[0])
            np.random.seed(goal_seeds[0])
            torch.manual_seed(goal_seeds[0])
            goal_seeds.pop(0)
            sess.init_session()

            usr_da_list = []
            failed_da_sys = []
            failed_da_usr = []
            last_sys_da = None

            step = 0

            for i in range(40):
                sys_response, user_response, session_over, reward = sess.next_turn(
                    sys_response)
                step += 2

                if hasattr(sess.sys_agent, "get_in_da") and isinstance(sess.sys_agent.get_in_da(), list) \
                        and sess.user_agent.get_out_da() != [] \
                        and sess.user_agent.get_out_da() != sess.sys_agent.get_in_da():
                    for da1 in sess.user_agent.get_out_da():
                        for da2 in sess.sys_agent.get_in_da():
                            if da1 != da2 and da1 is not None and da2 is not None and (da1, da2) not in failed_da_sys:
                                failed_da_sys.append((da1, da2))

                if isinstance(last_sys_da, list) \
                        and last_sys_da is not None and last_sys_da != [] and sess.user_agent.get_in_da() != last_sys_da:
                    for da1 in last_sys_da:
                        for da2 in sess.user_agent.get_in_da():
                            if da1 != da2 and da1 is not None and da2 is not None and (da1, da2) not in failed_da_usr:
                                failed_da_usr.append((da1, da2))

                last_sys_da = sess.sys_agent.get_out_da() if hasattr(sess.sys_agent, "get_out_da") else None
                usr_da_list.append(sess.user_agent.get_out_da())

                if session_over:
                    break

            task_success = sess.evaluator.task_success()
            task_complete = sess.user_agent.policy.policy.goal.task_complete()
            book_rate = sess.evaluator.book_rate()
            stats = sess.evaluator.inform_F1()
            if task_success:
                suc_num += 1
                turn_suc_num += step
            if task_complete:
                complete_num += 1
            if stats[2] is not None:
                precision.append(stats[0])
                recall.append(stats[1])
                f1.append(stats[2])
            if book_rate is not None:
                match.append(book_rate)
            domain_set = []
            for da in sess.evaluator.usr_da_array:
                if da.split('-')[0] != 'general' and da.split('-')[0] not in domain_set:
                    domain_set.append(da.split('-')[0])

            turn_num += step

            da_list = usr_da_list
            cycle_start = []
            for da in usr_da_list:
                if len(da) == 1 and da[0][2] == 'general':
                    continue

                if usr_da_list.count(da) > 1 and da not in cycle_start:
                    cycle_start.append(da)

            domain_turn = []
            for da in usr_da_list:
                if len(da) > 0 and da[0] is not None and len(da[0]) > 2:
                    domain_turn.append(da[0][1].lower())

            for domain in domain_set:
                reporter.record(domain, sess.evaluator.domain_success(domain), sess.evaluator.domain_reqt_inform_analyze(domain), failed_da_sys, failed_da_usr, cycle_start, domain_turn)

        tmp = 0 if suc_num == 0 else turn_suc_num / suc_num

        reporter.report(complete_num/total_dialog, suc_num/total_dialog, np.mean(precision), np.mean(recall), np.mean(f1), tmp, turn_num / total_dialog)

        # compute averages
        avg_complete = complete_num/total_dialog
        avg_succ = suc_num/total_dialog
        avg_turn = turn_num / total_dialog
        avg_book_rate = np.mean(match)
        avg_precision,avg_recall, avg_f1 = np.mean(precision), np.mean(recall), np.mean(f1)

        # get dict with results
        metrics_names = ['Total Dialog', 'Complete Num', 'Success Num', 'Turn Num', 'Avg Complete', 'Avg Success', 'Avg Turn', 
                         'Avg Book rate', 'Avg Precision','Avg Recall', 'Avg F1', 'Avg Turn Success']
        metrics_values = [total_dialog, complete_num, suc_num, turn_num, avg_complete, avg_succ, avg_turn, avg_book_rate, avg_precision,avg_recall, avg_f1, tmp]
        return pd.DataFrame.from_dict(dict(zip(metrics_names, metrics_values)), orient='index', columns = ['Metric Value'])

    def get_report_html(self, model_name):
      '''
      use IPython.display.HTML(report_html) to display on colab
      Not used, deprecated
      '''
      # get html report
      report_path = os.path.join(os.path.join(os.getcwd(), 'results'), model_name)
      report_html = os.path.join(report_path, 'report_multiwoz.html')
      return report_html

    def analyze(self, sys_agent, total_dialog):
      
      # auxiliar name for loading figures --> overwrites always but it's fine
      model_name_ = "analyzer_results"

      # get results
      results = self.comprehensive_analyze(sys_agent=sys_agent, model_name = model_name_, total_dialog = total_dialog)
      
      # get html report
      path_report = os.path.join(os.path.join(self.path_convlab, 'results'), model_name_)
      report_html = os.path.join(path_report, 'report_multiwoz.html')
      source = open(report_html, "r")
      soup = bs.BeautifulSoup(source, 'lxml')

      # Metrics results
      metrics_table = soup.find_all('table')[1]
      metrics = pd.read_html(str(metrics_table))[0].rename(columns = {"Unnamed: 0":""})

      return dict(results)['Metric Value'], metrics


# load pretrained mle weights
def load_mle(policy):
  policy.load_state_dict(torch.load(os.path.join(path_pretrained, 'mle.pt')))

# format time
def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

# save learning curve
def save_learning_curve(steps_vec, sr_vec):
  sr_vec = [float(e) for e in sr_vec]
  log_dict = {'steps':steps_vec, 'sr':sr_vec}
  with open(os.path.join(path_results, model_name + '_learning_curve.yaml'), 'w') as f:
      _ = yaml.dump(log_dict, f, default_flow_style=False) 

def utt_tensorize(user_utt, sys_utt):
  user_utt_tokens = ["[CLS]"] + tokenizer.tokenize(user_utt) + ["[SEP]"]
  sys_utt_tokens = tokenizer.tokenize(sys_utt) + ["[SEP]"]

  indexed_tokens = tokenizer.convert_tokens_to_ids(user_utt_tokens + sys_utt_tokens)
  mask_ids = [0]*len(user_utt_tokens) + [1]*len(sys_utt_tokens)

  tokens_tensor = torch.tensor([indexed_tokens]).to(device = DEVICE)
  mask_tensors = torch.tensor([mask_ids]).to(device = DEVICE)

  return tokens_tensor, mask_tensors

# request rate
def reqt_rate(goal):
  # get all domains
  domains = goal.keys()
  reqts = []
  for domain in domains:
    # get all goals for each domain 
    goals = goal[domain].keys()
    # check if 'reqt' is not '?'
    if 'reqt' in goals:
      # get all requestable slots
        reqt_slots = goal[domain]['reqt'].keys()
        # check all are filled
        for reqt_slot in reqt_slots:
          if goal[domain]['reqt'][reqt_slot] == '?':
            reqts.append(0)
          else:
            reqts.append(1)
  
  # compute reqt_rate
  if len(reqts) > 0:
    return np.mean(np.array(reqts))
  else:
    return 99 # meaning no info requests in initial user goal

def book_rate(goal):
  # get all domains
  domains = goal.keys()
  bookeds = []
  for domain in domains:
    # get all goals for each domain 
    goals = goal[domain].keys()
    # check if 'booked' is not '?'
    if 'booked' in goals: 
      if goal[domain]['booked'] == '?':
        bookeds.append(0)
      else:
        bookeds.append(1)

  if len(bookeds) > 0:
    return np.mean(np.array(bookeds))
  else:
    return 99 # meaning no book requests in initial user goal

def task_success(goal):
  task_reqt = reqt_rate(goal)
  task_booked = book_rate(goal)

  # first case: only info is requested (no booking)
  if task_booked == 99: 
    if task_reqt == 1:
      return True
    else:
      return False

  # second case: only  book is requested (no 'reqt')
  if task_reqt == 99: 
    if task_booked == 1:
      return True
    else:
      return False

  # second case: both info and booking are requested
  else:
    if task_reqt == 1 and task_booked == 1:
      return True
    else:
      return False

def get_reward(evaluator, done):
  if task_success(evaluator.goal) and done:
    return 40.
  else:
    return -1.

# sample dialogue
def sample_dialogue(policy, env, evaluator):

  user_das = []
  sys_das = []
  rewards = []
  dones = []

  # sample ignoring those init goals with fail info
  while True:
    fail_info = 0
    fail_book = 0
    s = env.reset()
    init_goal = copy.deepcopy(evaluator.goal)
    domains = init_goal.keys()
    for domain in domains:
      if 'fail_info' in init_goal[domain].keys():
        fail_info = 1     
      if 'fail_book' in init_goal[domain].keys():
        fail_book = 1
    if fail_info == 0 and fail_book == 0:
      break

  # s -> vector
  s_vector = policy.vector.state_vectorize(s)
  # sys response
  a = policy.predict(s)
  # interact with env, s_next -> s
  s, _, done = env.step(a)
  reward = get_reward(evaluator, done)
  # get das
  user_da = s['user_action']
  user_das.append(user_da)
  sys_da = s['system_action']
  sys_das.append(sys_da)
  rewards.append(reward)
  dones.append(done)

  # trajectory loop  
  for _ in range(30):    
      # s -> vector
      s_vector = policy.vector.state_vectorize(s)
      # sys response DA
      a = policy.predict(s)
      # env step
      next_s, _, done = env.step(a)
      reward = get_reward(evaluator, done)
      
      # get das
      user_da = next_s['user_action']
      user_das.append(user_da)
      sys_da = next_s['system_action']
      sys_das.append(sys_da)

      # a flag indicates ending or not
      dones.append(done)

      # rewards append
      rewards.append(reward)


      # next_s -> vector
      next_s_vec = policy.vector.state_vectorize(next_s)        
      # update per step
      s = next_s
      # break if trajectory is done or t > max_len
      if done == 1:
        break

  final_goal = copy.deepcopy(evaluator.goal)
  success = task_success(final_goal)
  task_reqt_rate = reqt_rate(final_goal)
  task_book_rate = book_rate(final_goal)

  return user_das, sys_das, rewards, dones, init_goal, final_goal, success, task_reqt_rate, task_book_rate

def show_dialogue(user_das, sys_das, rewards, dones, init_goal, final_goal, success, task_reqt_rate, task_book_rate):
  print("Init goal")
  pprint(init_goal)
  print()
  print("User (initial dialog):", user_nlg.generate(user_das[0]))
  print()
  for i in range(1, len(user_das)):
    print("Sys:", sys_nlg.generate(sys_das[i]))
    print("User:", user_nlg.generate(user_das[i]))
    print("Reward:", rewards[i])
    print("Done:", dones[i])
    print()
  print("Final goal")
  pprint(final_goal)
  print()
  print("Task success")
  print(success)
  print()
  print("Task reqt rate")
  if task_reqt_rate == 99:
    print("NA")
  else:
    print(task_reqt_rate)
  print()
  print("Task book rate")
  if task_book_rate == 99:
    print("NA")
  else:
    print(task_book_rate)
  print()

def policy_evaluate(policy_sys, env, evaluator, n_eval = 100, show_results = False):
  success_vec = []
  req_rate_vec = []
  book_rate_vec = []
  turns_vec = []
  rewards_vec = []
  final_goals = []
  for i in range(n_eval):
    user_das, sys_das, rewards, dones, init_goal, final_goal, success, task_reqt_rate, task_book_rate =  sample_dialogue(policy_sys, env, evaluator)
    final_goals.append(final_goal)
    success_vec.append(success)
    if task_reqt_rate != 99:
      req_rate_vec.append(task_reqt_rate)
    if task_book_rate != 99:
      book_rate_vec.append(task_book_rate)
    turns_vec.append(len(user_das))
    rewards_vec.append(np.mean(np.array(rewards)))

  # get averages
  avg_success = np.mean(np.array(success_vec))    
  avg_req_rate = np.mean(np.array(req_rate_vec)) 
  avg_book_rate = np.mean(np.array(book_rate_vec)) 
  avg_turns = np.mean(np.array(turns_vec))    
  avg_rewards = np.mean(np.array(rewards_vec)) 

  # show results
  if show_results:
    print("Success rate:", avg_success)
    print("Request rate:",avg_req_rate)
    print("Book rate:",avg_book_rate)
    print("Average turns:",avg_turns)
    print("Average reward:",avg_rewards)

  return avg_success, avg_req_rate, avg_book_rate, avg_turns, avg_rewards


# sample a transition given a policy and batch size
def sample(env, policy, batchsz):

    buff = Memory()

    # we need to sample batchsz of (state, action, next_state, reward, mask)
    # each trajectory contains `trajectory_len` num of items, so we only need to sample
    # `batchsz//trajectory_len` num of trajectory totally
    # the final sampled number may be larger than batchsz.

    sampled_num = 0
    sampled_traj_num = 0
    traj_len = 50
    real_traj_len = 0

    user_das = []
    sys_das = []

    while sampled_num < batchsz:
        # for each trajectory, we reset the env and get initial state
        # ignore if there is fail_book or fail_info!
          # sample ignoring those init goals with fail info
        while True:
          fail_info = 0
          fail_book = 0
          s = env.reset()
          init_goal = copy.deepcopy(evaluator.goal)
          domains = init_goal.keys()
          for domain in domains:
            if 'fail_info' in init_goal[domain].keys():
              fail_info = 1     
            if 'fail_book' in init_goal[domain].keys():
              fail_book = 1
          if fail_info == 0 and fail_book == 0:
            break

        for t in range(traj_len):

            # [s_dim] => [a_dim]
            s_vec = torch.Tensor(policy.vector.state_vectorize(s))
            a = policy.predict(s)

            # interact with env
            next_s, _, done = env.step(a)
            r = get_reward(env.evaluator, done)

            # get das
            user_da = next_s['user_action']
            user_das.append(user_da)
            sys_da = next_s['system_action']
            sys_das.append(sys_da)

            # a flag indicates ending or not
            mask = 0 if done else 1

            # get reward compared to demostrations
            next_s_vec = torch.Tensor(policy.vector.state_vectorize(next_s))

            # save to queue
            buff.push(s_vec.numpy(), policy.vector.action_vectorize(a), r, next_s_vec.numpy(), mask)

            # update per step
            s = next_s
            real_traj_len = t

            if done:
                break

        # this is end of one trajectory
        sampled_num += real_traj_len
        sampled_traj_num += 1
        # t indicates the valid trajectory length

    return buff.get_batch(), user_das, sys_das

def policy_evaluate_analyzer(policy, env, evaluator, n_eval):
  # instantiate analyzer
  analyzer = Analyzer(simulator, path_convlab)

  # instantiate sys_agent with current policy
  sys_agent = PipelineAgent(None, dst_sys, policy, None, name="sys")

  # get analyzer results
  m1, m2 = analyzer.analyze(sys_agent = sys_agent, total_dialog = n_eval)

  # grab metrics of interest
  avg_success = m1['Avg Success']   
  avg_req_rate = m1['Avg Complete']
  avg_book_rate = m1['Avg Book rate']
  avg_turns = m1['Avg Turn']/2  # convlab doubles this due wrong definition
  # for avg reward I'll use my own function with less n_dialog, just as an estimate
  # TO DO: improve
  _, _, _, _, avg_rewards = policy_evaluate(policy, env, evaluator, 50)

  return avg_success, avg_req_rate, avg_book_rate, avg_turns, avg_rewards

# save learning curve
def save_learning_curve(steps_vec, sr_vec):
  log_dict = {'steps':steps_vec, 'sr':sr_vec}
  with open(os.path.join(path_results, model_name + '_learning_curve.yaml'), 'w') as f:
      _ = yaml.dump(log_dict, f, default_flow_style=False) 


def update_rnd(user_das, sys_das, reward_ext, total_steps):

  reward_int = []
  
  for i in range(len(user_das)):
    user_da = user_das[i]
    sys_da = sys_das[i]

    if not use_das:
      user_utt = user_nlg.generate(user_da)   
      sys_utt = sys_nlg.generate(sys_da)
      tokens_tensor, mask_tensors = utt_tensorize(user_utt, sys_utt)

    # update model
    for _ in range(steps_update_rnd):
      optimizer_rnd.zero_grad() 

      # loss
      if use_das:
        pred_out = model(user_da, sys_da)
        target_out = target(user_da, sys_da)
      else:
        tokens_tensor, mask_tensors = utt_tensorize(user_utt, sys_utt)
        pred_out = model(tokens_tensor, mask_tensors)[0]
        target_out = target(tokens_tensor, mask_tensors)[0]

      loss_rnd = torch.mean((pred_out - target_out) ** 2)
      loss_rnd.backward()
      torch.nn.utils.clip_grad_norm(model.parameters(), rnd_grad_clip)
      optimizer_rnd.step()

    # compute running mean
    loss_running_mean = np.array(loss_vec_rnd[-m:]).mean()

    # compute intrinsic reward as the difference between the running mean and the turn loss
    r_i = loss_rnd.item() - loss_running_mean
    r_i_vec.append(r_i)
    r_intrinsic = (r_i/np.std(r_i_vec[-m:])) * r_mult * max(0.001, 1. - total_steps / max_annealing_step)

    r_i_normalised_vec.append(r_intrinsic)
    reward_int.append(r_intrinsic)
    loss_vec_rnd.append(loss_rnd.item())

  return reward_int   

  def load_policy(model_name, policy, last_policy = True):
  if last_policy:
    policy.policy.load_state_dict(torch.load(os.path.join(path_saved_models, model_name + '_last_policy.pt')))
  else:
    policy.policy.load_state_dict(torch.load(os.path.join(path_saved_models, model_name + '_best_policy.pt')))

def get_metrics(model_names, model_aliases, n_dialog = 1000, last_policy = True):

  columns = ['model',	'avg_complete',	'avg_success',	'avg_turn',	'avg_book_rate',	'avg_precision',	'avg_recall',	'avg_f1',	'avg_turn_success']

  avg_metrics = pd.DataFrame(columns = columns)

  metrics_per_domain = []

  for i, model_name in enumerate(model_names):
    # make DS for evaluation
    policy_usr = RulePolicy(character='usr')
    dst_sys = RuleDST()

    # load best policy
    policy_sys = PPO()
    load_policy(model_name, policy_sys, last_policy)

    sys_agent = PipelineAgent(None, dst_sys, policy_sys, None, name="sys")
    user_agent = PipelineAgent(None, None, policy_usr, None, 'user')

    user_nlg = TemplateNLG(is_user= True, mode='manual')
    sys_nlg = TemplateNLG(is_user = False, mode='manual')

    # instantiate analyzer
    analyzer = Analyzer(user_agent, path_convlab)

    # make session
    sess = analyzer.build_sess(sys_agent)

    # instantiate analyzer
    analyzer = Analyzer(user_agent, path_convlab)

    # get analyzer results
    m1, m2 = analyzer.analyze(sys_agent = sys_agent, total_dialog = n_dialog)

    # append metrics per domain (m2)
    metrics_per_domain.append(m2)

    # append m1 to avg_metrics
    dd = {'model' : model_aliases[i],
      'avg_complete' : m1['Avg Complete'],
      'avg_success' : m1['Avg Success'],
      'avg_turn' : m1['Avg Turn'],
      'avg_book_rate' : m1['Avg Book rate'],
      'avg_precision' : m1['Avg Precision'],
      'avg_recall' : m1['Avg Recall'],
      'avg_f1' : m1['Avg F1'],
      'avg_turn_success' : m1['Avg Turn Success']
      }

    avg_metrics = avg_metrics.append(dd, ignore_index=True)

  return avg_metrics, metrics_per_domain

def reshape_metrics_per_domain(metrics_per_domain, model_aliases):

  domains = list(metrics_per_domain[1].to_dict()[''].values())

  all_metrics = []

  # get all metrics
  for i in range(len(metrics_per_domain)):
    dft = metrics_per_domain[i].T
    for j in range(7):
      all_metrics.append([model_aliases[i]] + list(dft[j].values))

  # transform into dict
  d_all = dict()
  for domain in domains:
    dd = dict()
    for metrics in all_metrics:
      if metrics[1] == domain:
        mod = metrics[0]
        vals = metrics[2:]
        dd[mod] = vals
    d_all[domain] = dd
    d_all[domain]['metric'] = list(metrics_per_domain[0].to_dict().keys())[1:]

  # move dataframes to dict
  metrics_dict = dict()

  for domain in domains:
    df = pd.DataFrame.from_dict(d_all[domain])
    cols = ['metric'] + model_aliases
    df = df.reindex(columns=cols)
    metrics_dict[domain] = df
  
  return metrics_dict

def multi_domain_performance(sess, n_domain, n_dialog, show_dialogues = False):
  
  domains_counter = {'police' : 0, 
                     'attraction' : 0,  
                     'hotel' : 0,  
                     'taxi' : 0,  
                     'train' : 0,  
                     'restaurant' : 0,  
                     'hospital' : 0}
  task_complete_vec = []
  task_success_vec = []
  book_rate_vec = []
  precision_vec = []
  recall_vec = []
  f1_vec = []
  
  for _ in range(n_dialog):
    # get domain-specific goal by negative sampling
    max_samples = 200
    n_samples = 0
    while True:
      sess.init_session()
      goal = sess.evaluator.goal
      domains = list(goal.keys())
      n_samples += 1
      if len(domains) == n_domain:
        break
      if n_samples == max_samples:
        print("Not possible to sample from", n_domain, "domains jointly")
        return None, None, None, None, None, None, None

    # increase domains count
    for i in range(n_domain):
      domains_counter[domains[i]] += 1

    # sample dialog and get results
    sys_response = []
    for i in range(40):
      sys_response, user_response, session_over, reward = sess.next_turn(sys_response)
      # Show utterances
      if show_dialogues:
        print()
        print('user:', user_nlg.generate(sess.user_agent.get_out_da()))
        print('sys:',  sys_nlg.generate(sess.sys_agent.get_out_da()))
        print()
      if session_over is True:
          break
    # append partial results
    task_complete_vec.append(sess.user_agent.policy.policy.goal.task_complete())
    task_success_vec.append(sess.evaluator.task_success())
    book_rate = sess.evaluator.book_rate()
    if book_rate:
      book_rate_vec.append(book_rate)
    prf1 = sess.evaluator.inform_F1()
    if prf1[0]:
      precision_vec.append(prf1[0])
    if prf1[1]:
      recall_vec.append(prf1[1])
    if prf1[2]:
      f1_vec.append(prf1[2])

  # average results

  avg_task_complete = np.array(task_complete_vec).mean()
  avg_task_success = np.array(task_success_vec).mean()
  avg_book_rate = np.array(book_rate_vec).mean()
  avg_precision = np.array(precision_vec).mean()
  avg_recall = np.array(recall_vec).mean()
  avg_f1 = np.array(f1_vec).mean()

  return avg_task_complete, avg_task_success, avg_book_rate, avg_precision, avg_recall, avg_f1, domains_counter

def run_multidomain_study(sess, n_eval):
  avg_task_complete, avg_task_success, avg_book_rate, avg_precision, avg_recall, avg_f1, _ = [], [], [], [], [], [], []
  for i in range(1, 4):
    avg_task_complete_, avg_task_success_, avg_book_rate_, avg_precision_, avg_recall_, avg_f1_, _ = multi_domain_performance(sess, i, n_dialog = n_eval)
    # append results
    avg_task_complete.append(avg_task_complete_)
    avg_task_success.append(avg_task_success_)
    avg_book_rate.append(avg_book_rate_)
    avg_precision.append(avg_precision_)
    avg_recall.append(avg_recall_)
    avg_f1.append(avg_f1_)

  df = pd.DataFrame(list(zip(["1","2","3"], avg_task_complete, avg_task_success, avg_book_rate, avg_precision, avg_recall, avg_f1)),
                    columns = ('Number of domains', 'avg_task_complete', 'avg_task_success', 'avg_book_rate', 'avg_precision', 'avg_recall', 'avg_f1'))
  df = df.set_index('Number of domains')
  return df

def plot_multidomain_results(df):

  n_domains = list(range(1, 1 + len(df)))
  book_rate = list(df.avg_book_rate)
  success_rate = list(df.avg_task_success)
  f1 = list(df.avg_f1)

  # plot success rate for different parameters
  fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(15,7.5))

  ax1.plot(n_domains, f1, label= 'AC F1 score')
  ax1.set_xlabel('number of domains')
  ax1.set_ylabel('F1 score')

  ax2.plot(n_domains, book_rate, label= 'AC Book rate')
  ax2.set_xlabel('number of domains')
  ax2.set_ylabel('Book rate')

  ax3.plot(n_domains, success_rate, label= 'AC Success rate')
  ax3.set_xlabel('number of domains')
  ax3.set_ylabel('Success rate')

  plt.show()

def multidomain_study(model_names, model_aliases, n_eval = 100, last_policy = True):

  multidomain_results_all = dict()

  # run study for all models
  for i, model_name in enumerate(model_names):

    # make DS for evaluation
    policy_usr = RulePolicy(character='usr')
    dst_sys = RuleDST()

    # load best policy
    policy_sys = PPO()
    load_policy(model_name, policy_sys, last_policy)

    sys_agent = PipelineAgent(None, dst_sys, policy_sys, None, name="sys")
    user_agent = PipelineAgent(None, None, policy_usr, None, 'user')

    user_nlg = TemplateNLG(is_user= True, mode='manual')
    sys_nlg = TemplateNLG(is_user = False, mode='manual')

    # instantiate analyzer
    analyzer = Analyzer(user_agent, path_convlab)

    # make session
    sess = analyzer.build_sess(sys_agent)

    # get multi domain results for current model
    multidomain_results = run_multidomain_study(sess, n_eval)

    # add to dict
    multidomain_results_all[model_aliases[i]] = multidomain_results.to_dict('index')

  return multidomain_results_all

def plot_multidomain(model_aliases, multidomain_results_all): 
  # now plot
  n_domains = [1,2,3]

  fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(35,10))

  for i, model in enumerate(model_aliases):

    book_rate, success_rate, f1 = [], [], []

    for j in n_domains:
      book_rate.append(multidomain_results_all[model][str(j)]['avg_book_rate'])
      success_rate.append(multidomain_results_all[model][str(j)]['avg_task_success'])
      f1.append(multidomain_results_all[model][str(j)]['avg_f1'])

    ax1.plot(n_domains, f1, label = model)
    ax1.set_xlabel('number of domains')
    ax1.set_ylabel('F1 score')
    ax1.legend(loc='upper right', shadow=True, fontsize='x-large')

    ax2.plot(n_domains, book_rate, label = model)
    ax2.set_xlabel('number of domains')
    ax2.set_ylabel('Book rate')
    ax2.legend(loc='upper left', shadow=True, fontsize='x-large')

    ax3.plot(n_domains, success_rate, label = model)
    ax3.set_xlabel('number of domains')
    ax3.set_ylabel('Success rate')
    ax3.legend(loc='upper right', shadow=True, fontsize='x-large')

  plt.show()

def get_seed_for_goal(goal_domains):

  while True:
    seed = random.randint(1, 99999)

    # fix seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    policy_usr = RulePolicy(character='usr')
    # assemble
    simulator = PipelineAgent(None, None, policy_usr, None, 'user')

    evaluator = MultiWozEvaluator()
    env = Environment(None, simulator, None, dst_sys, evaluator)

    fail_info = 0
    fail_book = 0
    goal_achieved = 0
    s = env.reset()
    init_goal = copy.deepcopy(evaluator.goal)
    domains = init_goal.keys()
    if len(domains) != len(goal_domains):
      continue
    for domain in domains:
      if 'fail_info' in init_goal[domain].keys():
        fail_info = 1     
      if 'fail_book' in init_goal[domain].keys():
        fail_book = 1
    if fail_info == 0 and fail_book == 0:
      # if no fail anything then check for domains
      for goal_domain in goal_domains:
        if goal_domain in domains:
          goal_achieved = 1
        else:
          goal_achieved = 0
      if goal_achieved == 1:     
        break
      else:
        continue

  return seed

def sample_dialog_from_goal(policy_name, goal_domains, seed = None, show_das = False):
    
    # get seed for goal
    if not seed:
      seed = get_seed_for_goal(goal_domains)

    # fix seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # make DS for evaluation
    policy_usr = RulePolicy(character='usr')
    dst_sys = RuleDST()

    # load best policy
    policy_sys = PPO()
    load_policy(policy_name, policy_sys, last_policy= True)

    sys_agent = PipelineAgent(None, dst_sys, policy_sys, None, name="sys")
    user_agent = PipelineAgent(None, None, policy_usr, None, 'user')

    user_nlg = TemplateNLG(is_user= True, mode='manual')
    sys_nlg = TemplateNLG(is_user = False, mode='manual')

    # instantiate analyzer
    analyzer = Analyzer(user_agent, path_convlab)

    # fix seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    sess = analyzer.build_sess(sys_agent)
    sys_response = '' if analyzer.user_agent.nlu else []

    sess.init_session()
    print('init goal:')
    pprint(sess.evaluator.goal)
    print('-'*50)
    for i in range(40):
        sys_response, user_response, session_over, reward = sess.next_turn(sys_response)
        # Show utterances
        if show_das:          
          print('user in da:', sess.user_agent.get_in_da())
          print('user out da:', sess.user_agent.get_out_da())
          print()
          print('sys in da:', sess.sys_agent.get_in_da())
          print('sys out da:', sess.sys_agent.get_out_da())
        else:
          print('user:', user_nlg.generate(sess.user_agent.get_out_da()))
          print('sys:',  sys_nlg.generate(sess.sys_agent.get_out_da()))
        print()
        if session_over is True:
            break
    print('task complete:', sess.user_agent.policy.policy.goal.task_complete())
    print('task success:', sess.evaluator.task_success())
    print('book rate:', sess.evaluator.book_rate())
    print('inform precision/recall/f1:', sess.evaluator.inform_F1())
    print('-' * 50)
    print('final goal:')
    pprint(sess.evaluator.goal)

    return seed

# multiwoz distribution
def sample_domain_distribution(sess, n_samples):
  domains_counter = {'police' : 0, 
                     'attraction' : 0,  
                     'hotel' : 0,  
                     'taxi' : 0,  
                     'train' : 0,  
                     'restaurant' : 0,  
                     'hospital' : 0} 
  n_goals = 0
  for _ in range(n_samples):
    sess.init_session()
    goal = sess.evaluator.goal
    sampled_domains = list(goal.keys())
    # increase domains count
    for domain in sampled_domains:
      domains_counter[domain] += 1
    # increase number of sampled goals
    n_goals += len(sampled_domains)
  
  # get percentages
  for domain in domains_counter.keys():
    domains_counter[domain] /= n_goals
  
  # bar plot
  sns.barplot(list(domains_counter.keys()), list(domains_counter.values()))
  plt.xticks(rotation=70)

  return domains_counter

def sample_domain_numbers(sess, n_samples):
  n_counter = {'1' : 0, 
                     '2' : 0,  
                     '3' : 0,  
                     '4' : 0,  
                     '5' : 0,  
                     '6' : 0,  
                     '6' : 0} 
  for _ in range(n_samples):
    sess.init_session()
    goal = sess.evaluator.goal
    n_sampled_domains = len(list(goal.keys()))
    # increase number of domains count
    n_counter[str(n_sampled_domains)] += 1
 
  # get percentages
  for i in n_counter.keys():
    n_counter[i] /= n_samples
  
  # bar plot
  sns.barplot(list(n_counter.keys()), list(n_counter.values()))

  return n_counter