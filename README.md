# Reinforcement Learning with Natural Language-based Exploration Strategies for Task-oriented Dialog Systems

This repo contains the code and report of my dissertation for the MSc. Machine Learning at UCL. 

### Abstract
Many end-to-end multi-task Dialogue Systems architectures rely on different modules to compose an entire dialogue pipeline. One of these components is a Policy module, which decides the actions to take given the inputs from the user. Moreover, the Policy can be learned by using Reinforcement Learning algorithms. These algorithms rely on an environment in which a learning Agent acts and receives feedback about the taken actions through a reward signal. However, most of the current Dialogue Systems environments only provide naïve (and sparse) rewards. The aim of this project is to explore Intrinsic Motivation Reinforcement Learning approaches where the agent learns an intrinsic reward signal, allowing it to predict the quality of its actions and speed up training. In particular, I extend Random Network Distillation and Curiosity-driven Reinforcement Learning algorithms to use (semantic) similarity between utterances to judge how often a state has been visited through a reward signal, which encourages exploration. Results obtained for MultiWOZ, a multi-domain dialogue data set, show that Intrinsic Motivation-based Dialogue Systems can outperform a policy that only observes extrinsic rewards. In particular, Random Network Distillation trained by taking the semantic similarity between user-system dialogues achieves an average success rate of 73 percent, improving significantly over the baseline PPO, which has an average success rate of 60 percent. Moreover, other performance metrics such as complete and book rates were also increased by 10 percent with respect to the baseline. I also show that these intrinsic motivation models make the system policy more robust to an increasing number of domains, suggesting that they may be a good approach for scaling up to environments containing an even bigger number of domains.

### Results

#### Performance metrics
|         |Complete | Success | Turn | Book |
|---------|:-------:|:----:|:----:|-----:|
|PPO       | 0.72 |0.60 |19.65 |0.59|
|RND (DAs) | 0.78 | 0.67|  17.66|  0.612| 
|RND (utt)|  **0.82**|  **0.73**|  **16.80**|  **0.68**| 
|IC (DAs) | 0.74|  0.64|  17.98|  0.57| 
|IC (utt)|  0.76|  0.65|  17.67|  0.60| 
|IC joint (DAs)|  0.78|  0.65|  18.37|  **0.69**| 


#### Learning curves

<div class="row">
  <div class="row" align="left">
    <img src="https://github.com/thenickben/ucl_thesis/blob/master/report/assets/success_rate_curves.png" width = 700> 
    <img src="https://github.com/thenickben/ucl_thesis/blob/master/report/assets/book_rate_curves.png" width = 700> 
    <img src="https://github.com/thenickben/ucl_thesis/blob/master/report/assets/turns_curves.png" width = 700> 
    <img src="https://github.com/thenickben/ucl_thesis/blob/master/report/assets/rewards_curves.png" width = 700> 
  </div>
</div>

- [Convlab-2 repo](https://github.com/thu-coai/ConvLab-2/)

- Results and core modules correspond to [this](https://drive.google.com/drive/folders/1SeAEZ5__ulWxaXQaF_Y6N1iZ6G6t2qQq?usp=sharing) Convlab-2 version.
