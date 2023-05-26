# Preference-RL
A survey of Preference Reinforcement Learning


## Why preference?

For hard-to-specify tasks. It is difficult to quantify the utilities of decisions in a large number of daily tasks. For example, manually designing a reward function for general text-generation systems is impossible, since the contexts required to be concerned is massive. Similar to traditional Inverse Reinforcement Learning (IRL) learning a reward function based on expert demonstrations, Preference-based RL (PBRL) aims to learn a reward function given limited preference signals of paired trajectories / sub-trajectories from human / rule-based systems.


## Existing works

Categorized by the research problems then methods.

### <u> Directly learning from preferences without fitting a reward model?</u>

- **Programming by Feedback**. <br> Akrour R, Schoenauer M, Sebag M, et al. (**ICML 2014**). [[paper]](http://proceedings.mlr.press/v32/schoenauer14.pdf)
  
- **Model-free preference-based reinforcement learning**.  <br> Wirth C, Fürnkranz J, Neumann G. (**AAAI 2016**) [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/10269)


### <u> Learning with as few preferences as possible (Feedback-efficient) ? </u>

- Active Reward Learning

  - **Active preference-based learning of reward functions**.  <br> Sadigh D, Dragan A D, Sastry S, et al. (**RSS 2017**) [[paper]](https://escholarship.org/uc/item/88k894w7)
  
  - **Asking Easy Questions: A User-Friendly Approach to Active Reward Learning**. <br> Bıyık E, Palan M, Landolfi N C, et al. (**CoRL 2019**) [[paper]](https://arxiv.org/abs/1910.04365)

  - **Active Preference-Based Gaussian Process Regression for Reward Learning**. <br> Bıyık E, Huynh N, Kochenderfer M J, et al. (**RSS 2020**) [[paper]](https://arxiv.org/abs/2005.02575)

  - **Active preference learning using maximum regret**. <br> Wilde N, Kulić D, Smith S L. (**IROS 2020**) [[paper]](https://ieeexplore.ieee.org/abstract/document/9341530)

  - **Information Directed Reward Learning for Reinforcement Learning**. <br> Lindner D, Turchetta M, Tschiatschek S, et al. (**NIPS 2021**) [[paper]](https://proceedings.neurips.cc/paper/2021/hash/1fa6269f58898f0e809575c9a48747ef-Abstract.html)

- Propose Hypothetical/Generated Trajectories
  
  - **Learning Human Objectives by Evaluating Hypothetical Behavior**. <br> Reddy S, Dragan A, Levine S, et al. (**ICML 2020**) [[paper]](http://proceedings.mlr.press/v119/reddy20a.html) <br> * *Utilizing a generative/dynamics model to synthesize hypothetical behaviors*.

  - **Efficient Preference-Based Reinforcement Learning Using Learned Dynamics Models**. <br> Liu Y, Datta G, Novoseller E, et al. (**NIPS 2022 Workshop**) [[paper]](https://arxiv.org/abs/2301.04741)

- Pretraining for Diverse Trajectories

  - **PEBBLE: Feedback-Efficient Interactive Reinforcement Learning via Relabeling Experience and Unsupervised Pre-training**. <br> Lee K, Smith L, Abbeel P. Pebble. (**ICML 2021**) [[paper]](https://arxiv.org/abs/2106.05091) <br> * *Pretraining with unsupervised objectives (e.g., maximizing state coverage) to ensure informative feedbacks*.

- Actively collect uncertain data w.r.t the reward models through guided exploration

  - **Reward Uncertainty for Exploration in Preference-based Reinforcement Learning**. <br> Liang X, Shu K, Lee K, et al. (**ICLR 2022**) [[paper]](https://arxiv.org/abs/2205.12401) <br> * *Regard the disagreements of the reward model ensemble as intrinsic reward to collect informative data*.

- Training the reward model feedback-efficiently

  - **SURF: Semi-supervised reward learning with data augmentation for feedback-efficient preference-based reinforcement learning**. <br> Park J, Seo Y, Shin J, et al. (**ICLR 2022**) [[paper]](https://arxiv.org/abs/2203.10050) <br> * *Using semi-supervised loss with self-generated pseudo-labels, and augmenting the data by exchanging sub-trajectory between paired trajectories.*

- Training the reward model data-efficiently
  
  - **Meta-Reward-Net: Implicitly Differentiable Reward Learning for Preference-based Reinforcement Learning** <br> Liu R, Bai F, Du Y, et al. (**NIPS 2022**) [[paper]](https://proceedings.neurips.cc/paper_files/paper/2022/hash/8be9c134bb193d8bd3827d4df8488228-Abstract-Conference.html)

### <u> Can the learned reward models generalize or assist downstream tasks ? </u>

- **Few-Shot Preference Learning for Human-in-the-Loop RL**. <br> Hejna III D J, Sadigh D. (**CoRL 2022**) [[paper]](https://proceedings.mlr.press/v205/iii23a.html) <br> * *Training the reward model with MAML, and fast adapt the model with few-shot trajectories in downstream tasks*.

- **Learning a Universal Human Prior for Dexterous Manipulation from Human Preference**. <br> Ding Z, Chen Y, Ren A Z, et al. (**Preprint 2023**) [[paper]](https://arxiv.org/abs/2304.04602) <br> * *Train a universal reward model with multi-task dataset, and directly use the reward to guide learning in downstream tasks*

### <u> Reward misidentification / Robustness of reward models </u>

- **Causal Confusion and Reward Misidentification in Preference-Based Reward Learning**. <br> Tien J, He J Z Y, Erickson Z, et al. (**ICLR 2023**) [[paper]](https://openreview.net/forum?id=R0Xxvr_X3ZA)

### <u> General Models of rewards? </u>

- **Deep reinforcement learning from human preferences**.  <br> Christiano P F, Leike J, Brown T, et al. (**NIPS 2017**) [[paper]](https://proceedings.neurips.cc/paper_files/paper/2017/hash/d5e2c0adad503c91f91df240d0cd4e49-Abstract.html) <br> * *Firstly proposed Bradley-Terry Model.*

- **Models of human preference for learning reward functions**. <br> Knox W B, Hatgis-Kessell S, Booth S, et al. (**Preprint 2022**) [[paper]](https://arxiv.org/abs/2206.02231) <br> * *Propose using regrets as an alternative form to model the preferences instead of the trajectory return*.

- **Preference Transformer: Modeling Human Preferences using Transformers for RL** <br> Changyeon Kim, Jongjin Park, Jinwoo Shin, Honglak Lee, Pieter Abbeel, Kimin Lee. (**ICLR 2023**) [[paper]](https://arxiv.org/abs/2303.00957) 

### <u> Different forms of (human) feedbacks? </u>

- **Learning Human Objectives by Evaluating Hypothetical Behavior**. <br> Reddy S, Dragan A, Levine S, et al. (**ICML 2020**) [[paper]](http://proceedings.mlr.press/v119/reddy20a.html) <br> * *Requiring human choosing one of three labels ([good, unsafe, neutral]) to describe the trajectories*.

- **Widening the Pipeline in Human-Guided Reinforcement Learning with Explanation and Context-Aware Data Augmentation**. <br> Guan L, Verma M, Guo S S, et al. (**NIPS 2021**) [[paper]](https://proceedings.neurips.cc/paper/2021/hash/b6f8dc086b2d60c5856e4ff517060392-Abstract.html) <br> * *Requiring human annotating the task-related features on the observation features*.

### <u> Novel problem settings related to PBRL </u>

- **Beyond Reward: Offline Preference-guided Policy Optimization**. <br> Kang Y, Shi D, Liu J, et al. (**ICML 2023**) [[paper]](https://openreview.net/forum?id=i8AnfJYMvz) <br> * *Offline + Preference (Same as Preference Transformer)*

- **Efficient Meta Reinforcement Learning for Preference-based Fast Adaptation**. <br> Ren Z, Liu A, Liang Y, et al. (**NIPS 2022**) [[paper]](https://arxiv.org/abs/2211.10861) <br> * *For Meta RL, we can only access to the preferences at the meta-testing stage while the ground truth rewards are only accessible in the meta-training stage.*

- **Deploying Offline Reinforcement Learning with Human Feedback**. <br> Li Z, Xu K, Liu L, et al. (**Preprint 2023**) [[paper]](https://arxiv.org/abs/2303.07046) <br> * *Select the offline trained models for downstream deployment with human feedback*

### <u> (Skill Discovery) Discovering skills aligning with human intents </u>

- **Skill Preferences: Learning to Extract and Execute Robotic Skills from Human Feedback**. <br> Wang X, Lee K, Hakhamaneshi K, et al. (**CoRL 2021**) [[paper]](https://proceedings.mlr.press/v164/wang22g.html)

- **Controlled Diversity with Preference : Towards Learning a Diverse Set of Desired Skills**. <br> Hussonnois M, Karimpanal T G, Rana S. (**AAMAS 2023**) [[paper]](https://arxiv.org/abs/2303.04592) 

### <u> (Learning from Demonstrations) Learning with diverse sources of feedbacks </u>

- **Learning reward functions from diverse sources of human feedback: Optimally integrating demonstrations and preferences**. <br>Bıyık E, Losey D P, Palan M, et al. (**IJRR 2022**) [[paper]](https://www.researchgate.net/profile/Erdem-Biyik/publication/354194846_Learning_reward_functions_from_diverse_sources_of_human_feedback_Optimally_integrating_demonstrations_and_preferences/links/612e14af0360302a006cc309/Learning-reward-functions-from-diverse-sources-of-human-feedback-Optimally-integrating-demonstrations-and-preferences.pdf)

### <u> (Learning from Demonstrations) Extrapolating beyond the demonstrations using preferences </u>

- **Extrapolating beyond suboptimal demonstrations via inverse reinforcement learning from observations**. <br> Brown D, Goo W, Nagarajan P, et al. (**ICML 2019**) [[paper]](https://proceedings.mlr.press/v97/brown19a.html)
  
- **Better-than-Demonstrator Imitation Learning via Automatically-Ranked Demonstrations**. <br> Brown D S, Goo W, Niekum S. (**CoRL 2020**) [[paper]](http://proceedings.mlr.press/v100/brown20a.html)
  

### <u> (LLM) Align with human preferences for LLM / RLHF </u>

- **Learning to summarize from human feedback**. <br> Stiennon N, Ouyang L, Wu J, et al. (**NIPS 2020**) [[paper]](https://proceedings.neurips.cc/paper/2020/hash/1f89885d556929e98d3ef9b86448f951-Abstract.html)

- **Training language models to follow instructions with human feedback**. <br> Ouyang L, Wu J, Jiang X, et al. (**NIPS 2022**) [[paper]](https://proceedings.neurips.cc/paper_files/paper/2022/hash/b1efde53be364a73914f58805a001731-Abstract-Conference.html) 

- **Chain of Hindsight Aligns Language Models with Feedback**. <br> Hao Liu, Carmelo Sferrazza, Pieter Abbeel (**Preprint 2023**) [[paper]](https://arxiv.org/abs/2302.02676) 

- **RRHF: Rank Responses to Align Language Models with Human Feedback without tears**. <br> Yuan Z, Yuan H, Tan C, et al. (**Preprint 2023**) [[paper]](https://arxiv.org/abs/2304.05302)

- **Shattering the Agent-Environment Interface for Fine-Tuning Inclusive Language Models**. <br> Xu W, Dong S, Arumugam D, et al. (**Preprint 2023**) [[paper]](https://arxiv.org/abs/2305.11455)


### <u> Theory  </u>

- **Dueling RL: Reinforcement Learning with Trajectory Preferences**. <br> Pacchiano A, Saha A, Lee J. (**AISTATS 2023**) [[paper]](https://arxiv.org/abs/2111.04850) <br> * *Regret analysis of the preference-based algorithm under assumption of trajectory embedding and preferences*

- **Human-in-the-loop: Provably Efficient Preference-based Reinforcement Learning with General Function Approximation**. <br> Chen X, Zhong H, Yang Z, et al. (**ICML 2022**) [[paper]](https://proceedings.mlr.press/v162/chen22ag.html)

- **Provably Feedback-Efficient Reinforcement Learning via Active Reward Learning**. <br> Kong D, Yang L. (**NIPS 2022**) [[paper]](https://proceedings.neurips.cc/paper_files/paper/2022/hash/476c289f685e27936aa089e9d53a4213-Abstract-Conference.html) 

### <u> **Benchmarks** </u>

- **B-Pref: Benchmarking Preference-Based Reinforcement Learning** <br> Lee K, Smith L, Dragan A, et al.(**NIPS 2021 Track on Datasets and Benchmarks**) [[paper]](https://arxiv.org/abs/2111.03026)