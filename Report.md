# Report

This repository contains a simple implementation of [Deep Deterministic Policy Gradients (DDPG)](https://arxiv.org/abs/1509.02971)
and the code required to train it for Unity Tennis environment with two agents. 

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

## Learning Algorithm

Deep Deterministic Policy Gradients with implementation that works for multiple agents.


### Hyperparameters

The following hyper parameters were used for training.

#### Common hyper parameters for all agents

| parameter             | value |
| --------------------- | ----- |
| n_episodes            | 3000  |
| gamma_discount_factor | 0.95  |
| mean_score_threshold  | 0.5   |
| max_t                 | 1000  |
| learning_rate_actor   | 0.003 |
| learning_rate_critic  | 0.001 |
| tau_soft_update       | 0.003 |
| l2_weight_decay       | 0     |
| has_ou_noise          | True  |
| ou_noise_mu           | 0.0   |
| ou_noise_theta        | 0.15  |
| ou_noise_sigma_start  | 0.4   |
| ou_noise_sigma_end    | 0.05  |
| ou_noise_sigma_decay  | 0.9   |
| n_random_episodes     | 300   |
| agent_seed            | 11111 |
| logging_freq          | 1     |

#### Varying hyper parameters for different agents

Quite a few hyper-parameter variations and agent training approaches have been tried out. 
Here are the results from hyper-parameters with the most recent code changes 
(Ornstein-Uhlenbeck noise decay, initial experiences gathering from random actions);

| agent_id | batch_size | buffer_size | num_updates | update_every | n_episodes_to_solve |
| -------- | ---------- | ----------- | ----------- | ------------ | ------------------- |
| 0        | 128        | 100000      | 5           | 1            | 533                 |
| 1        | 128        | 100000      | 5           | 1            | 542                 |
| 2        | 1024       | 1000000     | 10          | 2            | 595                 |
| 3        | 1024       | 1000000     | 10          | 2            | 927                 |

We see that it's possible to solve the environment with these hyper parameter sets, 
but repeated runs show instability with the same hyper parameters in training 
that requires further investigation.

### Neural Network Model Architectures

Actor Neural Network with 3 fully connected hidden layers and batch normalization:

- fc1, in:`state_size`, out:128, relu activation
- Batch Normalization
- fc2, in: 128, out:128, relu activation
- fc3, in: 128, out: `action_size`, _tahn_ activation

here `state_size=33`, `action_size=4`.

### Critic Network Model Architectures

Critic Neural Network with 3 fully connected hidden layers and batch normalization:

- fcs1, in:`state_size`, out:128, relu activation
- Batch Normalization
- fc2, in: 128+`action_size`, out:128, relu activation
- fc3, in: 128, out: 1

here `state_size=33`, `action_size=4`, `output_size=1`

## Plot of Rewards

Initial 300 episodes have been used to gather experience from random actions and only the 
agents have began training.

![](https://github.com/daraliu/drl-collab-compet/blob/master/training_output/tuning_results/scores_all.png)

We see that multiple runs with the same hyper parameters (agents 1, 2 vs 3, 4) can show varying results. 
However, we also see that the environment be solved in less than 600 episodes (by looking at 3/4 cases).

![](https://github.com/daraliu/drl-continuous-control/blob/master/img/best_agent_so_far.png)



## Ideas for Future Work

To improve agent training stability and performance, the following steps could be taken:
- Implement Prioritised Experienced Replay
- Perform more thorough hyper parameter turing and analysis - multiple runs per hyper parameter set to evaluate their stability, do more exploration in hyper parameter space to draw better conclusions DDPG and Tennis environment.
- Implement [Multi-agent Deep Deterministic Policy Gradients (MADDPG)](https://papers.nips.cc/paper/7217-multi-agent-actor-critic-for-mixed-cooperative-competitive-environments.pdf).
- Experiment with more Neural Network architectures - evaluate trade-off between simpler networks for faster and more stable learning versus complex networks for greater score.
