import json
import logging
import pathlib
import time
from collections import deque

import numpy as np
import torch
import typing
import unityagents
import pandas as pd

from drl_cc import agents
from drl_cc import config as cfg
from drl_cc import path_util


def training(
        env: unityagents.UnityEnvironment,
        output_dir: typing.Union[pathlib.Path, str],
        agent_type: str = "DDPG",
        buffer_size: int = 100_000,
        batch_size: int = 128,
        gamma_discount_factor: float = 0.95,
        tau_soft_update: float = 1e-3,
        learning_rate_actor: float = 2e-3,
        learning_rate_critic: float = 1e-3,
        l2_weight_decay: float = 0.0,
        update_every: int = 10,
        num_updates: int = 20,
        has_ou_noise: bool = True,
        ou_noise_mu: float = 0.0,
        ou_noise_theta: float = 0.15,
        ou_noise_sigma_start: float = 0.5,
        ou_noise_sigma_end: float = 0.01,
        ou_noise_sigma_decay: float = 0.999,
        n_episodes: int = 500,
        mean_score_threshold: float = 30.0,
        max_t: int = 1000,
        n_random_episodes: int = 100,
        agent_seed=111_111,
        logging_freq: int = 10):
    """
    Train agent for Unity Tennis environment and save results.

    Train a deep reinforcement learning Tennis agent and
    save results (training scores, time taken by episode and total,
    agent neural network model weights, metadata with hyper-parameters)
    to provided output directory.

    Parameters
    ----------
    env
        Unity environment
    output_dir
        Path to output results output directory (scores, weights, metadata)
    agent_type
        A type of agent to train from the available ones
    buffer_size
        Maximum size of buffer for storing experiences
    batch_size
        Size of Each training batch
    gamma_discount_factor
        Discount factor
    tau_soft_update
        Interpolation parameter for soft network weight update
    learning_rate_actor
        Learning rate for Actor network
    learning_rate_critic
        Learning rate for Critic network
    l2_weight_decay
        Weight decay for critic optimizer
    update_every
        Update weights of networks every `update_every` time steps
    num_updates
        Number of simultaneous updates
    has_ou_noise
        If True, Ornstein-Uhlenbeck noise is added to actions
    ou_noise_mu
        Ornstein-Uhlenbeck process mu parameter
    ou_noise_theta
        Ornstein-Uhlenbeck process theta parameter
    ou_noise_sigma_start
        Ornstein-Uhlenbeck noise sigma starting value per episode
    ou_noise_sigma_end
        Ornstein-Uhlenbeck noise sigma minimum value per episode
    ou_noise_sigma_decay
        Ornstein-Uhlenbeck noise sigma multiplicative decay
    n_episodes
        Maximum number of episodes
    mean_score_threshold
        Threshold of mean last 100 weights to stop training and save results
    max_t:
        Maximum number of time steps per episode
    n_random_episodes
        Number of random episodes to gather experience
    agent_seed
        Random seed for agent epsilon-greedy policy
    logging_freq
        Logging frequency

    """
    logger = logging.getLogger(__name__)

    output_dir = pathlib.Path(output_dir)

    logger.info(f"Ensuring output directory exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    path_weights_actor = path_util.mk_path_weights_actor(output_dir)
    path_weights_critic = path_util.mk_path_weights_critic(output_dir)
    path_scores = path_util.mk_path_scores(output_dir)
    path_metadata = path_util.mk_path_metadata(output_dir)

    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    action_size = brain.vector_action_space_size

    env_info = env.reset(train_mode=True)[brain_name]
    state = env_info.vector_observations[0]
    state_size = len(state)

    agent = agents.DDPGAgent(
        state_size=state_size,
        action_size=action_size,
        num_agents=len(env_info.agents),
        buffer_size=buffer_size,
        batch_size=batch_size,
        gamma_discount_factor=gamma_discount_factor,
        tau_soft_update=tau_soft_update,
        learning_rate_actor=learning_rate_actor,
        learning_rate_critic=learning_rate_critic,
        l2_weight_decay=l2_weight_decay,
        update_network_every=update_every,
        num_updates=num_updates,
        ou_noise_mu=ou_noise_mu,
        ou_noise_theta=ou_noise_theta,
        seed=agent_seed)

    scores_df = train_agent(
        env=env,
        agent=agent,
        n_episodes=n_episodes,
        mean_score_threshold=mean_score_threshold,
        max_t=max_t,
        has_ou_noise=has_ou_noise,
        ou_noise_sigma_start=ou_noise_sigma_start,
        ou_noise_sigma_end=ou_noise_sigma_end,
        ou_noise_sigma_decay=ou_noise_sigma_decay,
        n_random_episodes=n_random_episodes,
        logging_freq=logging_freq)

    logger.info(f'Saving actor network model weights to {str(path_weights_actor)}')
    torch.save(agent.actor_local.state_dict(), str(path_weights_actor))
    logger.info(f'Actor model weights saved successfully!')

    logger.info(f'Saving critic network model weights to {str(path_weights_critic)}')
    torch.save(agent.critic_local.state_dict(), str(path_weights_critic))
    logger.info(f'Critic model weights saved successfully!')

    logger.info(f'Saving training scores to {str(path_scores)}')
    logger.info(f'Training scores saved successfully!')

    scores_df.to_csv(path_scores, index=False)

    logger.info(f'Saving training metadata to {str(path_metadata)}')
    metadata = {
        "agent_type": agent_type,
        "agent": agent.metadata,
        "mean_score_threshold": mean_score_threshold,
        "max_t": max_t,
        "has_ou_noise": has_ou_noise,
    }

    with open(path_metadata, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f'Training metadata saved successfully!')


def train_agent(
        env: unityagents.UnityEnvironment,
        agent: agents.DDPGAgent,
        n_episodes: int = 200,
        mean_score_threshold: float = 30.0,
        max_t: int = 1000,
        has_ou_noise: bool = True,
        scores_maxlen: int = 100,
        ou_noise_sigma_start: float = 0.5,
        ou_noise_sigma_end: float = 0.01,
        ou_noise_sigma_decay: float = 0.99,
        n_random_episodes: int = 100,
        logging_freq: int = 10
) -> pd.DataFrame:
    """
    Train agent for Unity Tennis environment and return results.

    Parameters
    ----------
    env
        Unity environment
    agent
        And instance of Deep Reinforcement Learning Agent from drl_ctrl.agents module
    n_episodes
        Maximum number of episodes
    mean_score_threshold
        Threshold of mean last 100 weights to stop training and save results
    max_t
        Maximum number of time steps per episode
    has_ou_noise
        If True, Ornstein-Uhlenbeck noise is added to actions
    scores_maxlen
        Maximum length of scores window
    ou_noise_sigma_start
        Ornstein-Uhlenbeck noise sigma starting value per episode
    ou_noise_sigma_end
        Ornstein-Uhlenbeck noise sigma minimum value per episode
    ou_noise_sigma_decay
        Ornstein-Uhlenbeck noise sigma multiplicative decay
    n_random_episodes
        Number of random episodes to gather experience
    logging_freq
        Logging frequency

    """

    logger = logging.getLogger(__name__)

    scores = []
    scores_avg100 = []
    scores_window = deque(maxlen=scores_maxlen)
    time_started = time.time()
    times_total = []
    times_per_episode = []
    time_steps = []

    for i_episode in range(1, (n_random_episodes + n_episodes + 1)):

        time_started_episode = time.time()

        brain_name = env.brain_names[0]
        env_info = env.reset(train_mode=True)[brain_name]
        agent.reset()
        states = env_info.vector_observations
        num_agents = len(env_info.agents)
        agent_scores = np.zeros(num_agents)

        ou_noise_sigma = ou_noise_sigma_start

        t = 1
        while True:
            # choose action (for each agent)
            if i_episode <= n_random_episodes:
                action_size = env.brains[brain_name].vector_action_space_size
                actions = np.random.randn(num_agents, action_size)
                actions = np.clip(actions, -1, 1)
            else:
                actions = agent.act(states, ou_noise_sigma=ou_noise_sigma, add_noise=has_ou_noise)
            ou_noise_sigma = max(ou_noise_sigma_end, ou_noise_sigma * ou_noise_sigma_decay)

            # take action in the environment(for each agent)
            env_info = env.step(actions)[brain_name]

            # get next state (for each agent)
            next_states = env_info.vector_observations

            # see if episode finished
            dones = env_info.local_done

            # update the score (for each agent)
            agent_scores += env_info.rewards

            if i_episode <= n_random_episodes:
                agent.memory.add_batch(states, actions, env_info.rewards, next_states, dones)
            else:
                agent.step(states, actions, env_info.rewards, next_states, dones)

            # roll over states to next time step
            states = next_states

            # exit loop if episode finished
            if np.any(dones):
                break
            t += 1

        score = float(np.max(agent_scores))
        scores_window.append(score)
        scores.append(score)
        scores_avg100.append(np.mean(scores_window))

        times_total.append(time.time() - time_started)
        times_per_episode.append(time.time() - time_started_episode)
        time_steps.append(t)

        if i_episode % logging_freq == 0:
            logger.info(
                f'\rEp: {i_episode}'
                f'\tSigma({t}): {ou_noise_sigma:.3f}'
                f'\tScore: {score:.2f}'
                f'\tAvg. Score: {np.mean(scores_window):.2f}'
                f'\tTime_e: {times_per_episode[-1]:.3f}s'
                f'\tTime: {times_total[-1]:.3f}s')

        if len(scores_window) == scores_maxlen and np.mean(scores_window) >= mean_score_threshold:
            logger.info(
                f'\nEnvironment solved in {i_episode-100:d} episodes!'
                f'\nScore: {score:.2f}'
                f'\tAverage Score: {np.mean(scores_window):.2f}'
                f'\tAverage Time_e: {np.mean(times_per_episode):.3f}s'
                f'\tTotal Time: {times_total[-1]:.3f}s')
            break

    return pd.DataFrame.from_records(
        zip(range(len(scores)), scores, scores_avg100, time_steps, times_per_episode, times_total),
        columns=[
            cfg.COL_EPISODE,
            cfg.COL_SCORE,
            cfg.COL_SCORE_AVG100,
            cfg.COL_N_TIME_STEPS,
            cfg.COL_TIME_PER_EPISODE,
            cfg.COL_TIME_TOTAL
        ])


def demo(
        env: unityagents.UnityEnvironment,
        dir_model: typing.Optional[pathlib.Path] = None
) -> float:
    """
    Run a demo on the environment

    Parameters
    ----------
    env
        Unity Environment
    dir_model
        If provided, agent model weights are loaded from path,
        Random Agent is used otherwise

    Returns
    -------
    float
        final score

    """
    if dir_model is not None:
        brain_name = env.brain_names[0]
        brain = env.brains[brain_name]
        action_size = brain.vector_action_space_size
        env_info = env.reset(train_mode=False)[brain_name]
        state = env_info.vector_observations[0]
        state_size = len(state)

        agent = agents.DDPGAgent(state_size=state_size, action_size=action_size, num_agents=20)
        agent.actor_local.load_state_dict(torch.load(path_util.mk_path_weights_actor(dir_model)))
        agent.critic_local.load_state_dict(torch.load(path_util.mk_path_weights_critic(dir_model)))

        return demo_trained(env, agent)
    else:
        return demo_random(env)


def demo_trained(env: unityagents.UnityEnvironment, agent: agents.DDPGAgent) -> float:
    """
    Run a demo of a trained agent

    Parameters
    ----------
    env
        Unity Environment
    agent
        trained agent

    Returns
    -------
    float
        final score

    """

    brain_name = env.brain_names[0]
    env_info = env.reset(train_mode=False)[brain_name]
    states = env_info.vector_observations
    num_agents = len(env_info.agents)
    scores = np.zeros(num_agents)
    while True:
        actions = agent.act(states)
        # send all actions to the environment
        env_info = env.step(actions)[brain_name]
        # get next state (for each agent)
        next_states = env_info.vector_observations
        # see if episode finished
        dones = env_info.local_done
        # update the score (for each agent)
        scores += env_info.rewards
        # roll over states to next time step
        states = next_states
        # exit loop if episode finished
        if np.any(dones):
            break

    score = float(np.mean(scores))
    print('Total score (averaged over agents) this episode: {}'.format(score))
    return score


def demo_random(env: unityagents.UnityEnvironment) -> float:
    """
    Run a demo of a Random Agent

    Parameters
    ----------
    env
        Unity Environment

    Returns
    -------
    float
        final score

    """
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    action_size = brain.vector_action_space_size

    env_info = env.reset(train_mode=False)[brain_name]
    states = env_info.vector_observations
    num_agents = len(env_info.agents)
    scores = np.zeros(num_agents)
    i_step = 1
    while True:
        actions = np.random.randn(num_agents, action_size)  # select an action (for each agent)
        actions = np.clip(actions, -1, 1)  # all actions between -1 and 1
        env_info = env.step(actions)[brain_name]  # send all actions to the environment
        next_states = env_info.vector_observations  # get next state (for each agent)
        dones = env_info.local_done  # see if episode finished
        scores += env_info.rewards  # update the score (for each agent)
        states = next_states  # roll over states to next time step
        if np.any(dones):  # exit loop if episode finished
            print(f'Episode finished in {i_step} steps')
            break
        i_step += 1

    score = float(np.mean(scores))
    print('Total score (averaged over agents) this episode: {}'.format(score))
    return score
