# -*- coding: utf-8 -*-
"""
# Import Libraries
"""

import numpy as np
import pandas  as pd

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

from datetime import datetime
import itertools
import argparse
import re
import os
import pickle

from sklearn.preprocessing import StandardScaler
import pdb

"""# Data getter"""


def get_data():
    """
    Returns a T x 3 list of stock prices

    Each column are the stocks of different companies
    0 = APPL
    1 = MSI
    2 = SBUX

    Parameters:
        None

    Returns:
        list: T x 3 list of stock prices
    """

    df = pd.read_csv('aapl_msi_sbux.csv')
    return df.values


"""# Replay Buffer"""


class ReplayBuffer:
    """
    A Replay Buffer class for storing state transitions.

    Attributes:
        obs1_buf (numpy.ndarray): Current state buffer.
        obs2_buf (numpy.ndarray): Next state buffer.
        acts_buf (numpy.ndarray): Buffer for ctions represented by integers 0 to 26 inclusive.
        rews_buf (numpy.ndarray): Rewards buffer.
        done_buf (numpy.ndarray): Done flag buffer.
        ptr (int): Current index of buffers.
        size (int): Current size of buffers.
        max_size (int): Maximum size of buffers.
    """

    def __init__(self, obs_dim, act_dim, size):
        """
        Constructor for ReplayBuffer class

        Parameters:
          obs_dim (int): Dimensions of states.
          act_dim (int): Dimensions of actions.
        """
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros(size, dtype=np.uint8)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.uint8)
        self.ptr = 0
        self.size = 0
        self.max_size = size

    def store(self, obs, act, rew, next_obs, done):
        """Stores new values for ReplayBuffer member variables"""
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        """
        Chooses random indices from 0 up to the size of the buffer.

        Parameters:
          batch_size (int): Size of batch over which gradient descent is calculated.

        Returns:
          dict: Dictionary containing states, actions, rewards, and done flags that are indexed by the random indices.
        """

        idxs = np.random.randint(0, self.size, size=batch_size)

        return dict(s=self.obs1_buf[idxs],
                    s2=self.obs2_buf[idxs],
                    a=self.acts_buf[idxs],
                    r=self.rews_buf[idxs],
                    d=self.done_buf[idxs])


"""# Scaler"""


# Scaler Function
def get_scaler(env):
    """
    Creates scaler for scaling states.

    Plays random episodes to get states used for creating state scaler.
    StandardScaler scales data by substracting it by its mean, and dividing the result by the data's standard deviation.

    Parameters:
        env (object): Environment object used for fitting scaler.

    Returns:
        StandardScaler: Scaler used to scale states.
    """

    states = []
    for _ in range(env.n_step):
        action = np.random.choice(env.action_space)
        state, reward, done, info = env.step(action)
        states.append(state)
        if done:
            break

    scaler = StandardScaler()
    scaler.fit(states)

    return scaler

"""# Directory Creator"""


# Make directory function
def try_to_make_dir(directory):
    """
    Creates directory if it doesn't already exist

    Parameters:
        directory (string): Directory address.
    """

    if not os.path.exists(directory):
        os.makedirs(directory)


"""# Model Builder"""


# Function for creating neural network model
def mlp(input_dim, n_action, n_hidden_layers=1, hidden_dim=32):
    """
    Builds and returns neural network model.

    Parameters:
        input_dim (tuple): Dimensions of input data.
        n_action (int): Number of actions.
        n_hidden_layers (int): Number of hidden layers.
        hidden_dim (int): Number of neurons for hidden layers.

    Returns:
        Model: Neural network model.
    """

    # Input layer
    i = Input(shape=(input_dim,))
    x = i

    # Hidden layers
    for _ in range(n_hidden_layers):
        Dense(hidden_dim, activation='relu')(x)

    # Output Layer
    # Linear regression, so no activation function for output layer
    x = Dense(n_action)(x)

    # Make and compile model
    model = Model(i, x)

    model.compile(optimizer='adam',
                loss='mse',)

    print(model.summary())

    return model


"""# Multi Stock Environment"""


class MultiStockEnv:
    """
    A 3-stock trading environment class.

    State: vector of size 7 (n_stock * 2 + 1)
        - # shares of stock 1 owned
        - # shares of stock 2 owned
        - # shares of stock 3 owned
        - closing price of stock 1
        - closing price of stock 2
        - closing price of stock 3

    Action: categorical variable with 27 (3^3) possibilities
        - for each stock, you can:
        - 0 = sell
        - 1 = hold
        - 2 = buy

    Attributes:
        stock_price_history (numpy.ndarray): Stock price history data.
        n_step (int): Steps/epochs of data.
        n_stock (int): Number of stocks.
        initial_investment (int): Initial amount of cash available to invest.
        cur_step (int): Current step/epoch
        stock_owned (numpy.ndarray): Number of each stock owned.
        stock_price (numpy.ndarray): Price of each stock.
        cash_in_hand (int): Cash available for investment.
        action_space (numpy.ndarray): Number of possible actions able to be taken with 3 stocks.
        action_list (list): List of possible actions able to be taken with 3 stocks.
        state_dim (int): Size of state.
    """

    def __init__(self, data, initial_investment=20000):
        """
        The Constructor for MultiStockEnv  class.

        Parameters:
          data (numpy.ndarray): Stock price history data.
          initial_investment (float): Initial amount of cash available to invest.
        """

        # Data initialization
        self.stock_price_history = data
        self.n_step, self.n_stock = self.stock_price_history.shape

        # Attribute initialization
        self.initial_investment = initial_investment
        self.cur_step = None
        self.stock_owned = None
        self.stock_price = None
        self.cash_in_hand = None

        # 3 ^ 4
        self.action_space = np.arange(3 ** self.n_stock)

        # Cartesion product of actions able to be taken with 3 stocks
        # returns a nested list, like so:
        #[[0, 0, 0], [0, 0, 1], ... , [0, 1, 1], .etc]
        self.action_list = list(map(list, itertools.product([0, 1, 2], repeat=self.n_stock)))

        # Calculate size of state
        self.state_dim = self.n_stock * 2 + 1

        self.reset()

    def reset(self):
        """Initializes attributes and returns initial state"""
        self.cur_step = 0  # Point to first day
        self.stock_owned = np.zeros(self.n_stock)  # Initially don't own any stock
        self.stock_price = self.stock_price_history[self.cur_step]  # Current price of stocks
        self.cash_in_hand = self.initial_investment

        # return the state
        return self._get_obs()

    def step(self, action):
        """
        Performs action in current step.

        Performs an action in the environment, and returns next state and reward.

        Parameters:
            action (list): Action to be performed on stocks in environment.
        """

        # check that the action exists in possible range of actions
        assert action in self.action_space

        # get current portfolio value
        prev_val = self._get_val()

        # update current stock prices
        self.cur_step += 1
        self.stock_price = self.stock_price_history[self.cur_step]

        # perform trade
        self._trade(action)

        # get new portfolio value after taking action
        cur_val = self._get_val()

        # reward is increase in portfolio value
        reward = cur_val - prev_val

        # done flag set if episode is over
        done = self.cur_step == self.n_step - 1

        # store new current value of portfolio (not part of state or reward)
        info = {'cur_val': cur_val}

        # conform to Gym API
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        """Returns the current state"""
        obs = np.empty(self.state_dim) # make empty state
        obs[:self.n_stock] = self.stock_owned # list of size 3 of the number of shares of each stock that we own
        obs[self.n_stock:2*self.n_stock] = self.stock_price # list of size 3 of the price of each stock
        obs[-1] = self.cash_in_hand # last element is cash_in_hand attribute
        return obs

    def _get_val(self):
        """Return current value of portfolio"""
        return self.stock_owned.dot(self.stock_price) + self.cash_in_hand

    def _trade(self, action):
        """
        Performs trade given an action.

        Performs a trade in the multistock environment given an action index.

        Parameters:
            action (list): Action to be performed on stocks in environment.
        """

        # Get action vector of current action from action list
        action_vec = self.action_list[action]

        # determine which stocks to buy or sell
        sell_idx = []
        buy_idx = []
        for i, a in enumerate(action_vec):
            if a == 0:
                sell_idx.append(i)
            if a == 2:
                buy_idx.append(i)

        # sell stocks, then buy stocks
        if sell_idx:
            # Sell all stocks we want to sell
            for i in sell_idx:
                self.cash_in_hand += self.stock_price[i] * self.stock_owned[i]
                self.stock_owned[i] = 0
        if buy_idx:
            # buy a share of each stock with round robin scheduling
            # (i.e., loop through each stock and buy a share, repeat until no more cash_in_hand available)
            can_buy = True
            while can_buy:
                for i in buy_idx:
                    if self.cash_in_hand >= self.stock_price[i]:
                        self.cash_in_hand -= self.stock_price[i]
                        self.stock_owned[i] += 1
                    else:
                        can_buy = False


"""# Agent"""


class DQNAgent(object):
    """
    Artificial Intelligence (AI) agent.

    Takes past experiences, learns from them,
    and takes actions such that they will maximize future rewards.

    Attributes:
        state_size (int): Size of state (number of inputs to neural network).
        action_size (int): Size of action (number of outputs to neural network).
        memory (ReplayBuffer): Replay buffer.
        gamma (float): Discount rate (hyperparameter).
        epsilon (float): Exploration rate (hyperparameter).
        epsilon_min (float): Minimum value of exploration rate (hyperparameter).
        epsilon_decay (float): Factor to change exploration rate by on each round (hyperparameter).
        model (Model): Neural network model.
    """
    def __init__(self, state_size, action_size):
        """
        The Constructor for DQN class.

        Parameters:
          state_size (int): Size of state.
          action_size (int): Size of action.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayBuffer(state_size, action_size, size=500)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = mlp(state_size, action_size)

    def update_replay_memory(self, state, action, reward, next_state, done):
        """Updates Replay Buffer"""
        self.memory.store(state, action, reward, next_state, done)

    def act(self, state):
        """
        Determine action given state.

        Determine action given state by using Epsilon Greedy algorithm.
        https://www.geeksforgeeks.org/epsilon-greedy-algorithm-in-reinforcement-learning/
        """

        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        act_vals = self.model.predict(state)
        return np.argmax(act_vals[0])

    def replay(self, batch_size=32):
        """
        Does learning for the agent

        Parameters:
          batch_size (int): Size of batch over which gradient is calculated.
        """

        # Check that replay buffer contains enough to data for batch
        if self.memory.size < batch_size:
            return

        # Get sample batch from replay buffer
        mini_batch = self.memory.sample_batch(batch_size)
        states = mini_batch['s']
        actions = mini_batch['a']
        rewards = mini_batch['r']
        next_states = mini_batch['s2']
        done = mini_batch['d']

        # Calculate tentative target (Expected value of future returns given next state and current action): Q(s', a)
        target = rewards + self.gamma * np.amax(self.model.predict(next_states), axis=1)

        # Set target of terminal state as reward, because terminal state value is 0
        target[done] = rewards[done]

        # Set target to be equal to the prediction for all values
        # Then only change the targets for the actions taken.
        # Q(s, a)
        #pdb.set_trace()
        target_full = self.model.predict(states) # Batch size x Number of actions
        target_full[np.arange(batch_size), actions] = target

        # Run one training step of gradient descent
        self.model.train_on_batch(states, target_full)

        # Decrease amount of exploration by factor of epsilon_decay (reduce likelihood of choosing a random action)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        """Load the model"""
        self.model.load_weights(name)

    def save(self, name):
        """Save the model"""
        self.model.save_weights(name)

def play_one_episode(agent, env, is_train):
    """
    Plays one episode with the deep reinforcment learning agent.

    Parameters:
      agent (DQNAgent): Agent that is learning to trade stocks.
      env (MultiStockEnv): Multi stock trading environment.
      is_train (bool): Flag indicating whether training is occurring.

    Returns:
      float: Current portfolio value.
    """

    # Reset environment, transform state, and reset done flag
    state = env.reset()
    state = scaler.transform([state]) # state is now 1 x D
    done = False

    # Play episode until it has terminated
    while not done:
        # retrieve agent action given current state
        action = agent.act(state)

        # perform action in environment
        next_state, reward, done, info = env.step(action)

        # scale next state
        next_state = scaler.transform([next_state]) # state is now 1 x D

        # check if training
        if is_train == 'train':

            # update replay buffer with latest state transition
            agent.update_replay_memory(state, action, reward, next_state, done)

            # call replay function to run 1 step of gradient descent
            #pdb.set_trace()
            agent.replay(batch_size)

        # update state
        state = next_state

    return info['cur_val']


"""# Main Function"""

if __name__ == '__main__':

    # set configurations for model
    models_folder = 'rl_trader_models'
    rewards_folder = 'rl_trader_rewards'
    num_episodes = 2000
    batch_size = 32
    initial_investment = 20000

    # create parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, required=True, help='either "train" or "test"')
    args = parser.parse_args()

    # try to make directories
    try_to_make_dir(models_folder)
    try_to_make_dir(rewards_folder)

    # load data
    data = get_data()

    # data preprocessing

    ## Train Test Split
    n_time_steps, n_stocks = data.shape
    n_train = n_time_steps // 2
    train_data = data[:n_train]
    test_data = data[n_train:]

    ## initialize environment
    env = MultiStockEnv(train_data, initial_investment)

    ## get dimensionality of state and action
    state_size = env.state_dim
    action_size = len(env.action_space)

    ## initialize agent, scaler, portfolio value list
    agent = DQNAgent(state_size, action_size)
    scaler = get_scaler(env)


    # store the final value of the portfolio (end of episode)
    portfolio_value = []

    # test model
    if args.mode == 'test':
        # get training scaler
        with open(f'{models_folder}/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)

        # reinitialize environment with test data
        env = MultiStockEnv(test_data, initial_investment)

        # initialize exploration factor
        agent.epsilon = 0.01

        # load training weights
        agent.load(f'{models_folder}/dqn.h5')

    # play game num_episodes times
    for e in range(num_episodes):
        # get current time
        t_0 = datetime.now()

        # play episode and get portfolio value
        val = play_one_episode(agent, env, args.mode)

        # get time passed by end of episode
        dt = datetime.now() - t_0
        print(f"episode: {e + 1}/{num_episodes}, episode end value: {val:.2f}, duration: {dt}")

        ## append portfolio value
        portfolio_value.append(val)

    # train model
    if args.mode == 'train':
        ## save DQN
        agent.save(f'{models_folder}/dqn.h5')

        ## save training scaler
        with open(f'{models_folder}/scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)


    # save portfolio value for each episode
    np.save(f'{rewards_folder}/{args.mode}.npy', portfolio_value)
