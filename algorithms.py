import gym
from typing import Optional
from collections import defaultdict, Sequence
from collections import defaultdict
from typing import Callable, Tuple
from typing import Optional, Sequence
import numpy as np
import tqdm 
import random
def generate_episode(env: gym.Env, policy: Callable, es: bool = False):
    """A function to generate one episode and collect the sequence of (s, a, r) tuples

    This function will be useful for implementing the MC methods

    Args:
        env (gym.Env): a Gym API compatible environment
        policy (Callable): A function that represents the policy.
        es (bool): Whether to use exploring starts or not
    """
    episode = []
    state = env.reset()[0]
    i = 0
    while True:
        if es and len(episode) == 0:
            # print("initial")
            action = env.action_space.sample()
        else:
            # print("State",state)
            action = policy(state)
            # print("action",action)

        # print(env.step(action))
        next_state, reward, done, _= env.step(action)
        # print(state)
        episode.append((state, action, reward))
        if done:
            break
        state = next_state
        i=i+1

    return episode

def argmax(arr):
    """Argmax that breaks ties randomly

    Takes in a list of values and returns the index of the item with the highest value, breaking ties randomly.

    Note: np.argmax returns the first index that matches the maximum, so we define this method to use in EpsilonGreedy and UCB agents.
    Args:
        arr: sequence of values

    """
    
    max = np.max(arr)
    index = np.where(arr == max)[0]
    
    

    if len(index)>1:
        index_choice = np.random.choice(index)
        # print("Choice of index: ",index_choice)
        return index_choice
    else:
        return index[0]

def choose_action(state, Q, epsilon, action_space):
    if np.random.uniform(0, 1) < epsilon:
        print("State Random",action_space.sample())
        return action_space.sample()
        
    else:
        print("State",state)
        return argmax(Q[state])
    
    

def create_epsilon_policy_1(Q: defaultdict, epsilon: float,env) -> Callable:
    """Creates an epsilon soft policy from Q values.

    A policy is represented as a function here because the policies are simple. More complex policies can be represented using classes.

    Args:
        Q (defaultdict): current Q-values
        epsilon (float): softness parameter
    Returns:
        get_action (Callable): Takes a state as input and outputs an action
    """
    # Get number of actions
    num_actions = len(env.action_space)
    # print(num_actions)
    
    # action = argmax(Q[state_tuple])

    def get_action(state: Tuple) -> int:
        # TODO
        state = tuple(state)
        # You can reuse code from ex1
        # Make sure to break ties arbitrarily
        if np.random.random() < epsilon:
            action = np.random.randint(num_actions)
        else:
            action = argmax(Q[state])
            print("action",action)
            #argmax(Q[state])

        return action   
    return get_action


def create_epsilon_policy(Q: defaultdict, epsilon: float, env) -> Callable:
    """
    Creates an epsilon-soft policy from Q values.
    
    Args:
        Q (defaultdict): current Q-values, where Q-values are stored in an array with indices corresponding to actions.
        epsilon (float): softness parameter, controlling the trade-off between exploration and exploitation.
        env: Environment object that contains the action space and its corresponding indices.
    
    Returns:
        get_action (Callable): Takes a state as input and outputs an action.
    """
    # Get the number of actions from the environment
    num_actions = len(env.action_space)
    
    def get_action(state: Tuple) -> Tuple:
        """Function to choose an action for the given state using an epsilon-greedy policy."""
        if np.random.random() < epsilon:
            # Explore: Randomly select an action
            action_idx = np.random.randint(num_actions)
        else:
            # Exploit: Choose the action with the highest Q-value
            action_idx = argmax(Q[tuple(state)])  # Directly using np.argmax to find the index with the highest Q-value

        # Convert index back to action tuple
        action = env.action_space[action_idx]
        # print("Action",action)
        return action
    
    return get_action

def expected_q(Q, state, epsilon, nA):
    expected_q = 0.0
    # Calculate probabilities for each action
    action_probs = np.ones(nA) * epsilon / nA
    best_action = np.argmax(Q[state])
    action_probs[best_action] += (1.0 - epsilon)
    # Calculate the expected Q value
    for action in range(nA):
        expected_q += action_probs[action] * Q[state][action]
    return expected_q
    
def getQ(env):
    num_rows = 10
    num_cols = 7

    # Initialize the dictionary
    Q = {}

    # Populate the dictionary with keys representing row x column pairs and values as zeros
    for row in range(num_rows):
        for col in range(num_cols):
            Q[(row, col)] = np.zeros(env.action_space.n)      
    return Q

def initialize_Q(env, mu=0, sigma=0.1):
    Q = {}
    for state in env.state_space:  # Ensure state is hashable, convert if needed (e.g., tuple(state) if it's a list)
        print(state)
        state = tuple(state)
        Q[state] = {}
        for action in env.action_space:
            # Initialize Q-values from a Gaussian distribution
            Q[state][action] = np.random.normal(mu, sigma)

    print(Q)        
    return Q

def sarsa(env: gym.Env, num_steps: int, gamma: float, epsilon: float):
    """SARSA algorithm.

    Args:
        env (gym.Env): a Gym API compatible environment
        num_steps (int): Number of steps
        gamma (float): Discount factor of MDP
        epsilon (float): epsilon for epsilon greedy
        step_size (float): step size
    """
    # TODO

    # Q = defaultdict(lambda: np.zeros(len(env.action_space)))
    #Q = defaultdict(lambda: defaultdict(float))
    # Q = defaultdict(lambda: defaultdict(lambda: np.random.normal(loc=0.0, scale=1.0, size=len(env.action_space))))
    # Q = defaultdict(lambda: defaultdict(lambda: np.random.normal(loc=0.0, scale=1.0)))
    Q = defaultdict(lambda: np.zeros(len(env.action_space)))
    # Q = initialize_Q(env, mu=0, sigma=0.1)
    # print(env.action_space)
    #Q = np.zeros((10,7))
    # Q = np.zeros((env.observation_space[0].n,env.observation_space[1].n, env.action_space.n))
    # Q = getQ(env)

    alpha = 0.2
    history = []
    pbar = tqdm.trange(num_steps)
    # for t_total in pbar:
    for i in pbar:
        state = env.reset()
        
        # state = tuple(state)
        # print("STATE",state)
        # action = choose_action(state, Q, gamma, env.action_space)
        policy = create_epsilon_policy(Q,epsilon,env)
        action = policy(state)
        # action = env.action_space[action_ret]
        
        total_reward = 0
        done = False
        j=0
        while not done:

            # print("-_________________________________")
            # print("state {} action {}".format(state,action))
            next_state, reward, done,_= env.step(action)
            state = tuple(state)
            # action = action
            # try:
            #     action = (action,) 
            # except:
            #     print(action,type(action))   
            next_state = tuple(next_state)
            policy = create_epsilon_policy(Q,epsilon,env)
            next_action= policy(next_state)
            # next_action = env.action_space[next_action_ret]
            # try:
            #     next_action = (next_action,) 
            # except:
            #     #if isinstance(next_action, np.int64) else tuple(next_action)    
            #     print(next_action,type(next_action))   
            # next_action = argmax(Q[next_state])
            action_idx = env.action_index[action]
            next_action_idx = env.action_index[next_action]
            Q[state][action_idx] += alpha*(reward+gamma*Q[next_state][next_action_idx] - Q[state][action_idx])
            # print("Q: {} current state : {} current action {} ".format(Q[state][action_idx],state,action))
            # print("Q next: {} next state : {} next action {} ".format(Q[next_state][next_action_idx],next_state,next_action))
            # print(Q)
            #print(Q)

            if done:
                break

            state = next_state
            action = next_action
            # history[i] +=1
            total_reward += reward
            j+=1
        history.append(total_reward)
    return Q, history



def nstep_sarsa(
    env: gym.Env,
    num_steps: int,
    gamma: float,
    epsilon: float,
    step_size: float,
):
    """N-step SARSA

    Args:
        env (gym.Env): a Gym API compatible environment
        num_steps (int): Number of steps
        gamma (float): Discount factor of MDP
        epsilon (float): epsilon for epsilon greedy
        step_size (float): step size
    """
    # TODO
    alpha = 0.5
    v  = defaultdict(lambda: np.zeros(env.action_space.n))
    n=4
    history = []
    for i in range(num_steps):

        states = np.zeros(n+1,dtype=object)
        rewards = np.zeros(n+1,dtype=object)

        state = env.reset()
        # states[0]=state

        T = float('inf')
        t =0
        ta = 0
        steps = 0
        while True:
            if t<T:
                policy = create_epsilon_policy(v,epsilon,env)
                action = policy(state)
                next_state,reward, done, _ = env.step(action)
                states[(t + 1) % (n + 1)] = next_state
                rewards[(t + 1) % (n + 1)] = reward
                
                if done:
                    T = t + 1

            ta = t-n+1

            if ta>=0:

                g=sum([gamma**(i - ta - 1) * rewards[i % (n + 1)] for i in range(ta + 1, min(ta + n, T) + 1)])
                if ta + n < T:
                    g=g+gamma**n*v[states[(ta+n) % (n+1)]]
                s_ta = states[ta%(n+1)]
                v[s_ta] = v[s_ta]+alpha*(g-v[s_ta])


            if ta ==(T-1):
                break

            t=t+1
            steps+=1
        history.append(steps)

    return v,history          



def exp_sarsa(
    env: gym.Env,
    num_steps: int,
    gamma: float,
    epsilon: float,
    step_size: float,
):
    """Expected SARSA

    Args:
        env (gym.Env): a Gym API compatible environment
        num_steps (int): Number of steps
        gamma (float): Discount factor of MDP
        epsilon (float): epsilon for epsilon greedy
        step_size (float): step size
    """
    # TODO
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    alpha = 0.5
    history = []
    for i in range(num_steps):
        state = env.reset()[0]
        policy = create_epsilon_policy(Q,epsilon,env)
        action = policy(state)
        steps = 0
        done = False
        j=0

        while j<500:

            next_state, reward, done, _ = env.step(action)
            policy = create_epsilon_policy(Q,epsilon,env)
            next_action = policy(next_state)
            # next_action = choose_action(next_state, Q, epsilon, env.action_space)
            exp_q = expected_q(Q,next_state,epsilon,env.action_space.n)
            Q[state][action] += alpha*(reward+gamma*exp_q - Q[state][action])

            if done:
                break

            state = next_state
            action = next_action
            j=j+1
            # history[i] +=1
            steps += 1
        history.append(steps)
    return Q, history


def q_learning(
    env: gym.Env,
    num_steps: int,
    gamma: float,
    epsilon: float,
    step_size: float,
):
    """Q-learning

    Args:
        env (gym.Env): a Gym API compatible environment
        num_steps (int): Number of steps
        gamma (float): Discount factor of MDP
        epsilon (float): epsilon for epsilon greedy
        step_size (float): step size
    """
    # TODO
    Q = getQ(env)

    alpha = 0.5
    history = []
    for i in range(num_steps):
        state = env.reset()[0]
        policy = create_epsilon_policy(Q,epsilon,env)
        action = policy(state)
        steps = 0
        done = False
        j=0
        while not done:

            next_state, reward, done, _ = env.step(action)
            policy = create_epsilon_policy(Q,epsilon,env)
            next_action = policy(next_state)

            next_action = np.argmax(Q[next_state])
            Q[state][action] += alpha*(reward+gamma*Q[next_state][next_action] - Q[state][action])

            if done:
                break

            state = next_state
            action = next_action
            j=j+1
            # history[i] +=1
            steps += 1
        history.append(steps)
    return Q, history

def on_policy_mc_control_epsilon_soft(
    env: gym.Env, num_episodes: int, gamma: float, epsilon: float
):
    """On-policy Monte Carlo policy control for epsilon soft policies.

    Args:
        env (gym.Env): a Gym API compatible environment
        num_episodes (int): Number of episodes
        gamma (float): Discount factor of MDP
        epsilon (float): Parameter for epsilon soft policy (0 <= epsilon <= 1)
    Returns:

    """
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    N = defaultdict(lambda: np.zeros(env.action_space.n))
    returns_reward = defaultdict(list)

    policy = create_epsilon_policy(Q, epsilon,env)
    history = []
    returns = np.zeros(num_episodes)
    for i in trange(num_episodes, desc="Episode", leave=False):
        # TODO Q4
        # For each episode calculate the return
        # Update Q
        # Note there is no need to update the policy here directly.
        # By updating Q, the policy will automatically be updated.
        episode = generate_episode(env,policy)
        G = 0
        steps = 0
        visited_states = set()
        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            G = gamma * G + reward
            if (state,action) not in visited_states:
                visited_states.add((state,action))
                returns_reward[(state, action)].append(G)
                # print(N[state][action])
                Q[state][action] = np.mean(returns_reward[(state, action)])
                # policy[state] = np.argmax(Q[state])
            steps+=1    

        # policy = create_epsilon_policy(Q, epsilon)
        returns[i] = G        

        history.append(steps)

    return returns,history

def td_prediction(env: gym.Env, gamma: float, episodes, n=1) -> defaultdict:
    """TD Prediction

    This generic function performs TD prediction for any n >= 1. TD(0) corresponds to n=1.

    Args:
        env (gym.Env): a Gym API compatible environment
        gamma (float): Discount factor of MDP
        episodes : the evaluation episodes. Should be a sequence of (s, a, r) tuples or a dict.
        n (int): The number of steps to use for TD update. Use n=1 for TD(0).
    """
    # TODO
    pass


def learning_targets(
    V: defaultdict, gamma: float, episodes, n: Optional[int] = None
) -> np.ndarray:
    """Compute the learning targets for the given evaluation episodes.

    This generic function computes the learning targets for Monte Carlo (n=None), TD(0) (n=1), or TD(n) (n=n).

    Args:
        V (defaultdict) : A dict of state values
        gamma (float): Discount factor of MDP
        episodes : the evaluation episodes. Should be a sequence of (s, a, r) tuples or a dict.
        n (int or None): The number of steps for the learning targets. Use n=1 for TD(0), n=None for MC.
    """
    # TODO
    targets = np.zeros(len(episodes))

    pass

def n_step_td(env, alpha, gamma, n, num_episodes):
    """
    n-step TD for estimating V ≈ v_π
    
    Args:
        env: OpenAI gym environment
        policy: a function that maps states to actions
        alpha: step size (learning rate)
        gamma: discount factor
        n: a positive integer for n-step TD
        num_episodes: number of episodes to run
    
    Returns:
        V: A dictionary (defaultdict) that estimates v_π
    """
    

    #referenced https://github.com/makaveli10/reinforcementLearning/blob/master/MultiStepBootstrapping/n_step_sarsa.py
    epsilon = 0.1
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    policy = create_epsilon_policy(Q,epsilon,env)
    history =[]
    for episode in range(num_episodes):
 
        state = env.reset()[0]
        
        T = np.inf
        tau = 0
        t = -1
        steps = 0
        stored_states = np.zeros(n+1,dtype=object)
        stored_rewards = np.zeros(n+1,dtype=object)
        stored_actions = np.zeros(n+1,dtype=object)
        stored_states[0] = state
        policy = create_epsilon_policy(Q,epsilon,env)
        action = policy(state)
        stored_actions[0] = action
        
        while tau < (T-1):
            t=t+1
            if t < T:
                # policy = create_epsilon_policy(Q,epsilon,env)
                # action = policy(state)
                
                next_state, reward, done, _ = env.step(action)
                stored_states[(t+1)%(n+1)] = next_state
                stored_rewards[(t+1)%(n+1)] = reward
                
                if done:
                    T = t + 1
                else:
                    action = policy(next_state)
                    stored_actions[(t+1) % (n+1)] = action

            tau = t - n + 1
            if tau >= 0:
                G = np.sum([gamma ** (i - tau - 1) * stored_rewards[i%(n+1)] for i in range(tau + 1, min(tau + n, T) + 1)])
                if tau + n < T:
                    G += gamma ** n * Q[stored_states[(tau + n)%(n+1)]][stored_actions[(tau+n) % (n+1)]]
                
                tau_s, tau_a = stored_states[tau % (n+1)], stored_actions[tau % (n+1)]


                Q[tau_s][tau_a] += alpha * (G - Q[tau_s][tau_a])
            steps+=1

        history.append(steps)       
            
    return Q, history

