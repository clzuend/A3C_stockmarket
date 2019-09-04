# Inspired by: https://github.com/Grzego/async-rl/blob/master/a3c/train.py
#from scipy.misc import imresize
#from skimage.color import rgb2gray
from multiprocessing import *
from collections import deque
import gym
import sys
import numpy as np
import h5py
import argparse
import tensorflow as tf
    
# -----
parser = argparse.ArgumentParser(description='Training model')
parser.add_argument('--game', default='market', help='OpenAI gym environment name', dest='game', type=str)
parser.add_argument('--processes', default=4, help='Number of processes that generate experience for agent',
                    dest='processes', type=int)
parser.add_argument('--lr', default=0.001, help='Learning rate', dest='learning_rate', type=float)
parser.add_argument('--steps', default=80000000, help='Number of frames to decay learning rate', dest='steps', type=int)
parser.add_argument('--batch_size', default=20, help='Batch size to use during training', dest='batch_size', type=int)
parser.add_argument('--swap_freq', default=100, help='Number of frames before swapping network weights',
                    dest='swap_freq', type=int)
parser.add_argument('--checkpoint', default=0, help='Frame to resume training', dest='checkpoint', type=int)
parser.add_argument('--save_freq', default=25000, help='Number of frames before saving weights', dest='save_freq',
                    type=int)
parser.add_argument('--queue_size', default=256, help='Size of queue holding agent experience', dest='queue_size',
                    type=int)
parser.add_argument('--n_step', default=5, help='Number of steps', dest='n_step', type=int)
parser.add_argument('--reward_scale', default=1., dest='reward_scale', type=float)
parser.add_argument('--beta', default=0.01, dest='beta', type=float)
parser.add_argument('--verbose_learner', default=1, dest='verbose_learner', type=float)
parser.add_argument('--verbose_worker', default=0, dest='verbose_worker', type=float)
parser.add_argument('--market_data', default='', dest='market_data', type=str)
parser.add_argument('--market_data_index', default='', dest='market_data_index', type=str)


# -----
args = parser.parse_args()

# -----

# Enable Market Env
sys.path.append("../../ENV")
from market_env import MarketEnv
import codecs
codeListFilename = args.market_data_index

codeMap = {}
f = codecs.open(codeListFilename, "r", "utf-8")
for line in f:
    if line.strip() != "":
        tokens = line.strip().split(",") if not "\t" in line else line.strip().split("\t")
        codeMap[tokens[0]] = tokens[1]
f.close()


def make_env(wrap=True):
    env = MarketEnv(dir_path = args.market_data, target_codes = codeMap.keys(), input_codes = [], start_date = "2010-08-25", end_date = "2015-08-25", sudden_death = -1.0)
    return env

# -----

def write_log(cb, names, logs, batch_no):
    import tensorflow as tf
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        cb.writer.add_summary(summary, batch_no)
        cb.writer.flush()


def build_network(state_shape, data_shape, output_shape):
    from keras.models import Model
    from keras.layers import Input, Concatenate, Flatten, Dense, LSTM
    from keras.utils.vis_utils import plot_model
    # -----
    '''
    Model takes two arrays as inputs - the state of the portfolio and the data - and it
    outputs three networks - the value network, the policy network and the training networt 
    '''
    try:
        state = Input(shape=state_shape)
        data = Input(shape=data_shape)
        lstm = LSTM(60)(data)
        c = Concatenate()([lstm, state])
        h = Dense(256, activation='relu')(c)
        
        value = Dense(1, activation='linear', name='value')(h)
        policy = Dense(output_shape, activation='softmax', name='policy')(h)
    except:
        print("\n Error in model definition (build_network):", sys.exc_info())
    value_network = Model(inputs=[state,data], outputs=value)
    policy_network = Model(inputs=[state,data], outputs=policy)

    advantage = Input(shape=(1,))
    train_network = Model(inputs=[state, data, advantage], outputs=[value, policy])
    return value_network, policy_network, train_network, advantage


def policy_loss(advantage=0., beta=0.01):
    from keras import backend as K

    def loss(y_true, y_pred):
        return -K.sum(K.log(K.sum(y_true * y_pred, axis=-1) + K.epsilon()) * K.flatten(advantage)) + \
               beta * K.sum(y_pred * K.log(y_pred + K.epsilon()))

    return loss


def value_loss():
    from keras import backend as K

    def loss(y_true, y_pred):
        return 0.5 * K.sum(K.square(y_true - y_pred))

    return loss

# -----

class LearningAgent(object):
    def __init__(self, action_space, batch_size=20, swap_freq=200):
        from keras.optimizers import RMSprop
        from keras.callbacks import TensorBoard
        import os, datetime
        # -----
        self.input_depth = 1
        self.past_range = 1
        # -----
        test_env = make_env()
        self.state_size = np.asarray(test_env.state[0]).shape
        self.observation_size = np.asarray(test_env.state[1]).shape
        # -----
        self.observation_shape = (self.input_depth * self.past_range,) + self.observation_size
        self.state_shape = (self.input_depth * self.past_range,) + self.state_size

        
        self.batch_size = batch_size
        _, _, self.train_net, advantage = build_network(self.state_size, self.observation_size, action_space.n)

        self.train_net.compile(optimizer=RMSprop(epsilon=0.1, rho=0.99),
                               loss=[value_loss(), policy_loss(advantage, args.beta)])
        
        self.pol_loss = deque(maxlen=25)
        self.val_loss = deque(maxlen=25)
        self.values = deque(maxlen=25)
        self.entropy = deque(maxlen=25)
        self.advantage = deque(maxlen=25)
        self.swap_freq = swap_freq
        self.swap_counter = self.swap_freq
        self.unroll = np.arange(self.batch_size)
        self.targets = np.zeros((self.batch_size, action_space.n))
        self.counter = 0

    def learn(self, last_states, last_observations, actions, rewards, var_dict, learning_rate=0.001):
        import keras.backend as K
        K.set_value(self.train_net.optimizer.lr, learning_rate)
        frames = len(last_observations)
        self.counter += frames
        var_dict['global_counter'] += frames
        # -----
        try:
            last_observations_squeezed = np.squeeze(last_observations)
            last_states_squeezed = np.squeeze(last_states)
            values, policy = self.train_net.predict([last_states_squeezed, last_observations_squeezed, self.unroll])
        except:
            print("\n Error in predicting the values and policy (learn):", sys.exc_info())
        # -----
        try:
            self.targets.fill(0.)
            advantage = rewards - values.flatten()
            mean_advantage = np.mean(advantage)
        except:
            print("\n Error in calculating the advantage (learn):", sys.exc_info())
        self.targets[self.unroll, actions] = 1.
        # -----
        loss = self.train_net.train_on_batch([last_states_squeezed, last_observations_squeezed, advantage], 
                                             [rewards, self.targets])
        entropy = np.mean(-policy * np.log(policy + 0.00000001))
        self.pol_loss.append(loss[2])
        self.val_loss.append(loss[1])
        self.entropy.append(entropy)
        self.advantage.append(mean_advantage)
        self.values.append(np.mean(values))
        min_val, max_val, avg_val = min(self.values), max(self.values), np.mean(self.values)
        write_log(self.train_net.tbCallback,["pol_loss","val_loss","entropy","advantage","val_min","val_max","val_avg"],
                  [loss[1],loss[2],entropy,mean_advantage,min_val,max_val,avg_val], var_dict['global_counter'])
        if args.verbose_learner == 1:
            print('\rFrames: %8d; Policy-Loss: %10.6f; Avg: %10.6f '
              '--- Value-Loss: %10.6f; Avg: %10.6f '
              '--- Entropy: %7.6f; Avg: %7.6f '
              '--- Advantage: %7.6f; Avg: %7.6f '
              '--- V-value; Min: %6.3f; Max: %6.3f; Avg: %6.3f' % (
                  self.counter,
                  loss[2], np.mean(self.pol_loss),
                  loss[1], np.mean(self.val_loss),
                  mean_advantage, np.mean(self.advantage),
                  entropy, np.mean(self.entropy),
                  min_val, max_val, avg_val), end='')
        # -----
        self.swap_counter -= frames
        if self.swap_counter < 0:
            self.swap_counter += self.swap_freq
            return True
        return False


def learn_proc(mem_queue, weight_dict, var_dict):
    import os
    from keras.callbacks import TensorBoard
    pid = os.getpid()
    # -----
    if args.verbose_learner == 1:
        print('\n %5d> Learning process' % (pid,))
    # -----
    save_freq = args.save_freq
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    checkpoint = args.checkpoint
    steps = args.steps
    # -----
    try:
        env = make_env()
    except:
        print("\n Error in creating the environment (learn_proc):", sys.exc_info())
    # -----
    try:
        agent = LearningAgent(env.action_space, batch_size=args.batch_size, swap_freq=args.swap_freq)
    except:
        print("\n Error in defining the LearningAgent (learn_proc):", sys.exc_info())
    # -----   
    try:
        agent.train_net.tbCallback = TensorBoard(log_dir=var_dict['datedir'], histogram_freq=32, batch_size=20, write_graph=True,
                                write_grads=False,write_images=False, embeddings_freq=0, embeddings_layer_names=None,
                                embeddings_metadata=None)
        agent.train_net.tbCallback.set_model(agent.train_net)
    except:
        print("\n Error in setting the TensorBoard callback (learn_proc):", sys.exc_info())
    
    # -----
    if checkpoint > 0:
        if args.verbose_learner == 1:
            print('\n %5d> Loading weights from file' % (pid,))
        agent.train_net.load_weights('weights/model-%s-%d.h5' % (args.game, checkpoint,))
        # -----
    if args.verbose_learner == 1:
        print('\n %5d> Setting weights in dict' % (pid,))
    weight_dict['update'] = 0
    weight_dict['weights'] = agent.train_net.get_weights()
    # -----
    last_sts = np.zeros((batch_size,) + agent.state_shape)
    last_obs = np.zeros((batch_size,) + agent.observation_shape)
    actions = np.zeros(batch_size, dtype=np.int32)
    rewards = np.zeros(batch_size)
    worker_ids = np.zeros(batch_size)
    # -----
    idx = 0
    agent.counter = checkpoint
    save_counter = checkpoint % save_freq + save_freq
    while True:
        # -----
        try:
            last_sts[idx, ...], last_obs[idx, ...], actions[idx], rewards[idx], worker_ids[idx] = mem_queue.get()
        except:
            print("\n Error in reading the memory queue (learn_proc):", sys.exc_info())
        # -----
        
        idx = (idx + 1) % batch_size
        if idx == 0:
            lr = max(0.00000001, (steps - agent.counter) / steps * learning_rate)
            write_log(agent.train_net.tbCallback,["learning_rate"],[lr], var_dict['global_counter'])
            write_log(agent.train_net.tbCallback,["w%d:actions" % (worker_ids[idx],), "w%d:rewards" % (worker_ids[idx],)],
                      [actions[idx], rewards[idx]], var_dict['global_counter'])
            # -----
            try:
                updated = agent.learn(last_sts, last_obs, actions, rewards, var_dict, learning_rate=lr)
            except:
                print("\n Error in learning the model (learn_proc):", sys.exc_info())
            # -----
            if updated:
                if args.verbose_learner == 1:
                    print('\n %5d> Updating weights in dict' % (pid,))
                weight_dict['weights'] = agent.train_net.get_weights()
                weight_dict['update'] += 1
        # -----
        save_counter -= 1
        if save_counter < 0:
            save_counter += save_freq
            if args.verbose_learner == 1:
                print('\n %5d> Save weights' % (pid,))
            agent.train_net.save_weights('weights/model-%s-%d.h5' % (args.game, agent.counter,), overwrite=True)


class ActingAgent(object):
    def __init__(self, action_space, n_step=8, discount=0.99):
        from keras.callbacks import TensorBoard
        self.input_depth = 1
        self.past_range = 1 # CHANGE THIS?
        # -----
        test_env = make_env()
        self.state_size = np.asarray(test_env.state[0]).shape
        self.observation_size = np.asarray(test_env.state[1]).shape
        # -----
        self.observation_shape = (self.input_depth * self.past_range,) + self.observation_size
        self.state_shape = (self.input_depth * self.past_range,) + self.state_size
        
        self.value_net, self.policy_net, self.load_net, adv = build_network(self.state_size, 
                                                                            self.observation_size, action_space.n)
        
        self.value_net.compile(optimizer='rmsprop', loss='mse')
        self.policy_net.compile(optimizer='rmsprop', loss='categorical_crossentropy')
        self.load_net.compile(optimizer='rmsprop', loss='mse', loss_weights=[0.5, 1.]) # dummy loss

        self.action_space = action_space
        self.observations = np.zeros(self.observation_shape)
        self.states = np.zeros(self.state_shape)
        self.last_observations = np.zeros_like(self.observations)
        self.last_states = np.zeros_like(self.states)
        # -----
        self.n_step_states = deque(maxlen=n_step)
        self.n_step_observations = deque(maxlen=n_step)
        self.n_step_actions = deque(maxlen=n_step)
        self.n_step_rewards = deque(maxlen=n_step)
        self.n_step = n_step
        self.discount = discount
        self.counter = 0

    def init_episode(self, state, observation):
        for _ in range(self.past_range):
            self.save_observation(observation)
            self.save_state(state)

    def reset(self):
        self.counter = 0
        self.n_step_observations.clear()
        self.n_step_states.clear()
        self.n_step_actions.clear()
        self.n_step_rewards.clear()

    def sars_data(self, action, reward, state, observation, terminal, worker_id, mem_queue):
        self.save_observation(observation)
        self.save_state(state)
        reward = np.clip(reward, -1., 1.)
        # reward /= args.reward_scale
        # -----
        self.n_step_observations.appendleft(self.last_observations)
        self.n_step_states.appendleft(self.last_states)
        self.n_step_actions.appendleft(action)
        self.n_step_rewards.appendleft(reward)
        # -----
        self.counter += 1
        if terminal or self.counter >= self.n_step:
            r = 0.
            if not terminal:
                r = self.value_net.predict([self.states, self.observations])[0]
            for i in range(self.counter):
                r = self.n_step_rewards[i] + self.discount * r
                mem_queue.put((self.n_step_states[i], self.n_step_observations[i], self.n_step_actions[i], r, worker_id))
            self.reset()

    def choose_action(self):
        try:
            policy = self.policy_net.predict([self.states, self.observations])[0]
        except:
            print("\n Error in predicting the policy (choose_action):", sys.exc_info())
        return np.random.choice(np.arange(self.action_space.n), p=policy)

    def save_observation(self, observation):
        self.last_observations = self.observations[...]
        self.observations = np.roll(self.observations, -self.input_depth, axis=0)
        # -----
        try:
            self.observations[-self.input_depth:, ...] = observation
        except:
            print("\n Error in saving the observation (save_observation):", sys.exc_info())

    def save_state(self, state):
        self.last_states = self.states[...]
        self.states = np.roll(self.states, -self.input_depth, axis=0)
        # -----
        try:
            self.states[-self.input_depth:, ...] = state
        except:
            print("\n Error in saving the state (save_state):", sys.exc_info())
    
def generate_experience_proc(mem_queue, weight_dict, var_dict, no):
    import os
    pid = os.getpid()
    worker_id = no+1
    # -----
    if args.verbose_worker == 1:
        print('\n %5d> worker%d process started' % (pid,worker_id,))
    # -----
    frames = 0
    batch_size = args.batch_size
    # -----
    try:
        env = make_env()
    except:
        print("\n Error in creating the environment (generate_experience_proc):", sys.exc_info())
    # -----
    try:
        agent = ActingAgent(env.action_space, n_step=args.n_step)
    except:
        print("\n Error in creating the environment (generate_experience_proc):", sys.exc_info())   
    # -----
    if frames > 0:
        if args.verbose_worker == 1:
            print('\n %5d> worker%d loaded weights from file' % (pid,worker_id,))
        agent.load_net.load_weights('weights/model-%s-%d.h5' % (args.game, frames))
    else:
        import time
        while 'weights' not in weight_dict:
            time.sleep(0.1)
        agent.load_net.set_weights(weight_dict['weights'])
        if args.verbose_worker == 1:
            print('\n %5d> worker%d loaded weights from dict' % (pid,worker_id,))
    # -----
    best_score = 0
    avg_score = deque([0], maxlen=25)
    last_update = 0
    while True:
        done = False
        episode_reward = 0
        op_last, op_count = 0, 0
        # -----
        state, obs = env.reset()
        observation = obs
        # -----
        try:
            agent.init_episode(state, observation)
        except:
            print("\n Error in initiating the episode (generate_experience_proc):", sys.exc_info())
        # -----
        while not done:
            frames += 1
            # -----
            try:
                action = agent.choose_action()
            except:
                print("\n Error in choosing the action (generate_experience_proc):", sys.exc_info())
            # -----
            try:
                state, obs, reward, done, _ = env.step(action)
                observation = obs
            except:
                print("\n Error in processing the step (generate_experience_proc):", sys.exc_info())
            episode_reward += reward
            best_score = max(best_score, episode_reward)
            # -----
            try:
                agent.sars_data(action, reward, state, observation, done, worker_id, mem_queue)
            except:
                print("\n Error in pushing to the memory queue (generate_experience_proc):", sys.exc_info())
            # -----
            op_count = 0 if op_last != action else op_count + 1
            done = done or op_count >= 100
            op_last = action
            # -----
            if frames % 10 == 0:
                if args.verbose_worker == 1:
                    print('\n %5d> worker%d - Best: %4d; Avg: %6.2f; Max: %4d' % (
                        pid, worker_id, best_score, np.mean(avg_score), np.max(avg_score)))
            if frames % batch_size == 0:
                update = weight_dict.get('update', 0)
                if update > last_update:
                    last_update = update
                    if args.verbose_worker == 1:
                        print('\n %5d> worker%d getting weights from dict' % (pid,worker_id,))
                    agent.load_net.set_weights(weight_dict['weights'])
        # -----
        avg_score.append(episode_reward)


def init_worker():
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def main():
    import os, datetime
    
    manager = Manager()
    weight_dict = manager.dict()
    var_dict = manager.dict()
    mem_queue = manager.Queue(args.queue_size)
    pool = Pool(args.processes + 1, init_worker)

    var_dict['global_counter'] = 0
    var_dict['datedir'] = os.path.join(os.getcwd()+'/Graph', datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    
    try:
        pool.apply_async(learn_proc, (mem_queue, weight_dict, var_dict))
        for i in range(args.processes):
            pool.apply_async(generate_experience_proc, (mem_queue, weight_dict, var_dict, i))

        pool.close()
        pool.join()

    except KeyboardInterrupt:
        pool.terminate()
        pool.join()


if __name__ == "__main__":
    main()
