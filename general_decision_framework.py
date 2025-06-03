import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics, regularizers
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib.pyplot as plt
from Observation import observation
from Markov_state_transition import Markov_state_transition_matrix, state_evolution

def initial_state_function(component_type):
    """
    :param component_type: 0 concrete_bridge, 1 steel_bridge, 2 deck, 3 superstructure, 4 substructure, 5...
    :return:
    """
    state_start = np.array([1, 0, 0, 0, 0])
    if component_type == 0:
        # expected life span: 40~80 years 60 years [12, 15, 20, 13]
        protection_time = 15
        concrete_bridge_lower = np.array([8, 10, 13.33, 8.67])
        concrete_bridge_upper = np.array([16, 20, 26.67, 17.33])
        mean = (concrete_bridge_lower + concrete_bridge_upper) / 2  # 均值位于区间中点
        std = (concrete_bridge_upper - concrete_bridge_lower) / 6  # 标准差 = 区间宽度/6
        state_matrix = np.random.normal(loc=mean, scale=std)

    elif component_type == 1:
        # expected life span: 30~70 years 50 years [10, 13, 16, 11]
        protection_time = 10
        steel_bridge_lower = np.array([6, 7.8, 9.6, 6.6])
        steel_bridge_upper = np.array([14, 18.2, 22.4, 15.4])
        mean = (steel_bridge_lower + steel_bridge_upper) / 2
        std = (steel_bridge_upper - steel_bridge_lower) / 6
        state_matrix = np.random.normal(loc=mean, scale=std)

    elif component_type == 2:
        # expected life span: 10~30 years 20 years [4, 6, 6, 4]
        protection_time = 4
        deck_lower = np.array([2, 3, 3, 2])
        deck_upper = np.array([6, 9, 9, 6])
        mean = (deck_lower + deck_upper) / 2
        std = (deck_upper - deck_lower) / 6
        state_matrix = np.random.normal(loc=mean, scale=std)

    elif component_type == 3:
        # expected life span: 30~50 years 40 years [8, 10, 12, 10]
        protection_time = 12
        superstructure_lower = np.array([6, 7.5, 9, 7.5])
        superstructure_upper = np.array([10, 12.5, 15, 12.5])
        mean = (superstructure_lower + superstructure_upper) / 2
        std = (superstructure_upper - superstructure_lower) / 6
        state_matrix = np.random.normal(loc=mean, scale=std)

    elif component_type == 4:
        # expected life span: 40~60 years 50 years [10, 14, 16, 10]
        protection_time = 15
        substructure_lower = np.array([8, 11.2, 12.8, 8])
        substructure_upper = np.array([12, 16.8, 19.2, 12])
        mean = (substructure_lower + substructure_upper) / 2
        std = (substructure_upper - substructure_lower) / 6
        state_matrix = np.random.normal(loc=mean, scale=std)

    elif component_type == 5:
        # expected life span: 40~60 years 50 years [8.67, 17.33, 15, 9]
        protection_time = 16
        girder_lower = np.array([6.93, 13.87, 12, 7.2])
        girder_upper = np.array([10.4, 20.8, 18, 10.8])
        mean = (girder_lower + girder_upper) / 2
        std = (girder_upper - girder_lower) / 6
        state_matrix = np.random.normal(loc=mean, scale=std)

    elif component_type == 6:
        # expected life span: 20~40 years 30 years [5.2, 10.4, 9, 5.4]
        protection_time = 8
        slab_lower = np.array([3.47, 6.93, 6, 3.6])
        slab_upper = np.array([6.93, 13.87, 12, 7.2])
        mean = (slab_lower + slab_upper) / 2
        std = (slab_upper - slab_lower) / 6
        state_matrix = np.random.normal(loc=mean, scale=std)

    elif component_type == 7:
        # expected life span: 35~45 years 40 years [6, 12, 14, 8]
        protection_time = 9
        diaphragm_lower = np.array([5.25, 10.5, 12.25, 7])
        diaphragm_upper = np.array([6.75, 13.5, 15.75, 9])
        mean = (diaphragm_lower + diaphragm_upper) / 2
        std = (diaphragm_upper - diaphragm_lower) / 6
        state_matrix = np.random.normal(loc=mean, scale=std)

    elif component_type == 8:
        # 50-70 years, expected 60 years [5.67, 18, 26.33, 10]
        protection_time = 10
        arch_lower = np.array([4.72, 15, 21.94, 8.33])
        arch_upper = np.array([16.61, 21, 30.72, 11.67])
        mean = (arch_lower + arch_upper) / 2
        std = (arch_upper - arch_lower) / 6
        state_matrix = np.random.normal(loc=mean, scale=std)

    elif component_type == 9:
        # 20-60 years, expected 40 years [3.78, 12, 17.55, 6.67]
        protection_time = 10
        transverse_lower = np.array([1.89, 6, 8.78, 3.33])
        transverse_upper = np.array([5.67, 18, 26.33, 10])
        mean = (transverse_lower + transverse_upper) / 2
        std = (transverse_upper - transverse_lower) / 6
        state_matrix = np.random.normal(loc=mean, scale=std)

    elif component_type == 10:
        # expected life span: 10~20 years 15 years [3, 4, 5, 3]
        protection_time = 4
        hanger_lower = np.array([2, 2.67, 3.33, 2])
        hanger_upper = np.array([4, 5.33, 6.67, 4])
        mean = (hanger_lower + hanger_upper) / 2
        std = (hanger_upper - hanger_lower) / 6
        state_matrix = np.random.normal(loc=mean, scale=std)

    elif component_type == 11:
        # expected life span: 40~60 years 50 years [12, 13, 13, 12]
        protection_time = 16
        spandrel_lower = np.array([9.6, 10.4, 10.4, 9.6])
        spandrel_upper = np.array([14.4, 15.6, 15.6, 14.4])
        mean = (spandrel_lower + spandrel_upper) / 2
        std = (spandrel_upper - spandrel_lower) / 6
        state_matrix = np.random.normal(loc=mean, scale=std)

    elif component_type == 12:
        # expected life span: 10~40 years 25 years [6.25, 7.5, 6.25, 5]
        protection_time = 3
        support_lower = np.array([2.5, 3, 2.5, 2])
        support_upper = np.array([10, 12, 10, 8])
        mean = (support_lower + support_upper) / 2
        std = (support_upper - support_lower) / 6
        state_matrix = np.random.normal(loc=mean, scale=std)

    elif component_type == 13:
        # expected life span: 60~80 years 70 years [15 15 20 20]
        protection_time = 20
        tower_lower = np.array([12.86, 12.86, 17.14, 17.14])
        tower_upper = np.array([17.14, 17.14, 22.86, 22.86])
        mean = (tower_lower + tower_upper) / 2
        std = (tower_upper - tower_lower) / 6
        state_matrix = np.random.normal(loc=mean, scale=std)

    elif component_type == 14:
        # expected life span: 7~17 years 12 years [2, 3, 4, 3]
        protection_time = 4
        cable_lower = np.array([1.17, 1.75, 2.33, 1.75])
        cable_upper = np.array([2.83, 4.25, 5.67, 4.25])
        mean = (cable_lower + cable_upper) / 2
        std = (cable_upper - cable_lower) / 6
        state_matrix = np.random.normal(loc=mean, scale=std)

    elif component_type == 15:
        # expected life span: 40~80 years, 60 years [10, 16, 20, 14]
        protection_time = 7
        steel_girder_lower = np.array([6.67, 10.67, 13.33, 9.33])
        steel_girder_upper = np.array([13.33, 21.33, 26.67, 18.67])
        mean = (steel_girder_lower + steel_girder_upper) / 2
        std = (steel_girder_upper - steel_girder_lower) / 6
        state_matrix = np.random.normal(loc=mean, scale=std)

    elif component_type == 16:
        # expected life span: 5~15 years 10 years [3, 2, 3, 2]
        protection_time = 4
        pavement_lower = np.array([1.5, 1, 1.5, 1])
        pavement_upper = np.array([4.5, 3, 4.5, 3])
        mean = (pavement_lower + pavement_upper) / 2
        std = (pavement_upper - pavement_lower) / 6
        state_matrix = np.random.normal(loc=mean, scale=std)

    elif component_type == 17:
        # expected life span: 40~60 years 50 years [5.125, 12.5, 16.5, 15.875]
        protection_time = 15
        column_lower = np.array([4.1, 10, 13.2, 12.7])
        column_upper = np.array([6.15, 15, 19.8, 19.05])
        mean = (column_lower + column_upper) / 2
        std = (column_upper - column_lower) / 6
        state_matrix = np.random.normal(loc=mean, scale=std)

    elif component_type == 18:
        # expected life span: 25~50 years 40 years [8 10 12 10]
        protection_time = 12
        abutment_lower = np.array([5, 6.25, 7.5, 6.25])
        abutment_upper = np.array([11, 13.75, 16.5, 13.75])
        mean = (abutment_lower + abutment_upper) / 2
        std = (abutment_upper - abutment_lower) / 6
        state_matrix = np.random.normal(loc=mean, scale=std)

    elif component_type == 19:
        # expected life span: 40~80 years 60 years [10, 15, 20, 15]
        protection_time = 17
        foundation_lower = np.array([6.67, 10, 13.33, 10])
        foundation_upper = np.array([13.33, 20, 26.67, 20])
        mean = (foundation_lower + foundation_upper) / 2
        std = (foundation_upper - foundation_lower) / 6
        state_matrix = np.random.normal(loc=mean, scale=std)

    else:
        # expected life span: 20~30 years 25 years [5, 7, 9, 4]
        protection_time = 6
        lower = np.array([4, 5.6, 7.2, 3.2])
        upper = np.array([6, 8.4, 10.8, 4.8])
        mean = (lower + upper) / 2
        std = (upper - lower) / 6
        state_matrix = np.random.normal(loc=mean, scale=std)

    time_normalized = protection_time / 20
    state_matrix /= 30
    component_ID = tf.one_hot(component_type, 21).numpy()
    initial_time = tf.one_hot(0, 100).numpy()

    initial_states = np.concatenate((state_start, [time_normalized], state_matrix, component_ID, initial_time))
    initial_states = initial_states.reshape(1, len(initial_states))
    return initial_states


class environment():
    """this part define the bridge degradation process"""
    def __init__(self):
        self.state_number = 5
        self.normalized_time = 15

        self.deterioration_rate = np.array(
            [2, 3, 1.8, 2.3, 2.4, 2, 1.9, 2.3, 3, 2.8, 3.4, 1.9, 1.2, 1.5, 3.6, 3, 1.5, 1.9, 2.1, 2.0])
        self.protection = np.array([15, 10, 4, 12, 15, 16, 8, 9, 10, 10, 4, 16, 3, 20, 4, 7, 4, 15, 12, 17])

        # observation matrix
        self.accuracy_visual = 0.6
        self.accuracy_NDT = 0.99
        self.observation_visual = observation(self.accuracy_visual, self.state_number, Matrix_type=False)
        self.observation_NDT = observation(self.accuracy_NDT, self.state_number, Matrix_type=True)

        # defined repair or replace action state transition matrix
        self.repair_matrix = np.zeros((self.state_number, self.state_number))
        self.repair_matrix[:, 0] = 1

        # define the cost & risk value in different component,
        self.risk = np.array([0, 0, 0, -0.6, -1.5])
        self.cost = np.array([-0.01, -0.1, -0.15, -1])


    def step(self, states, actions, hidden_state):
        """
        :param states: last item is time with one-hot
        :param actions: 0: visual inspection, 1: NDT, 2: preventive maintenance+visual inspection,
        3: preventive maintenance+NDT, 4: replacement
        :param hidden_state: 0-4, [3 + 20 + 4 + 44 + 63 + 6 + 6 + 6]
        :return: new_state, reward, new_hidden_state
        """
        # obtain the state information from inputting vector
        state = states.flatten()
        component_ID = tf.argmax(state[10:31]).numpy()

        component_state = state[0:5]
        protection_time = state[5] * 20  # normalization
        duration_time = state[6:10] * 30  # normalization


        time = tf.argmax(state[31:131]).numpy()

        protection_time += -1
        time += 1

        """
        Repair or replace component based on the action-----------------------------------------------------------------
        """
        cost_repair = 0
        if actions == 4:
            component_state = component_state @ self.repair_matrix
            hidden_state = 0
            protection_time = self.protection[component_ID]
            cost_repair += self.cost[3]

        """
        based protection time and duration time calculate state-transition matrix---------------------------------------
        """

        state_T_D = Markov_state_transition_matrix(self.state_number,
                                                               duration_time / self.deterioration_rate[component_ID])
        state_T = Markov_state_transition_matrix(self.state_number, duration_time)

        component_new_state, new_protection_time, new_hidden_state = state_evolution(component_state, protection_time,
                                                                                     hidden_state, state_T, state_T_D,
                                                                                     self.normalized_time)
        """
        inspection part-------------------------------------------------------------------------------------------------
        """
        cost_inspection = 0
        if actions == 0 or actions == 2:
            cost_inspection += self.cost[0]
            obser_mark = 0.
            random_number = np.random.uniform(0, 1)
            for j in range(self.state_number):
                obser_mark = obser_mark + self.observation_visual[new_hidden_state, j]
                if random_number <= obser_mark:
                    observation_value = j
                    break
            component_new_state[0: 5] = component_new_state[0: 5] * self.observation_visual[:,observation_value]
            component_new_state[0: 5] = component_new_state[0: 5] / np.sum(component_new_state[0: 5])

        if actions == 1 or actions == 3:
            cost_inspection += self.cost[1]
            obser_mark = 0.
            random_number = np.random.uniform(0, 1)
            for j in range(self.state_number):
                obser_mark = obser_mark + self.observation_NDT[new_hidden_state, j]
                if random_number <= obser_mark:
                    observation_value = j
                    break
            component_new_state[0: 5] = component_new_state[0: 5] * self.observation_NDT[:, observation_value]
            component_new_state[0: 5] = component_new_state[0: 5] / np.sum(component_new_state[0: 5])

        """
        preventative maintenance action---------------------------------------------------------------------------------
        """
        cost_prevention = 0
        if actions == 2 or actions == 3:
            protection_time = self.protection[component_ID]
            cost_prevention += self.cost[2]

        """
        calculate the cost based on the cost & risk table---------------------------------------------------------------
        """
        risk = self.risk @ component_state
        reward = cost_inspection + cost_prevention + cost_repair + risk

        """
        calculate new state---------------------------------------------------------------------------------------------
        """
        ID = tf.one_hot(component_ID, 21).numpy()
        Time = tf.one_hot(time, 100).numpy()

        protection_time /= 20
        duration_time /= 30
        new_state = np.concatenate((component_new_state, [protection_time], duration_time, ID, Time))
        new_state = new_state.reshape(1, len(new_state))

        """
        if time > 100, the agent finish the assignment of management a type of component, go next-----------------------
        """
        if time > 99:
            #  initail the parameter with a new start
            component_ID += 1
            if component_ID <= 20:
                new_start_state = initial_state_function(component_ID)
                new_hidden_state = 0
            else:
                raise ValueError("iteration is not reasonable")
        else:
            new_start_state = new_state.copy()

        return new_state, reward, new_hidden_state, new_start_state

class ActorWithAttention(tf.keras.Model):
    def __init__(self, state_size, action_size, batch_norm=True,
                 hidden=[256, 256, 256], num_heads=4, key_dim=4, num_groups=16):
        super(ActorWithAttention, self).__init__()

        # 基础网络层
        self.fc1 = layers.Dense(hidden[0], input_shape=(None, state_size),
                                kernel_regularizer=regularizers.l2(0.001))
        self.fc2 = layers.Dense(hidden[1], kernel_regularizer=regularizers.l2(0.001))
        self.fc3 = layers.Dense(hidden[2], kernel_regularizer=regularizers.l2(0.001))
        self.fc4 = layers.Dense(action_size, kernel_regularizer=regularizers.l2(0.001))

        # 注意力机制
        self.state_attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            value_dim=hidden[0] // num_groups  # 新增value_dim对齐
        )

        # 分组参数验证
        assert hidden[0] % num_groups == 0, "hidden[0]必须能被num_groups整除"
        self.num_groups = num_groups
        self.group_dim = hidden[0] // num_groups
        self.hidden = hidden

        # 其他配置保持不变
        self.bn_layers = [layers.BatchNormalization() for _ in range(3)]
        self.batch_norm = batch_norm

    def call(self, inputs):
        x = inputs

        # 第一全连接层
        x = self.fc1(x)
        if self.batch_norm:
            x = self.bn_layers[0](x)
        x = tf.nn.relu(x)

        # === 状态注意力模块 ===
        # 分组处理
        batch_size = tf.shape(x)[0]
        seq_x = tf.reshape(x, (batch_size, self.num_groups, self.group_dim))

        # 注意力计算
        attn_output = self.state_attention(seq_x, seq_x)

        # 维度恢复
        x = tf.reshape(attn_output, (batch_size, self.hidden[0]))

        # 残差块（保持不变）
        res1 = x
        x = self.fc2(x)
        if self.batch_norm:
            x = self.bn_layers[1](x)
        x = tf.nn.relu(x)
        x += res1

        res2 = x
        x = self.fc3(x)
        if self.batch_norm:
            x = self.bn_layers[2](x)
        x = tf.nn.relu(x)
        x += res2
        return self.fc4(x)

class CriticWithAttention(tf.keras.Model):
    def __init__(self, state_size, batch_norm=True,
                 hidden=[256, 256, 256], num_heads=4, key_dim=4, num_groups=16):
        super(CriticWithAttention, self).__init__()

        # 基础网络层
        self.fc1 = layers.Dense(hidden[0], input_shape=(None, state_size),
                                kernel_regularizer=regularizers.l2(0.001))
        self.fc2 = layers.Dense(hidden[1], kernel_regularizer=regularizers.l2(0.001))
        self.fc3 = layers.Dense(hidden[2], kernel_regularizer=regularizers.l2(0.001))
        self.fc4 = layers.Dense(1, kernel_regularizer=regularizers.l2(0.001))

        # 注意力机制
        self.state_attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            value_dim=hidden[0] // num_groups  # 新增value_dim对齐
        )

        # 分组参数验证
        assert hidden[0] % num_groups == 0, "hidden[0]必须能被num_groups整除"
        self.num_groups = num_groups
        self.group_dim = hidden[0] // num_groups
        self.hidden = hidden

        # 其他配置保持不变
        self.bn_layers = [layers.BatchNormalization() for _ in range(3)]
        self.batch_norm = batch_norm

    def call(self, inputs):
        x = inputs

        # 第一全连接层
        x = self.fc1(x)
        if self.batch_norm:
            x = self.bn_layers[0](x)
        x = tf.nn.relu(x)

        # === 状态注意力模块 ===
        # 分组处理
        batch_size = tf.shape(x)[0]
        seq_x = tf.reshape(x, (batch_size, self.num_groups, self.group_dim))

        # 注意力计算
        attn_output = self.state_attention(seq_x, seq_x)

        # 维度恢复
        x = tf.reshape(attn_output, (batch_size, self.hidden[0]))

        # 残差块（保持不变）
        res1 = x
        x = self.fc2(x)
        if self.batch_norm:
            x = self.bn_layers[1](x)
        x = tf.nn.relu(x)
        x += res1

        res2 = x
        x = self.fc3(x)
        if self.batch_norm:
            x = self.bn_layers[2](x)
        x = tf.nn.relu(x)
        x += res2

        return self.fc4(x)


def get_action(Actor_network, state, greedy=False):
    logit = Actor_network(state)
    prob = tf.nn.softmax(logit).numpy()
    if greedy:
        return np.argmax(prob.ravel())
    action = np.random.choice(logit.shape[1], p=prob.ravel())
    return action

class Agent():
    def __init__(self,
                 Actor,
                 Critic,
                 Environment_set,
                 n_actions=5,
                 input_shape=131,
                 gamma=0.99,
                 ):
        # state vector and action number
        self.n_actions = n_actions
        self.input_shape = input_shape
        self.gamma = gamma

        # define the Actor-Critic networks
        self.Actor = Actor
        self.Critic = Critic
        self.Environment = Environment_set

        self.Critic_optimizer = tf.optimizers.Adam(1e-4)
        self.Actor_optimizer = tf.optimizers.Adam(1e-5)

    def save(self, folder_name):
        # Check if folder_name is a directory
        if not os.path.isdir(folder_name):
            # If it's not a directory, create it
            os.makedirs(folder_name)

        self.Actor.save_weights(folder_name + '/Actor.weights.h5')
        self.Critic.save_weights(folder_name + '/Critic.weights.h5')
        print(f"Actor and Critic weights saved in folder: {folder_name}")

    def load(self, folder_name):
        dump_input = tf.random.normal((1, 131))
        self.Actor(dump_input)
        self.Critic(dump_input)
        self.Actor.load_weights(folder_name + '/Actor.weights.h5')
        self.Critic.load_weights(folder_name + '/Critic.weights.h5')

    def Critic_learn(self, state_old, reward, state_new, gamma):
        reward = tf.reshape(tf.constant(reward, dtype=float), [-1, 1])
        with tf.GradientTape() as tape:
            v_old = self.Critic(state_old)
            v_new = self.Critic(state_new)
            TD_error = reward + gamma * v_new - v_old
            loss_critic = tf.square(TD_error)
        model_gradients = tape.gradient(loss_critic, self.Critic.trainable_variables)
        self.Critic_optimizer.apply_gradients(zip(model_gradients, self.Critic.trainable_variables))
        return TD_error

    def Actor_learn(self, state, action, TD_error):
        one_action = tf.cast(tf.one_hot(action, 5), dtype=tf.float32)
        with tf.GradientTape() as tape:
            logit = self.Actor(state)
            cross_entropy = - tf.multiply(tf.math.log(tf.nn.softmax(logit) + 1e-20), one_action)
            loss_actor = tf.reduce_sum(tf.multiply(cross_entropy, TD_error))
            entropy = tf.reduce_sum(-tf.nn.softmax(logit) * tf.math.log(tf.nn.softmax(logit) + 1e-20))
            loss = loss_actor - 0.02 * entropy
        grads = tape.gradient(loss, self.Actor.trainable_variables)
        self.Actor_optimizer.apply_gradients(zip(grads, self.Actor.trainable_variables))
        print('loss_actor', loss_actor.numpy(), 'entropy', entropy.numpy())

    def estimation(self):
        # estimate whether performance become better
        # initial the parameters
        max_over_step = 2000
        Reward = []

        for li in range(1000):
            initial_states = initial_state_function(0)
            t = 0
            states = initial_states.copy()
            hidden_state = 0
            reward_sum = 0
            while t < max_over_step:
                action = get_action(self.Actor, states, greedy=True)

                new_state, reward, hidden_state, new_start_state = self.Environment.step(states, action, hidden_state)

                states = new_start_state.copy()
                reward_sum += reward
                t += 1

            Reward.append(reward_sum)
            print("\r", end="")
            print("进度: {}%: ".format(li / 10), "▓" * (li // 20), end="")

        Reward_sum = sum(Reward) / 1000
        print(Reward_sum)
        return Reward_sum

    def get_trajectory(self, initial_state, hidden_state, exploration_step):
        """
        :param Actor_network:
        :param Critic_network:
        :param initial_state:
        :param exploration_step:
        :return: trajectory[states, actions, rewards], terminal(boolean)
        """
        # memory
        memory_states = np.zeros((exploration_step, self.input_shape))
        memory_actions = np.zeros((exploration_step))
        memory_reward = np.zeros((exploration_step, 1))
        memory_new_states = np.zeros((exploration_step, self.input_shape))
        memory_hidden_state = np.zeros((exploration_step))

        # get MINI_STEP trajectory
        for i in range(exploration_step):
            memory_states[i, :] = initial_state.copy()
            #  action = 1
            action = get_action(self.Actor, initial_state, greedy=False)

            New_state, reward, new_hidden_state, new_start_state = self.Environment.step(initial_state, action, hidden_state)

            memory_actions[i] = action
            memory_reward[i, :] = reward
            memory_hidden_state[i] = hidden_state
            memory_new_states[i, :] = New_state.copy()

            initial_state = new_start_state.copy()
            hidden_state = new_hidden_state
            reward_sum = np.sum(memory_reward[:, 0])

        return (memory_states, memory_actions, memory_reward, memory_new_states, new_start_state, reward_sum,
                new_hidden_state, memory_hidden_state)

def main():
    Actor_network = ActorWithAttention(131, 5)
    Critic_network = CriticWithAttention(131)
    Environment = environment()
    initial_states = initial_state_function(0)

    num_training = 100001
    max_over_step = 2000
    gamma = 0.99
    exploration_step = 25
    agent = Agent(Actor_network, Critic_network, Environment)
    t_plot = []
    reward_plot = []
    action_episode = np.zeros((2000))
    hidden_state_episode = np.zeros((2000))
    state_episode = np.zeros((2000, 131))

    continue_training = False
    if continue_training:
        agent.load('ActorCritic')
        estimate_value = np.loadtxt('Performance')
    else:
        estimate_value = -1000

    for i in range(0, num_training):
        t = 0
        reward_sum = 0
        states = initial_states.copy()
        hidden_state = 0

        if i % 5000 == 0 and i > 1:
            New_reward_sum = agent.estimation()

            if New_reward_sum - estimate_value >= 0:
                agent.save('ActorCritic')
                li = np.array(New_reward_sum, dtype=np.float64).reshape(1, 1)
                np.savetxt('Performance', li)
                estimate_value = New_reward_sum

        while t < max_over_step:
            memory_states, memory_actions, memory_reward, memory_new_states, New_state, reward, hidden_state, memory_hidden_state = agent.get_trajectory(
                states, hidden_state, exploration_step)

            TD_error = agent.Critic_learn(memory_states, memory_reward[:, 0], memory_new_states, gamma)
            agent.Actor_learn(memory_states, memory_actions, TD_error)

            states = New_state.copy()
            reward_sum = reward_sum + reward
            action_episode[t:t + exploration_step] = memory_actions
            hidden_state_episode[t:t + exploration_step] = memory_hidden_state
            state_episode[t:t + exploration_step, :] = memory_states

            t += exploration_step
            if t == max_over_step:
                concrete_bridge_state = state_episode[0:100, 0:5] @ np.array([1, 2, 3, 4, 5])
                steel_bridge_state = state_episode[100:200, 0:5] @ np.array([1, 2, 3, 4, 5])
                deck_state = state_episode[200:300, 0:5] @ np.array([1, 2, 3, 4, 5])
                superstructure_state = state_episode[300:400, 0:5] @ np.array([1, 2, 3, 4, 5])
                substructure_state = state_episode[400:500, 0:5] @ np.array([1, 2, 3, 4, 5])
                girder_state = state_episode[500:600, 0:5] @ np.array([1, 2, 3, 4, 5])
                slab_state = state_episode[600:700, 0:5] @ np.array([1, 2, 3, 4, 5])
                diaphragm_state = state_episode[700:800, 0:5] @ np.array([1, 2, 3, 4, 5])
                arch_state = state_episode[800:900, 0:5] @ np.array([1, 2, 3, 4, 5])
                transverse_state = state_episode[900:1000, 0:5] @ np.array([1, 2, 3, 4, 5])
                hanger_state = state_episode[1000:1100, 0:5] @ np.array([1, 2, 3, 4, 5])
                support_state = state_episode[1100:1200, 0:5] @ np.array([1, 2, 3, 4, 5])
                spandrel_state = state_episode[1200:1300, 0:5] @ np.array([1, 2, 3, 4, 5])
                tower_state = state_episode[1300:1400, 0:5] @ np.array([1, 2, 3, 4, 5])
                cable_state = state_episode[1400:1500, 0:5] @ np.array([1, 2, 3, 4, 5])
                steel_girder_state = state_episode[1500:1600, 0:5] @ np.array([1, 2, 3, 4, 5])
                pavement_state = state_episode[1600:1700, 0:5] @ np.array([1, 2, 3, 4, 5])
                column_state = state_episode[1700:1800, 0:5] @ np.array([1, 2, 3, 4, 5])
                abutment_state = state_episode[1800:1900, 0:5] @ np.array([1, 2, 3, 4, 5])
                foundation_state = state_episode[1900:2000, 0:5] @ np.array([1, 2, 3, 4, 5])

                print("epoch num:", i, "Reward:", np.round(reward_sum, 3))
                print("-----------------------------------------------------------------------------------------")
                plt.ion()
                fig1 = plt.figure(1, figsize=(3.5, 2))
                fig1.canvas.manager.window.move(50, 50)
                plt.clf()
                t_plot.append(i)
                reward_plot.append(reward_sum)
                plt.plot(t_plot, reward_plot, label='Sum reward', color='blueviolet', alpha=1, linewidth=0.4)
                plt.legend(loc='best', fontsize=8)
                plt.xlabel("life-cycle(year)", fontsize=8)
                plt.ylabel("reward", fontsize=8)
                plt.ylim((-250, -100))
                plt.draw()
                plt.pause(0.001)

                fig2 = plt.figure(2, figsize=(3.5, 2))
                fig2.canvas.manager.window.move(400, 50)
                plt.clf()
                plt.plot(np.linspace(0, 100, 100), action_episode[0:100], c="green", alpha=0.5, label='Action')
                plt.plot(np.linspace(0, 100, 100), concrete_bridge_state, label='concrete_bridge state',
                         color='deepskyblue')
                plt.legend(loc='best', fontsize=8)
                plt.xlabel("life-cycle(year)", fontsize=8)
                plt.ylabel("damage percentage", fontsize=8)
                plt.draw()
                plt.pause(0.001)

                fig3 = plt.figure(3, figsize=(3.5, 2))
                fig3.canvas.manager.window.move(750, 50)
                plt.clf()
                plt.plot(np.linspace(0, 100, 100), action_episode[100:200], c="green", alpha=0.5, label='Action')
                plt.plot(np.linspace(0, 100, 100), steel_bridge_state, label='steel_bridge state',
                         color='deepskyblue')
                plt.legend(loc='best', fontsize=8)
                plt.xlabel("life-cycle(year)", fontsize=8)
                plt.ylabel("expected state", fontsize=8)
                plt.draw()
                plt.pause(0.001)

                # fig4 = plt.figure(4, figsize=(3.5, 2))
                # fig4.canvas.manager.window.move(1100, 50)
                # plt.clf()
                # plt.plot(np.linspace(0, 100, 100), action_episode[200:300], c="green", alpha=0.5, label='Action')
                # plt.plot(np.linspace(0, 100, 100), deck_state, label='deck state',
                #          color='deepskyblue')
                # plt.legend(loc='best', fontsize=8)
                # plt.xlabel("life-cycle(year)", fontsize=8)
                # plt.ylabel("expected state", fontsize=8)
                # plt.draw()
                # plt.pause(0.001)
                #
                # fig5 = plt.figure(5, figsize=(3.5, 2))
                # fig5.canvas.manager.window.move(1450, 50)
                # plt.clf()
                # plt.plot(np.linspace(0, 100, 100), action_episode[300:400], c="green", alpha=0.5, label='Action')
                # plt.plot(np.linspace(0, 100, 100), superstructure_state, label='superstructure state',
                #          color='deepskyblue')
                # plt.legend(loc='best', fontsize=8)
                # plt.xlabel("life-cycle(year)", fontsize=8)
                # plt.ylabel("expected state", fontsize=8)
                # plt.draw()
                # plt.pause(0.001)
                #
                # fig6 = plt.figure(6, figsize=(3.5, 2))
                # fig6.canvas.manager.window.move(50, 350)
                # plt.clf()
                # plt.plot(np.linspace(0, 100, 100), action_episode[400:500], c="green", alpha=0.5, label='Action')
                # plt.plot(np.linspace(0, 100, 100), substructure_state, label='substructure state',
                #          color='deepskyblue')
                # plt.legend(loc='best', fontsize=8)
                # plt.xlabel("life-cycle(year)", fontsize=8)
                # plt.ylabel("expected state", fontsize=8)
                # plt.draw()
                # plt.pause(0.001)
                #
                # fig7 = plt.figure(7, figsize=(3.5, 2))
                # fig7.canvas.manager.window.move(400, 350)
                # plt.clf()
                # plt.plot(np.linspace(0, 100, 100), action_episode[600:700], c="green", alpha=0.5, label='Action')
                # plt.plot(np.linspace(0, 100, 100), slab_state, label='slab state',
                #          color='deepskyblue')
                # plt.legend(loc='best', fontsize=8)
                # plt.xlabel("life-cycle(year)", fontsize=8)
                # plt.ylabel("expected state", fontsize=8)
                # plt.draw()
                # plt.pause(0.001)
                #
                # fig8 = plt.figure(8, figsize=(3.5, 2))
                # fig8.canvas.manager.window.move(750, 350)
                # plt.clf()
                # plt.plot(np.linspace(0, 100, 100), action_episode[700:800], c="green", alpha=0.5, label='Action')
                # plt.plot(np.linspace(0, 100, 100), diaphragm_state, label='diaphragm state',
                #          color='deepskyblue')
                # plt.legend(loc='best', fontsize=8)
                # plt.xlabel("life-cycle(year)", fontsize=8)
                # plt.ylabel("expected state", fontsize=8)
                # plt.draw()
                # plt.pause(0.001)
                #
                # fig9 = plt.figure(9, figsize=(3.5, 2))
                # fig9.canvas.manager.window.move(1100, 350)
                # plt.clf()
                # plt.plot(np.linspace(0, 100, 100), action_episode[800:900], c="green", alpha=0.5, label='Action')
                # plt.plot(np.linspace(0, 100, 100), arch_state, label='arch state',
                #          color='deepskyblue')
                # plt.legend(loc='best', fontsize=8)
                # plt.xlabel("life-cycle(year)", fontsize=8)
                # plt.ylabel("expected state", fontsize=8)
                # plt.draw()
                # plt.pause(0.001)
                #
                # fig10 = plt.figure(10, figsize=(3.5, 2))
                # fig10.canvas.manager.window.move(1450, 350)
                # plt.clf()
                # plt.plot(np.linspace(0, 100, 100), action_episode[1000:1100], c="green", alpha=0.5, label='Action')
                # plt.plot(np.linspace(0, 100, 100), hanger_state, label='hanger state',
                #          color='deepskyblue')
                # plt.legend(loc='best', fontsize=8)
                # plt.xlabel("life-cycle(year)", fontsize=8)
                # plt.ylabel("expected state", fontsize=8)
                # plt.draw()
                # plt.pause(0.001)
                #
                # fig11 = plt.figure(11, figsize=(3.5, 2))
                # fig11.canvas.manager.window.move(50, 650)
                # plt.clf()
                # plt.plot(np.linspace(0, 100, 100), action_episode[1100:1200], c="green", alpha=0.5, label='Action')
                # plt.plot(np.linspace(0, 100, 100), support_state, label='support state',
                #          color='deepskyblue')
                # plt.legend(loc='best', fontsize=8)
                # plt.xlabel("life-cycle(year)", fontsize=8)
                # plt.ylabel("expected state", fontsize=8)
                # plt.draw()
                # plt.pause(0.001)
                #
                # fig12 = plt.figure(12, figsize=(3.5, 2))
                # fig12.canvas.manager.window.move(400, 650)
                # plt.clf()
                # plt.plot(np.linspace(0, 100, 100), action_episode[1300:1400], c="green", alpha=0.5, label='Action')
                # plt.plot(np.linspace(0, 100, 100), tower_state, label='tower state',
                #          color='deepskyblue')
                # plt.legend(loc='best', fontsize=8)
                # plt.xlabel("life-cycle(year)", fontsize=8)
                # plt.ylabel("expected state", fontsize=8)
                # plt.draw()
                # plt.pause(0.001)
                #
                # fig13 = plt.figure(13, figsize=(3.5, 2))
                # fig13.canvas.manager.window.move(750, 650)
                # plt.clf()
                # plt.plot(np.linspace(0, 100, 100), action_episode[1400:1500], c="green", alpha=0.5, label='Action')
                # plt.plot(np.linspace(0, 100, 100), cable_state, label='cable state',
                #          color='deepskyblue')
                # plt.legend(loc='best', fontsize=8)
                # plt.xlabel("life-cycle(year)", fontsize=8)
                # plt.ylabel("expected state", fontsize=8)
                # plt.draw()
                # plt.pause(0.001)
                #
                # fig14 = plt.figure(14, figsize=(3.5, 2))
                # fig14.canvas.manager.window.move(1100, 650)
                # plt.clf()
                # plt.plot(np.linspace(0, 100, 100), action_episode[1600:1700], c="green", alpha=0.5, label='Action')
                # plt.plot(np.linspace(0, 100, 100), pavement_state, label='pavement state',
                #          color='deepskyblue')
                # plt.legend(loc='best', fontsize=8)
                # plt.xlabel("life-cycle(year)", fontsize=8)
                # plt.ylabel("expected state", fontsize=8)
                # plt.draw()
                # plt.pause(0.001)
                #
                # fig15 = plt.figure(15, figsize=(3.5, 2))
                # fig15.canvas.manager.window.move(1450, 650)
                # plt.clf()
                # plt.plot(np.linspace(0, 100, 100), action_episode[1900:2000], c="green", alpha=0.5, label='Action')
                # plt.plot(np.linspace(0, 100, 100), foundation_state, label='foundation state',
                #          color='deepskyblue')
                # plt.legend(loc='best', fontsize=8)
                # plt.xlabel("life-cycle(year)", fontsize=8)
                # plt.ylabel("expected state", fontsize=8)
                # plt.draw()
                # plt.pause(0.01)


if __name__ == '__main__':
    main()



