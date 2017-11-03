import tensorflow as tf
import numpy as np

class Mineral(object):
    def __init__(self, n_actions,n_features,
                 learning_rate=0.01,
                 reward_decay = 0.9,
                 e_greed = 0.9,
                 memory_size = 500,
                 batch_size = 32,
                 replace_target_iter = 300,
                 output_graph = False):

        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greed
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.replace_target_iter = replace_target_iter

        self.memory = np.zeros([self.memory_size,n_features*2+2])
        self.learn_step_counter = 0

        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t,e) for t,e in zip(t_params,e_params)]

        self.sess = tf.Session()

        if output_graph == True:
            writer = tf.summary.FileWriter("./logs", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def build_net(self):
        w_initializer = tf.truncated_normal_initializer(stddev=0.1)
        b_initializer = tf.constant_initializer(0.1)
        with tf.name_scope("input"):
            self.s = tf.placeholder(tf.float32, [None, 64, 64, 1],name = "state")
            self.q_target = tf.placeholder(tf.float32,[None,self.n_actions],name = "TargetQ")

        #q_eval
        with tf.variable_scope("q_eval"):
            c_name = ['eval_net_param',tf.GraphKeys.GLOBAL_VARIABLES]
            with tf.variable_scope("conv_1"):
                W = tf.get_variable("conv1_W",[5,5,1,16],initializer=w_initializer)
                b = tf.get_variable("conv1_b",[16],initializer=b_initializer)
                h=tf.nn.conv2d(self.s, W,strides=[1,1,1,1],padding='SAME')+b
                conv_1 = tf.nn.relu(h)
            with tf.variable_scope("pool_1"):
                pool_1 = tf.nn.max_pool(conv_1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

            with tf.variable_scope("conv_2"):
                W = tf.get_variable("conv2_W",[5,5,16,16],initializer=w_initializer)
                b = tf.get_variable("conv2_b", [16],initializer=b_initializer)
                h = tf.nn.conv2d(pool_1,W,strides=[1,1,1,1],padding='SAME')+b
                conv_2 = tf.nn.relu(h)
            with tf.variable_scope("pool_2"):
                pool_2 = tf.nn.max_pool(conv_2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

            with tf.variable_scope("fc_1"):
                shape = pool_2.getshape().as_list()
                W = tf.get_variable('fc1_w',[shape[1]*shape[2],128],initializer=w_initializer)
                b = tf.get_variable("fc1_b", [128],initializer=b_initializer)
                pool_2_flat = tf.reshape(pool_2,[-1,shape[1]*shape[2]])
                fc_1 = tf.nn.relu(tf.matmul(pool_2_flat,W)+b)

            with tf.variable_scope("output"):
                W = tf.get_variable('output_w',[128,4],initializer=w_initializer)
                b = tf.get_variable('output_b',[4],initializer = b_initializer)
                self.q_eval = tf.matmul(fc_1,W)+b

        with tf.variable_scope("loss"):
            loss = tf.reduce_mean(tf.squared_difference(self.q_target,self.q_eval))

        with tf.variable_scope("train"):
            self.train_step = tf.train.RMSPropOptimizer(self.lr).minimize(loss)
        #--------------------------------------------------------------------
        #q_target
        self.s_ = tf.placeholder(tf.float32, [None, 64, 64, 1])
        with tf.variable_scope("q_target"):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            with tf.variable_scope("conv_1"):
                W = tf.get_variable("conv1_W", [5, 5, 1, 16], initializer=w_initializer)
                b = tf.get_variable("conv1_b", [16], initializer=b_initializer)
                h = tf.nn.conv2d(self.s, W, strides=[1, 1, 1, 1], padding='SAME') + b
                conv_1 = tf.nn.relu(h)
            with tf.variable_scope("pool_1"):
                pool_1 = tf.nn.max_pool(conv_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            with tf.variable_scope("conv_2"):
                W = tf.get_variable("conv2_W", [5, 5, 16, 16], initializer=w_initializer)
                b = tf.get_variable("conv2_b", [16], initializer=b_initializer)
                h = tf.nn.conv2d(pool_1, W, strides=[1, 1, 1, 1], padding='SAME') + b
                conv_2 = tf.nn.relu(h)
            with tf.variable_scope("pool_2"):
                pool_2 = tf.nn.max_pool(conv_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            with tf.variable_scope("fc_1"):
                shape = pool_2.getshape().as_list()
                W = tf.get_variable('fc1_w', [shape[1] * shape[2], 128], initializer=w_initializer)
                b = tf.get_variable("fc1_b", [128], initializer=b_initializer)
                pool_2_flat = tf.reshape(pool_2, [-1, shape[1] * shape[2]])
                fc_1 = tf.nn.relu(tf.matmul(pool_2_flat, W) + b)

            with tf.variable_scope("output"):
                W = tf.get_variable('output_w', [128, 4], initializer=w_initializer)
                b = tf.get_variable('output_b', [4], initializer=b_initializer)
                self.q_next = tf.matmul(fc_1, W) + b

    def store_transition(self,s,a,r,s_):
        if not hasattr(self,'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((s, [a,r], s_))
        index = self.memory_counter % self.memory_counter
        self.memory[index, :] = transition
        self.memory_counter +=1

    def choose_action(self,observation):
        observation = observation[np.newaxis,:]
        if np.random.uniform() < self.epsilon:
            action_value = self.sess.run(self.q_eval,feed_dict={self.s :observation})
            action = np.argmax(action_value)
        else:
            action = np.random.choice(0,self.n_actions)
        return action

    def action(self, observation):
        observation = observation[np.newaxis,:]
        action_value = self.sess.run(self.q_eval,feed_dict={self.s:observation})
        return np.argmax(action_value)

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replace\n')

        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size,size= self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter,size= self.batch_size)

        batch_memory = self.memory[sample_index,:]

        q_next,q_eval = self.sess.run(
            [self.q_next,self.q_eval],
            feed_dict={self.s_: batch_memory[:, -self.n_features:],
                       self.s : batch_memory[:,self.n_features:]
            }
        )

        q_target = q_eval.copy()
        batch_index = np.arange(self.batch_size,dtype= np.int32)
        eval_act_index = batch_memory[:,self.n_features]
        reward = batch_memory[:,self.n_features+1]

        q_target[batch_index,eval_act_index] = reward + self.gamma*np.max(q_next,axis=1)
        self.sess.run(self.train_step)
        self.learn_step_counter +=1

    def saver(self):
        saver = tf.train.Saver()
        if self.learn_step_counter % 100==0:
            saver.restore(self.sess,"./mynet/save_net.ckpt")
            print("saved!")








