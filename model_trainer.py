Adaimport tensorflow as tf
import helper_functions
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.losses import MSE
from tensorflow.keras.optimizers import Adam
import numpy as np

STATE_SIZE = 11
Q_VALUE_SIZE = 3
LR = 1e-3


q_net = Sequential([
    Input(shape = (STATE_SIZE,)),
    Dense(units = 64, activation = 'relu'),
    Dropout(0.1),
    Dense(units = 64, activation = 'relu'),
    Dropout(0.05),
    Dense(units = Q_VALUE_SIZE, activation = 'linear')
])

target_q_net = Sequential([
    Input(shape = (STATE_SIZE,)),
    Dense(units = 64, activation = 'relu'),
    Dropout(0.1),
    Dense(units = 64, activation = 'relu'),
    Dropout(0.05),
    Dense(units = Q_VALUE_SIZE, activation = 'linear')
])


optimizer = Adam(learning_rate = LR)


def compute_loss(experiences, gamma):
    states, actions, rewards, next_states, done_values = experiences
    max_target_qsa = tf.reduce_max(target_q_net(next_states), axis = -1)
    target_q = done_values * rewards + (1 - done_values) * (rewards + gamma* max_target_qsa)
    predicted_q = q_net(states)
    # reshape predicted_q from (64,3) to (64,) to match target_q
    predicted_q = tf.gather_nd(params = predicted_q, indices = tf.stack([tf.range(predicted_q.shape[0]), tf.cast(tf.argmax(actions, axis = 1), tf.int32)], axis = 1))
    loss = MSE(target_q, predicted_q)
    return loss

@tf.function
def agent_learn(experiences, gamma): # one training step
    with tf.GradientTape() as tape:
        loss = compute_loss(experiences, gamma)
    
    # calculate the gradients for all Ws and Bs
    gradients = tape.gradient(loss, q_net.trainable_variables)
    # update q_net via gradient descent
    optimizer.apply_gradients(zip(gradients, q_net.trainable_variables))
    # softupdate target_q_net
    helper_functions.target_qNet_softupdate(q_net, target_q_net)


print(q_net.summary())


print(q_net(np.array([[1,1,1,1,1,1,1,1,1,1,1]])))