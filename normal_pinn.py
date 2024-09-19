import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt


class Nuclear_Reactor_PINN:
    def __init__(self, t, y, layers):
        self.lb = t.min()
        self.ub = t.max()

        self.t = t
        self.n = np.array(y[0])
        self.c1 = np.array(y[1])
        self.c2 = np.array(y[2])
        self.c3 = np.array(y[3])
        self.c4 = np.array(y[4])
        self.c5 = np.array(y[5])
        self.c6 = np.array(y[6])
        
        self.coef = [-0.0032, 0.2564, -5.8336, 54.353, -1168.5]
        self.beta_ = np.array([0.000213, 0.001413, 0.001264, 0.002548, 0.000742, 0.000271])
        self.lambda_ = np.array([0.01244, 0.0305, 0.1114, 0.3013, 1.1361, 3.013])
        self.n_0 = 70000
        self.LAMBDA = 8.13e-5
        self.c_0 = self.beta_ * self.n_0 / self.LAMBDA / self.lambda_
        self.S_0 = 10000000

        self.layers = layers

        self.weights, self.biases = self.initialize_NN(layers)

        self.t_tf = tf.convert_to_tensor(self.t[:, None], dtype=tf.float32)
        self.n_tf = tf.convert_to_tensor(self.n, dtype=tf.float32)
        self.c1_tf = tf.convert_to_tensor(self.c1, dtype=tf.float32)
        self.c2_tf = tf.convert_to_tensor(self.c2, dtype=tf.float32)
        self.c3_tf = tf.convert_to_tensor(self.c3, dtype=tf.float32)
        self.c4_tf = tf.convert_to_tensor(self.c4, dtype=tf.float32)
        self.c5_tf = tf.convert_to_tensor(self.c5, dtype=tf.float32)
        self.c6_tf = tf.convert_to_tensor(self.c6, dtype=tf.float32)

        self.optimizer_Adam = tf.optimizers.Adam()

    def rho(self, t):
        soln = 0
        for i in range(5):
            soln += (t * 11 / 60) ** i * self.coef[4 - i]
        return soln

    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(tf.random.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    def neural_net(self, t, weights, biases):
        num_layers = len(weights) + 1
        H = 2.0 * (t - self.lb) / (self.ub - self.lb) - 1.0
        for l in range(num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_pinn(self, t):
        nn_output = self.neural_net(t, self.weights, self.biases)
        n = nn_output[:, 0:1]
        c1 = nn_output[:, 1:2]
        c2 = nn_output[:, 2:3]
        c3 = nn_output[:, 3:4]
        c4 = nn_output[:, 4:5]
        c5 = nn_output[:, 5:6]
        c6 = nn_output[:, 6:7]

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(t)
            n = self.neural_net(t, self.weights, self.biases)[:, 0:1]
            c1 = self.neural_net(t, self.weights, self.biases)[:, 1:2]
            c2 = self.neural_net(t, self.weights, self.biases)[:, 2:3]
            c3 = self.neural_net(t, self.weights, self.biases)[:, 3:4]
            c4 = self.neural_net(t, self.weights, self.biases)[:, 4:5]
            c5 = self.neural_net(t, self.weights, self.biases)[:, 5:6]
            c6 = self.neural_net(t, self.weights, self.biases)[:, 6:7]

        n_t = tape.gradient(n, t)
        c1_t = tape.gradient(c1, t)
        c2_t = tape.gradient(c2, t)
        c3_t = tape.gradient(c3, t)
        c4_t = tape.gradient(c4, t)
        c5_t = tape.gradient(c5, t)
        c6_t = tape.gradient(c6, t)
        del tape

        return n, n_t, c1, c1_t, c2, c2_t, c3, c3_t, c4, c4_t, c5, c5_t, c6, c6_t

    def compute_loss(self):
        n_pred, n_t, c1_pred, c1_t, c2_pred, c2_t, c3_pred, c3_t, c4_pred, c4_t, c5_pred, c5_t, c6_pred, c6_t = self.net_pinn(self.t_tf)
        mse_loss = tf.reduce_mean(tf.square(self.n_tf - n_pred)) + \
                   tf.reduce_mean(tf.square(self.c1_tf - c1_pred)) + \
                   tf.reduce_mean(tf.square(self.c2_tf - c2_pred)) + \
                   tf.reduce_mean(tf.square(self.c3_tf - c3_pred)) + \
                   tf.reduce_mean(tf.square(self.c4_tf - c4_pred)) + \
                   tf.reduce_mean(tf.square(self.c5_tf - c5_pred)) + \
                   tf.reduce_mean(tf.square(self.c6_tf - c6_pred))

        f_n = self.S_0 + (self.rho(self.t_tf) - sum(self.beta_)) / self.LAMBDA * self.n_tf + \
              self.c1_tf * self.lambda_[0] + self.c4_tf * self.lambda_[3] + \
              self.c2_tf * self.lambda_[1] + self.c5_tf * self.lambda_[4] + \
              self.c3_tf * self.lambda_[2] + self.c6_tf * self.lambda_[5]

        f_c1 = self.beta_[0] / self.LAMBDA * self.n_tf - self.lambda_[0] * self.c1_tf
        f_c2 = self.beta_[1] / self.LAMBDA * self.n_tf - self.lambda_[1] * self.c2_tf
        f_c3 = self.beta_[2] / self.LAMBDA * self.n_tf - self.lambda_[2] * self.c3_tf
        f_c4 = self.beta_[3] / self.LAMBDA * self.n_tf - self.lambda_[3] * self.c4_tf
        f_c5 = self.beta_[4] / self.LAMBDA * self.n_tf - self.lambda_[4] * self.c5_tf
        f_c6 = self.beta_[5] / self.LAMBDA * self.n_tf - self.lambda_[5] * self.c6_tf

        ode_loss = tf.reduce_mean(tf.square(n_t - f_n)) + \
                   tf.reduce_mean(tf.square(c1_t - f_c1)) + \
                   tf.reduce_mean(tf.square(c2_t - f_c2)) + \
                   tf.reduce_mean(tf.square(c3_t - f_c3)) + \
                   tf.reduce_mean(tf.square(c4_t - f_c4)) + \
                   tf.reduce_mean(tf.square(c5_t - f_c5)) + \
                   tf.reduce_mean(tf.square(c6_t - f_c6))

        IC_loss = tf.reduce_mean(tf.square(self.neural_net(tf.convert_to_tensor([[0.0]], dtype=tf.float32), self.weights, self.biases) - tf.convert_to_tensor([[self.n_0, *self.c_0]], dtype=tf.float32)))

        loss = mse_loss + ode_loss + IC_loss
        return loss

    def train(self, nIter):
        start_time = time.time()
        for it in range(nIter):
            with tf.GradientTape() as tape:
                loss_value = self.compute_loss()
            grads = tape.gradient(loss_value, self.weights + self.biases)
            self.optimizer_Adam.apply_gradients(zip(grads, self.weights + self.biases))
            if it %50 == 0:
                print(f'{it} epochs completed')

    def predict(self, t_star):
        t_star_tf = tf.convert_to_tensor(t_star, dtype=tf.float32)
        nn_output = self.neural_net(t_star_tf, self.weights, self.biases)
        n_star = nn_output[:, 0:1]
        c1_star = nn_output[:, 1:2]
        c2_star = nn_output[:, 2:3]
        c3_star = nn_output[:, 3:4]
        c4_star = nn_output[:, 4:5]
        c5_star = nn_output[:, 5:6]
        c6_star = nn_output[:, 6:7]

        return n_star.numpy(), c1_star.numpy(), c2_star.numpy(), c3_star.numpy(), c4_star.numpy(), c5_star.numpy(), c6_star.numpy()
    

# Define the neural network layers
layers = [1, 128, 128, 128, 128, 7]


with open('data.csv', 'r') as data_file:
    data = data_file.readlines()

for i in range(len(data)):
    data[i] = data[i].strip()
    
y = []
for i in range(int(len(data)/100)):
    y.append(data[100*i:100*(i+1)])

t = np.linspace(0, 240, 100)
# Instantiate the PINN model
model = Nuclear_Reactor_PINN(t, y, layers)

# Train the model
model.train(500)

# Predict the position and velocity at new time points
t_star = np.linspace(0, 2, 50)[:, None]
n_star, c1_star, c2_star, c3_star, c4_star, c5_star, c6_star = model.predict(t)

plt.plot(t, n_star)
plt.plot(t, y[0])
