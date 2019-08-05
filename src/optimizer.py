import tensorflow as tf


class Optimizer:
    def __init__(self, learning_rate=0.1, n_epochs=10):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs

    def minimize_loss(self, loss_func, params, args):
        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=self.learning_rate)
        for _ in range(self.n_epochs):
            with tf.GradientTape() as tape:
                loss = loss_func(*args)
            grads = tape.gradient(loss, params)
            optimizer.apply_gradients(zip(grads, params))
        return loss
