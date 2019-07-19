import numpy as np
import tensorflow as tf
from src_utils import stack_all_columns


class ModelAggregator:
    def __init__(self, hyp_list, n_epochs):
        self.n_epochs = n_epochs
        self.models = self._construct_all_models(hyp_list)
        self.train_images, self.train_labels = self._get_train_data()
        self.histories = []

    def _get_train_data(self):
        fashion_mnist = tf.keras.datasets.fashion_mnist
        train_images, train_labels = fashion_mnist.load_data()[0]
        return train_images / 255.0, train_labels

    def _construct_all_models(self, hyp_list):
        models = []
        for hyp in hyp_list:
            models.append(self._construct_model(*hyp))
        return models

    def train_all_models(self):
        self.histories = []
        for model in self.models:
            self.histories.append(self._train_model(model))

    def _train_model(self, model):
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        history = model.fit(self.train_images, self.train_labels,
                            validation_split=0.2, epochs=self.n_epochs)
        return history.history

    def _construct_model(self, num_layers, width):
        components = [tf.keras.layers.Flatten(input_shape=(28, 28))]
        for i in range(num_layers):
            components.append(tf.keras.layers.Dense(width, activation=tf.nn.relu))
        components.append(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
        return tf.keras.Sequential(components)

    def get_all_losses(self):
        """Returns NxM matrix for N epochs and M models."""
        losses = []
        for history in self.histories:
            losses.append(np.array(history['loss']).reshape((-1, 1)))
        return stack_all_columns(losses)
