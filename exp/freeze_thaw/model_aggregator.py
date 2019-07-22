import numpy as np
import tensorflow as tf
from exp.freeze_thaw.utils import hyp_to_key


class ModelAggregator:
    def __init__(self, hyp_list):
        self.models = self._construct_all_models(hyp_list)
        self.train_images, self.train_labels = self._get_train_data()
        self.val_losses = {}

    def get_all_losses(self):
        """Returns NxM matrix for N epochs and M models."""
        losses = {}
        for key in self.val_losses:
            single_val_losses = self.val_losses[key]
            losses[key] = np.array(single_val_losses).reshape((-1, 1))
        return losses

    def get_specified_models(self, keys):
        return {key: self.models[key] for key in keys}

    def get_all_keys(self):
        return tuple(self.models.keys())

    def train_model_given_key(self, key, n_epochs=1):
        model = self.models[key]
        history = self._train_model(model, n_epochs)
        self._update_val_losses(key, history)

    def train_all_models(self, n_epochs):
        for key in self.models:
            model = self.models[key]
            history = self._train_model(model, n_epochs)
            self._update_val_losses(key, history)

    def _get_train_data(self):
        fashion_mnist = tf.keras.datasets.fashion_mnist
        train_images, train_labels = fashion_mnist.load_data()[0]
        return train_images / 255.0, train_labels

    def _construct_all_models(self, hyp_list):
        models = {}
        for hyp in hyp_list:
            key = hyp_to_key(hyp)
            models[key] = self._construct_model(*hyp)
        return models

    def _update_val_losses(self, key, history):
        if key not in self.val_losses:
            self.val_losses[key] = []
        self.val_losses[key] += history['val_loss']

    def _train_model(self, model, n_epochs):
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        history = model.fit(self.train_images, self.train_labels, verbose=2,
                            validation_split=0.2, epochs=n_epochs)
        return history.history

    def _construct_model(self, num_layers, width):
        components = [tf.keras.layers.Flatten(input_shape=(28, 28))]
        for i in range(num_layers):
            components.append(tf.keras.layers.Dense(width, activation=tf.nn.relu))
        components.append(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
        return tf.keras.Sequential(components)
