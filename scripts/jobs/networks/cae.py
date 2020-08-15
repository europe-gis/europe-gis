import itertools
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as pyplot


class CustomMSE(tf.keras.losses.Loss):
    def __init__(self, regularization_factor=0.1, name="custom_mse"):
        super().__init__(name=name)
        self.regularization_factor = regularization_factor

    def call(self, y_true, y_pred):
        mse = tf.math.reduce_mean(tf.square(y_true - y_pred))
        reg = tf.math.reduce_mean(tf.square(0.5 - y_pred))
        return mse + reg * self.regularization_factor


class CAE(tf.keras.Model):
    """https://medium.com/red-buffer/autoencoders-guide-and-code-in-tensorflow-2-0-a4101571ce56"""
    def __init__(self):
        super(CAE, self).__init__()
        filter_base = 128
        filter_multiplicator = 1
        latent_dim = 10

        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(2 * filter_base, (3, 3), activation='relu', padding='same'),
                tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
                tf.keras.layers.Conv2D(filter_base, (3, 3), activation='relu', padding='same'),
                tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
                tf.keras.layers.Conv2D(filter_base, (3, 3), activation='relu', padding='same'),

                tf.keras.layers.MaxPooling2D((2, 2), padding='same'),

                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(latent_dim)
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(4 * 4 * filter_base),
                tf.keras.layers.Reshape((4, 4, filter_base)),
                tf.keras.layers.Conv2D(filter_base, (3, 3), activation='relu', padding='same'),
                tf.keras.layers.UpSampling2D((2, 2)),
                tf.keras.layers.Conv2D(filter_base, (3, 3), activation='relu', padding='same'),
                tf.keras.layers.UpSampling2D((2, 2)),
                tf.keras.layers.Conv2D(2 * filter_base, (3, 3), activation='relu'),
                tf.keras.layers.UpSampling2D((2, 2)),
                tf.keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')
            ]
        )

        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
                tf.keras.layers.Conv2D(
                    filters=filter_base * filter_multiplicator, kernel_size=3, strides=(2, 2), activation='relu'),
                tf.keras.layers.Conv2D(
                    filters=filter_base * 2 * filter_multiplicator, kernel_size=3, strides=(2, 2), activation='relu'),
                tf.keras.layers.Flatten(),
                # No activation
                tf.keras.layers.Dense(latent_dim)
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                tf.keras.layers.Dense(units=7 * 7 * filter_base * filter_multiplicator, activation=tf.nn.relu),
                tf.keras.layers.Reshape(target_shape=(7, 7, filter_base * filter_multiplicator)),
                tf.keras.layers.Conv2DTranspose(
                    filters=filter_base * 2 * filter_multiplicator, kernel_size=3, strides=2, padding='same',
                    activation='relu'),
                tf.keras.layers.Conv2DTranspose(
                    filters=filter_base * filter_multiplicator, kernel_size=3, strides=2, padding='same',
                    activation='relu'),
                # No activation
                tf.keras.layers.Conv2DTranspose(
                    filters=1, kernel_size=3, strides=1, padding='same')
            ]
        )

        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(28, 28, 1)),
                tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=2),
                tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same', strides=2)
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2DTranspose(8, kernel_size=3, strides=2, activation='relu', padding='same'),
                tf.keras.layers.Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same'),
                tf.keras.layers.Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same')
            ]
        )

        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(latent_dim, activation='relu'),
            ]
        )
        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(784, activation='sigmoid'),
                tf.keras.layers.Reshape((28, 28))
            ]
        )

    def call(self, x, training=False):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encode(self, x):
        x = self.encoder(x)
        return x

    @tf.function
    def train_step(self, x):
        """Executes one training step and returns the loss.

        This function computes the loss and gradients, and uses the latter to
        update the model's parameters.
        """
        with tf.GradientTape() as tape:
            x_pred = self(x, training=True)
            loss = self.compiled_loss(x, x_pred, regularization_losses=self.losses)

        gradients = tape.gradient(loss, self.trainable_variables)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(x, x_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def test_step(self, x):
        # Compute predictions
        x_pred = self(x, training=False)
        # Updates the metrics tracking the loss
        self.compiled_loss(x, x_pred, regularization_losses=self.losses)
        # Update the metrics.
        self.compiled_metrics.update_state(x, x_pred)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}


class PrintSampleOfImages(tf.keras.callbacks.Callback):
    def __init__(self, test_ds):
        super(PrintSampleOfImages, self).__init__()
        self.test_ds = test_ds

    def on_epoch_end(self, epoch, logs=None):
        x_test = np.stack(list(itertools.islice(self.test_ds, 1)), axis=0)
        i = 0
        fig = pyplot.figure(figsize=(10, 10))
        for image in x_test[0][0:40]:
            pyplot.subplot(10, 8, i + 1)
            pyplot.imshow(image, cmap='pink')
            pred = self.model(tf.expand_dims(
                image,
                axis=0
            ))
            pyplot.subplot(10, 8, i + 2)
            pyplot.imshow(tf.squeeze(pred), cmap='pink')
            i += 2
        pyplot.show()


def TrainCAEModel(train_ds, test_ds):

    model = CAE()
    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=0.001),  # 0.001
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=["mae"]
    )

    num_epochs = 100

    model.fit(
        train_ds,
        epochs=num_epochs,
        steps_per_epoch=100,
        verbose=1,
        # validation_data=test_ds,
        # validation_steps=1,
        callbacks=[PrintSampleOfImages(test_ds)]
    )

    return model
