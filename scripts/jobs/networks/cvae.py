import numpy as np
import tensorflow as tf
import matplotlib.pyplot as pyplot


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)


def compute_loss(model, x):
    x = tf.dtypes.cast(x, tf.float32)
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)
    x_logit = tf.dtypes.cast(x_logit, tf.float32)
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    logpz = tf.dtypes.cast(logpz, tf.float32)
    logqz_x = tf.dtypes.cast(logqz_x, tf.float32)
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)


train_loss = tf.keras.metrics.Mean(name='train_loss')
test_loss = tf.keras.metrics.Mean(name='test_loss')


class CCVAE(tf.keras.Model):
    """Convolutional variational autoencoder."""
    """https://www.tensorflow.org/tutorials/generative/cvae"""
    def __init__(self, latent_dim):
        super(CCVAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
                tf.keras.layers.Conv2D(
                    filters=256, kernel_size=3, strides=(2, 2), activation='relu'),
                tf.keras.layers.Conv2D(
                    filters=512, kernel_size=3, strides=(2, 2), activation='relu'),
                tf.keras.layers.Flatten(),
                # No activation
                tf.keras.layers.Dense(latent_dim + latent_dim)
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                tf.keras.layers.Dense(units=7 * 7 * 256, activation=tf.nn.relu),
                tf.keras.layers.Reshape(target_shape=(7, 7, 256)),
                tf.keras.layers.Conv2DTranspose(
                    filters=512, kernel_size=3, strides=2, padding='same',
                    activation='relu'),
                tf.keras.layers.Conv2DTranspose(
                    filters=256, kernel_size=3, strides=2, padding='same',
                    activation='relu'),
                # No activation
                tf.keras.layers.Conv2DTranspose(
                    filters=1, kernel_size=3, strides=1, padding='same')
            ]
        )

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

    def call(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        predictions = self.sample(z)
        return predictions

    @tf.function
    def train_step(self, x):
        """Executes one training step and returns the loss.

        This function computes the loss and gradients, and uses the latter to
        update the model's parameters.
        """
        with tf.GradientTape() as tape:
            loss = compute_loss(self, x)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        train_loss.update_state(loss)
        return {"loss": train_loss.result()}

    @tf.function
    def test_step(self, x):
        t_loss = compute_loss(self, x)
        return {"loss": t_loss}

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def BatchTrain(self, train_ds, STEPS_PER_EPOCH):
        train_loss.reset_states()
        test_loss.reset_states()

        i = 0
        loss_list = []
        for image in train_ds:
            loss_list.append(self.train_step(
                tf.dtypes.cast(
                    tf.expand_dims(
                        image,
                        axis=0
                    ),
                    tf.float32
                )
            )['loss'])
            i += 1
            if i > STEPS_PER_EPOCH:
                break
        train_loss.update_state(loss_list)

        return {"training_loss": train_loss.result().numpy()}


def TrainCCVAEModel(train_ds, test_ds):

    model = CCVAE(10)

    STEPS_PER_EPOCH = 64
    EPOCHS = 10000

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4)
    )
    for epoch in range(EPOCHS):
        print("Epoch: ", epoch)

        model.fit(
            train_ds,
            epochs=1,
            steps_per_epoch=20,
            verbose=1,
            validation_data=test_ds,
            validation_steps=1
        )

        print(model.BatchTrain(train_ds, STEPS_PER_EPOCH))

        i = 0
        fig = pyplot.figure(figsize=(10, 10))
        loss_list = []
        for image in test_ds:
            loss_list.append(model.test_step(
                tf.expand_dims(
                    image,
                    axis=0
                )
            )['loss'])

            pyplot.subplot(10, 8, i + 1)
            pyplot.imshow(image, cmap='pink')
            pred = model.call(
                tf.expand_dims(
                    image,
                    axis=0
                )
            )
            pyplot.subplot(10, 8, i + 2)
            pyplot.imshow(tf.squeeze(pred), cmap='pink')
            i += 2
            if i > 10 * 4 + 2:
                test_loss.update_state(loss_list)
                print({"validation_loss": test_loss.result().numpy()})
                pyplot.show()
                break

    return model
