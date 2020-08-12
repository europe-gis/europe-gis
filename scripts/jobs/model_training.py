import numpy as np
import tensorflow as tf


class BorderRecognitionNetwork(object):
    def __init__(
        self,
        input_fields,
        image_size=100,
        nb_labels=2,
        architecture="resnet",
    ):
        if architecture == "resnet":
            self.model = self.resnet(
                input_fields = input_fields,
                hidden_layer_size = 1000,
                input_size = image_size
            )
        else:
            raise Exception

    def resnet(self, input_fields, hidden_layer_size = 1000, input_size = 100):

        full_input = tf.keras.layers.Input(shape=(input_size, input_size, len(input_fields)))

        inputs = [
            tf.keras.backend.expand_dims(
                tf.keras.layers.Lambda(
                    lambda x: x[:, :, :, i],
                    output_shape=(input_size, input_size, 1),
                    name = input_fields[i]
                )(full_input),
                axis=-1
            )
            for i in range(len(input_fields))
        ]
        parallel_models = [
            tf.keras.applications.ResNet50V2(
                include_top=True,
                input_shape=(input_size, input_size, 1),
                weights=None,
                pooling=None,
                classes=hidden_layer_size,
                classifier_activation='softmax'
            )
            for input_layer in inputs
        ]

        for model in parallel_models:
            model.layers.pop()

        for model, input_name in zip(parallel_models, input_fields):
            model._name = input_name + '_resnet50v2'
            for layer in model.layers:
                layer._name = layer.name + '_' + input_name

        concatenate = tf.keras.layers.concatenate(
            [
                model(
                    tf.keras.applications.resnet_v2.preprocess_input(
                        input_layer
                    )
                )
                for model, input_layer in zip(parallel_models, inputs)
            ],
            axis=1
        )
        prediction = tf.keras.layers.Dense(
            1,
            activation='sigmoid',
            use_bias=True
        )(concatenate)

        siamese_net = tf.keras.Model(
            inputs=full_input,
            outputs=prediction
        )
        siamese_net.summary()
        return siamese_net


class CVAE(tf.keras.Model):
    """Convolutional variational autoencoder."""
    """https://www.tensorflow.org/tutorials/generative/cvae"""
    def __init__(self, latent_dim):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
                tf.keras.layers.Flatten(),
                # No activation
                tf.keras.layers.Dense(latent_dim + latent_dim),
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                tf.keras.layers.Dense(units=7 * 7 * 32, activation=tf.nn.relu),
                tf.keras.layers.Reshape(target_shape=(7, 7, 32)),
                tf.keras.layers.Conv2DTranspose(
                    filters=64, kernel_size=3, strides=2, padding='same',
                    activation='relu'),
                tf.keras.layers.Conv2DTranspose(
                    filters=32, kernel_size=3, strides=2, padding='same',
                    activation='relu'),
                # No activation
                tf.keras.layers.Conv2DTranspose(
                    filters=1, kernel_size=3, strides=1, padding='same'),
            ]
        )

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

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

    optimizer = tf.keras.optimizers.Adam(1e-4)

    def log_normal_pdf(sample, mean, logvar, raxis=1):
        log2pi = tf.math.log(2. * np.pi)
        return tf.reduce_sum(
            -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
            axis=raxis)


    def compute_loss(model, x):
        mean, logvar = model.encode(x)
        z = model.reparameterize(mean, logvar)
        x_logit = model.decode(z)
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
        logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        logpz = log_normal_pdf(z, 0., 0.)
        logqz_x = log_normal_pdf(z, mean, logvar)
        return -tf.reduce_mean(logpx_z + logpz - logqz_x)


    @tf.function
    def train_step(model, x, optimizer):
        """Executes one training step and returns the loss.

        This function computes the loss and gradients, and uses the latter to
        update the model's parameters.
        """
        with tf.GradientTape() as tape:
            loss = compute_loss(model, x)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))



vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam())
vae.fit(mnist_digits, epochs=30, batch_size=128)
