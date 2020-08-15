import itertools
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as pyplot


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
                tf.keras.layers.Dense(latent_dim + latent_dim),
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
                    filters=1, kernel_size=3, strides=1, padding='same'),
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
    optimizer = tf.keras.optimizers.Adam(1e-4)
    STEPS_PER_EPOCH = 64
    EPOCHS = 10000
    model.compile(
        optimizer=optimizer
    )
    for epoch in range(EPOCHS):
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


class CAE(tf.keras.Model):
    """https://medium.com/red-buffer/autoencoders-guide-and-code-in-tensorflow-2-0-a4101571ce56"""
    def __init__(self):
        super(CAE, self).__init__()
        filter_base = 8 * 1
        latent_dim = 10

        self.conv1 = tf.keras.layers.Conv2D(2 * filter_base, (3, 3), activation='relu', padding='same')
        self.maxp1 = tf.keras.layers.MaxPooling2D((2, 2), padding='same')
        self.conv2 = tf.keras.layers.Conv2D(filter_base, (3, 3), activation='relu', padding='same')
        self.maxp2 = tf.keras.layers.MaxPooling2D((2, 2), padding='same')
        self.conv3 = tf.keras.layers.Conv2D(filter_base, (3, 3), activation='relu', padding='same')

        self.encoded = tf.keras.layers.MaxPooling2D((2, 2), padding='same')

        self.encoded_flatten = tf.keras.layers.Flatten()
        self.latent = tf.keras.layers.Dense(latent_dim)
        self.rebuild = tf.keras.layers.Dense(4 * 4 * filter_base)
        self.reshape = tf.keras.layers.Reshape((4, 4, filter_base))

        self.conv4 = tf.keras.layers.Conv2D(filter_base, (3, 3), activation='relu', padding='same')
        self.upsample1 = tf.keras.layers.UpSampling2D((2, 2))
        self.conv5 = tf.keras.layers.Conv2D(filter_base, (3, 3), activation='relu', padding='same')
        self.upsample2 = tf.keras.layers.UpSampling2D((2, 2))
        self.conv6 = tf.keras.layers.Conv2D(2 * filter_base, (3, 3), activation='relu')
        self.upsample3 = tf.keras.layers.UpSampling2D((2, 2))
        self.conv7 = tf.keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')

    def call(self, x):
        x = self.conv1(x)
        x = self.maxp1(x)
        x = self.conv2(x)
        x = self.maxp2(x)
        x = self.conv3(x)
        x = self.encoded(x)

        x = self.encoded_flatten(x)
        x = self.latent(x)
        x = self.rebuild(x)
        x = self.reshape(x)

        x = self.conv4(x)
        x = self.upsample1(x)
        x = self.conv5(x)
        x = self.upsample2(x)
        x = self.conv6(x)
        x = self.upsample3(x)
        x = self.conv7(x)
        return x

    def encode(self, x):
        x = self.conv1(x)
        x = self.maxp1(x)
        x = self.conv2(x)
        x = self.maxp2(x)
        x = self.conv3(x)
        x = self.encoded(x)
        x = self.encoded_flatten(x)
        x = self.latent(x)
        return x


def loss(x, x_bar):
    return tf.losses.mean_squared_error(x, x_bar)


def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        reconstruction = model(inputs)
        loss_value = loss(targets, reconstruction)
    return loss_value, tape.gradient(loss_value, model.trainable_variables), reconstruction


def TrainCAEModel(train_ds, test_ds):

    model = CAE()
    optimizer = tf.optimizers.Adam(learning_rate=0.1)  # 0.001
    num_epochs = 100
    batch_size = 256
    for epoch in range(num_epochs):
        print("Epoch: ", epoch)
        for x in range(0, 10):
            x_inp = np.stack(list(itertools.islice(train_ds, batch_size)), axis=0)
            loss_value, grads, reconstruction = grad(model, x_inp, x_inp)
            optimizer.apply_gradients(
                zip(
                    grads,
                    model.trainable_variables
                )
            )

            print("Loss: {}".format(
                np.sum(loss(x_inp, reconstruction).numpy()) / batch_size)
            )
        if epoch % 5 == 0:
            x_test = np.stack(list(itertools.islice(test_ds, 40)), axis=0)
            i = 0
            fig = pyplot.figure(figsize=(10, 10))
            for image in x_test:
                pyplot.subplot(10, 8, i + 1)
                pyplot.imshow(image, cmap='pink')
                pred = model(tf.expand_dims(
                    image,
                    axis=0
                ))
                pyplot.subplot(10, 8, i + 2)
                pyplot.imshow(tf.squeeze(pred), cmap='pink')
                i += 2
            pyplot.show()

    return model
