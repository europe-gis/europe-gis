import tensorflow as tf


def build_network(network_name, **kwargs):

    if network_name == 'linear':
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Activation('relu'),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(1, use_bias=True, activation='sigmoid')
            ]
        )
        return model
    elif network_name == 'simple_dense':
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(kwargs['internal_dense_size'], activation='relu'),
                tf.keras.layers.Dense(1, use_bias=True, activation='sigmoid')
            ]
        )
        return model
    elif network_name == 'simple_cnn':
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=2),
                tf.keras.layers.SpatialDropout2D(0.5),
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', strides=2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(kwargs['internal_dense_size'], activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(1, use_bias=True, activation='sigmoid')
            ]
        )
        return model
    elif network_name == 'siamese_resnet':

        full_input = tf.keras.layers.Input(shape=(kwargs.input_size, kwargs.input_size, len(kwargs.input_fields)))
        inputs = [
            tf.keras.backend.expand_dims(
                tf.keras.layers.Lambda(
                    lambda x: x[:, :, :, i],
                    output_shape=(kwargs.input_size, kwargs.input_size, 1),
                    name = kwargs.input_fields[i]
                )(full_input),
                axis=-1
            )
            for i in range(len(kwargs.input_fields))
        ]
        parallel_models = [
            tf.keras.applications.ResNet50V2(
                include_top=True,
                input_shape=(kwargs.input_size, kwargs.input_size, 1),
                weights=None,
                pooling=None,
                classes=kwargs.hidden_layer_size,
                classifier_activation='relu'
            )
            for input_layer in inputs
        ]

        for model in parallel_models:
            model.layers.pop()

        for model, input_name in zip(parallel_models, kwargs.input_fields):
            model._name = input_name + '_resnet50v2'
            for layer in model.layers:
                layer._name = layer.name + '_' + input_name

        if len(kwargs.input_fields) > 1:
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
        else:
            concatenate = parallel_models[0](
                tf.keras.applications.resnet_v2.preprocess_input(
                    inputs[0]
                )
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

    elif network_name == 'simple_resnet':

        full_input = tf.keras.layers.Input(shape=(kwargs['input_size'], kwargs['input_size'], kwargs['channel_n']))
        parallel_models = tf.keras.applications.ResNet50V2(
            include_top=True,
            input_shape=(kwargs['input_size'], kwargs['input_size'], kwargs['channel_n']),
            weights=None,
            pooling=None,
            classes=1,
            classifier_activation='sigmoid'
        )
        prediction = parallel_models(
            tf.keras.applications.resnet_v2.preprocess_input(
                full_input
            )
        )
        # prediction = parallel_models(full_input)
        network = tf.keras.Model(
            inputs=full_input,
            outputs=prediction
        )
        return network

    elif network_name == 'simple_vgg':

        full_input = tf.keras.layers.Input(shape=(kwargs['input_size'], kwargs['input_size'], kwargs['channel_n']))
        parallel_models = tf.keras.applications.VGG16(
            include_top=True,
            input_shape=(kwargs['input_size'], kwargs['input_size'], kwargs['channel_n']),
            weights=None,
            pooling=None,
            classes=1,
            classifier_activation='sigmoid'
        )
        prediction = parallel_models(
            tf.keras.applications.vgg16.preprocess_input(
                full_input
            )
        )
        # prediction = parallel_models(full_input)
        network = tf.keras.Model(
            inputs=full_input,
            outputs=prediction
        )
        return network

    elif network_name == 'simple_inception':

        full_input = tf.keras.layers.Input(shape=(kwargs['input_size'], kwargs['input_size'], kwargs['channel_n']))
        parallel_models = tf.keras.applications.InceptionV3(
            include_top=True,
            input_shape=(kwargs['input_size'], kwargs['input_size'], kwargs['channel_n']),
            weights=None,
            pooling=None,
            classes=1,
            classifier_activation='sigmoid'
        )
        prediction = parallel_models(
            tf.keras.applications.inception_v3.preprocess_input(
                full_input
            )
        )
        # prediction = parallel_models(full_input)
        network = tf.keras.Model(
            inputs=full_input,
            outputs=prediction
        )
        return network

    return
