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
