import network_factory
import nn_modelling
import feature_generator

import pickle


if __name__ == "__main__":
    model_name = 'pop_dem_ww_nuts_linear'
    rasters, generator_sequences, window_size = pickle.load(open("../../tmp/features.p", "rb"))
    train_gen = feature_generator.InMemoryStridedArrayGenerator(
        rasters,
        window_size = window_size,
        generator_sequences = [generator_sequence[0:int(len(generator_sequence) * 0.8)] for generator_sequence in generator_sequences]
    )
    test_gen = feature_generator.InMemoryStridedArrayGenerator(
        rasters,
        window_size = window_size,
        generator_sequences = [generator_sequence[int(len(generator_sequence) * 0.8):] for generator_sequence in generator_sequences]
    )
    train_dataset = feature_generator.create_tfds_from_imgenerator(train_gen, batch_size = 64, window_size = window_size, channel_n = sum([1 for x in rasters if rasters[x]['type'] == 'input']))
    test_dataset = feature_generator.create_tfds_from_imgenerator(test_gen, batch_size = 64, window_size = window_size, channel_n = sum([1 for x in rasters if rasters[x]['type'] == 'input']))

    internal_model = network_factory.build_network('linear', output_layer_count = len(generator_sequences))

    model, history = nn_modelling.train_classifier(
        train_dataset,
        test_dataset,
        num_epochs = 100,
        steps_per_epoch = 1000,
        internal_model = internal_model
    )

    model.save('/mnt/share/mnt/RESEARCH/SATELLITE/WORK/' + model_name + '_model')
