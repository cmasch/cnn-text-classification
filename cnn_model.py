# -*- coding: utf-8 -*-
"""
CNN model for text classification implemented in TensorFlow / Keras.
This implementation is based on the original paper of Yoon Kim [1] for classification using words.
Besides I add charachter level input [2].

# References
- [1] [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)
- [2] [Character-level Convolutional Networks for Text Classification](https://arxiv.org/abs/1509.01626)

@author: Christopher Masch
"""

import keras
from keras import layers


class CNN:
    __version__ = '0.1.0'

    def __init__(self, embedding_layer=None, num_words=None, embedding_dim=None,
                 max_seq_length=100, kernel_sizes=[3, 4, 5], feature_maps=[100, 100, 100],
                 use_char=False, char_max_length=200, alphabet_size=None, char_kernel_sizes=[3, 10, 20],
                 char_feature_maps=[100, 100, 100], hidden_units=100, dropout_rate=None, nb_classes=None):
        """
        Arguments:
            embedding_layer : If not defined with pre-trained embeddings it will be created from scratch (default: None)
            num_words       : Maximal amount of words in the vocabulary (default: None)
            embedding_dim   : Dimension of word representation (default: None)
            max_seq_length  : Max length of word sequence (default: 100)
            filter_sizes    : An array of filter sizes per channel (default: [3,4,5])
            feature_maps    : Defines the feature maps per channel (default: [100,100,100])
            use_char        : If True, char-based model will be added to word-based model
            char_max_length : Max length of char sequence (default: 200)
            alphabet_size   : Amount of differnent chars used for creating embeddings (default: None)
            hidden_units    : Hidden units per convolution channel (default: 100)
            dropout_rate    : If defined, dropout will be added after embedding layer & concatenation (default: None)
            nb_classes      : Number of classes which can be predicted
        """

        # WORD-level
        self.embedding_layer = embedding_layer
        self.num_words = num_words
        self.max_seq_length = max_seq_length
        self.embedding_dim = embedding_dim
        self.kernel_sizes = kernel_sizes
        self.feature_maps = feature_maps
        # CHAR-level
        self.use_char = use_char
        self.char_max_length = char_max_length
        self.alphabet_size = alphabet_size
        self.char_kernel_sizes = char_kernel_sizes
        self.char_feature_maps = char_feature_maps
        # General
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.nb_classes = nb_classes

    def build_model(self):
        """
        Build the model

        Returns:
            Model           : Keras model instance
        """

        # Checks
        if len(self.kernel_sizes) != len(self.feature_maps):
            raise Exception('Please define `kernel_sizes` and `feature_maps` with the same amount.')
        if not self.embedding_layer and (not self.num_words or not self.embedding_dim):
            raise Exception('Please define `num_words` and `embedding_dim` if you not using a pre-trained embedding.')
        if self.use_char and (not self.char_max_length or not self.alphabet_size):
            raise Exception('Please define `char_max_length` and `alphabet_size` if you are using char.')

        # Building word-embeddings from scratch
        if self.embedding_layer is None:
            self.embedding_layer = layers.Embedding(
                input_dim=self.num_words,
                output_dim=self.embedding_dim,
                input_length=self.max_seq_length,
                weights=None, trainable=True,
                name="word_embedding"
            )

        # WORD-level
        word_input = layers.Input(shape=(self.max_seq_length,), dtype='int32', name='word_input')
        x = self.embedding_layer(word_input)
        if self.dropout_rate:
            x = layers.Dropout(self.dropout_rate)(x)
        x = self.building_block(x, self.kernel_sizes, self.feature_maps)
        x = layers.Activation('relu')(x)
        prediction = layers.Dense(self.nb_classes, activation='softmax')(x)

        # CHAR-level
        if self.use_char:
            char_input = layers.Input(shape=(self.char_max_length,), dtype='int32', name='char_input')
            x_char = layers.Embedding(
                input_dim=self.alphabet_size + 1,
                output_dim=50,
                input_length=self.char_max_length,
                name='char_embedding'
            )(char_input)
            x_char = self.building_block(x_char, self.char_kernel_sizes, self.char_feature_maps)
            x_char = layers.Activation('relu')(x_char)
            x_char = layers.Dense(self.nb_classes, activation='softmax')(x_char)

            prediction = layers.Average()([prediction, x_char])
            return keras.Model(inputs=[word_input, char_input], outputs=prediction, name='CNN_Word_Char')

        return keras.Model(inputs=word_input, outputs=prediction, name='CNN_Word')

    def building_block(self, input_layer, kernel_sizes, feature_maps):
        """
        Creates several CNN channels in parallel and concatenate them

        Arguments:
            input_layer : Layer which will be the input for all convolutional blocks
            kernel_sizes: Array of kernel sizes (working as n-gram filter)
            feature_maps: Array of feature maps

        Returns:
            x           : Building block with one or several channels
        """
        channels = []
        for ix in range(len(kernel_sizes)):
            x = self.create_channel(input_layer, kernel_sizes[ix], feature_maps[ix])
            channels.append(x)

        # Check how many channels, one channel doesn't need a concatenation
        if (len(channels) > 1):
            x = layers.concatenate(channels)
        return x

    def create_channel(self, x, kernel_size, feature_map):
        """
        Creates a layer, working channel wise

        Arguments:
            x           : Input for convoltuional channel
            kernel_size : Kernel size for creating Conv1D
            feature_map : Feature map

        Returns:
            x           : Channel including (Conv1D + {GlobalMaxPooling & GlobalAveragePooling} + Dense [+ Dropout])
        """
        x = layers.SeparableConv1D(feature_map, kernel_size=kernel_size, activation='relu',
                                   strides=1, padding='valid', depth_multiplier=4)(x)

        x1 = layers.GlobalMaxPooling1D()(x)
        x2 = layers.GlobalAveragePooling1D()(x)
        x = layers.concatenate([x1, x2])

        x = layers.Dense(self.hidden_units)(x)
        if self.dropout_rate:
            x = layers.Dropout(self.dropout_rate)(x)
        return x