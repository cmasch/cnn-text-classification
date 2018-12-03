# -*- coding: utf-8 -*-
"""
CNN model for text classification
This implementation is based on the original paper of Yoon Kim [1].

# References
- [1] [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)

@author: Christopher Masch
"""

from keras.layers import Activation, Input, Dense, Dropout, Embedding
from keras.layers.convolutional import SeparableConv1D
from keras.layers import GlobalMaxPooling1D
from keras.layers.merge import concatenate
from keras.models import Model
from keras import initializers
from keras import backend as K

class CNN:
    
    __version__ = '0.0.2'
    
    def __init__(self, embedding_layer=None, num_words=None, embedding_dim=None,
                 max_seq_length=100, filter_sizes=[3,4,5], feature_maps=[100,100,100],
                 hidden_units=100, dropout_rate=None, nb_classes=None):
        """
        Arguments:
            embedding_layer : If not defined with pre-trained embeddings it will be created from scratch (default: None)
            num_words       : Maximal amount of words in the vocabulary (default: None)
            embedding_dim   : Dimension of word representation (default: None)
            max_seq_length  : Max length of sequence (default: 100)
            filter_sizes    : An array of filter sizes per channel (default: [3,4,5])
            feature_maps    : Defines the feature maps per channel (default: [100,100,100])
            hidden_units    : Hidden units per convolution channel (default: 100)
            dropout_rate    : If defined, dropout will be added after embedding layer & concatenation (default: None)
            nb_classes      : Number of classes which can be predicted
        """
        self.embedding_layer = embedding_layer
        self.num_words       = num_words
        self.max_seq_length  = max_seq_length
        self.embedding_dim   = embedding_dim
        self.filter_sizes    = filter_sizes
        self.feature_maps    = feature_maps
        self.hidden_units    = hidden_units
        self.dropout_rate    = dropout_rate
        self.nb_classes      = nb_classes
        
    def build_model(self):
        """
        Build the model
        
        Returns:
            Model           : Keras model instance
        """

        # Checks
        if len(self.filter_sizes)!=len(self.feature_maps):
            raise Exception('Please define `filter_sizes` and `feature_maps` with the same length.')
        if not self.embedding_layer and (not self.num_words or not self.embedding_dim):
            raise Exception('Please define `num_words` and `embedding_dim` if you not use a pre-trained embeddings')
        
        
        # Building embeddings from scratch
        if self.embedding_layer is None:
            self.embedding_layer = Embedding(
                input_dim=self.num_words, 
                output_dim=self.embedding_dim,       
                input_length=self.max_seq_length,
                weights=None, trainable=True,
                name="word_embedding"
            )
        
        word_input = Input(shape=(self.max_seq_length,), dtype='int32', name='word_input')
        x = self.embedding_layer(word_input)
        x = Dropout(self.dropout_rate)(x)
        x = self.building_block(x, self.filter_sizes, self.feature_maps)
        x = Activation('relu')(x)
        prediction = Dense(self.nb_classes, activation='softmax')(x)
        return Model(inputs=word_input, outputs=prediction)
    
    
    def building_block(self, input_layer, filter_sizes, feature_maps):
        """ 
        Creates several CNN channels in parallel and concatenate them 
        
        Arguments:
            input_layer : Layer which will be the input for all convolutional blocks
            filter_sizes: Array of filter sizes
            feature_maps: Array of feature maps
            
        Returns:
            x           : Building block with one or several channels
        """
        channels = []
        for ix in range(len(self.filter_sizes)):
            x = self.create_channel(input_layer, filter_sizes[ix], feature_maps[ix])
            channels.append(x)
            
        # Checks how many channels, one channel doesn't need a concatenation
        if (len(channels)>1):
            x = concatenate(channels)
        return x
    
    
    def create_channel(self, x, filter_size, feature_map):
        """
        Creates a layer, working channel wise
        
        Arguments:
            x           : Input for convoltuional channel
            filter_size : Filter size for creating Conv1D
            feature_map : Feature map 
            
        Returns:
            x           : Channel including (Conv1D + GlobalMaxPooling + Dense + Dropout)
        """
        x = SeparableConv1D(feature_map, kernel_size=filter_size, activation='relu', strides=1, padding='same',
                            depth_multiplier=4)(x)
        x = GlobalMaxPooling1D()(x)
        x = Dense(self.hidden_units)(x)
        x = Dropout(self.dropout_rate)(x)
        return x