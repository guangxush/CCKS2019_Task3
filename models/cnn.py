# -*- coding: utf-8 -*-
# from keras import backend as K
# from keras import initializers, regularizers, constraints, activations
# from keras.utils import conv_utils
# from keras.engine.topology import Layer
# from keras.engine.base_layer import InputSpec
#
# # Legacy support.
# from keras.legacy import interfaces
#
#
# class Att_Conv(Layer):
#
#     def __init__(self,
#                  filters,
#                  kernel_size,
#                  rank=1,
#                  strides=1,
#                  padding='valid',
#                  data_format=None,
#                  dilation_rate=1,
#                  activation=None,
#                  use_bias=True,
#                  kernel_initializer='glorot_uniform',
#                  bias_initializer='zeros',
#                  kernel_regularizer=None,
#                  bias_regularizer=None,
#                  activity_regularizer=None,
#                  kernel_constraint=None,
#                  bias_constraint=None,
#                  **kwargs):
#         super(Att_Conv, self).__init__(**kwargs)
#         self.rank = rank
#         self.filters = filters
#         self.kernel_size = conv_utils.normalize_tuple(kernel_size, rank, 'kernel_size')
#         self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
#         self.padding = conv_utils.normalize_padding(padding)
#         self.data_format = K.normalize_data_format(data_format)
#         self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, rank, 'dilation_rate')
#         self.activation = activations.get(activation)
#         self.use_bias = use_bias
#         self.kernel_initializer = initializers.get(kernel_initializer)
#         self.bias_initializer = initializers.get(bias_initializer)
#         self.kernel_regularizer = regularizers.get(kernel_regularizer)
#         self.bias_regularizer = regularizers.get(bias_regularizer)
#         self.activity_regularizer = regularizers.get(activity_regularizer)
#         self.kernel_constraint = constraints.get(kernel_constraint)
#         self.bias_constraint = constraints.get(bias_constraint)
#         # self.input_spec = InputSpec(ndim=self.rank + 2)
#
#     def build(self, input_shape):
#         att_shape = input_shape[1]
#         input_shape = input_shape[0]
#         if self.data_format == 'channels_first':
#             channel_axis = 1
#         else:
#             channel_axis = -1
#         if input_shape[channel_axis] is None:
#             raise ValueError('The channel dimension of the inputs '
#                              'should be defined. Found `None`.')
#         input_dim = input_shape[channel_axis]
#         kernel_shape = self.kernel_size + (input_dim, self.filters)
#
#         self.kernel = self.add_weight(shape=kernel_shape,
#                                       initializer=self.kernel_initializer,
#                                       name='kernel',
#                                       regularizer=self.kernel_regularizer,
#                                       constraint=self.kernel_constraint)
#         self.kernel_att = self.add_weight(shape=(att_shape[-1], self.filters),
#                                       initializer=self.kernel_initializer,
#                                       name='kernel_att',
#                                       regularizer=self.kernel_regularizer,
#                                       constraint=self.kernel_constraint)
#         if self.use_bias:
#             self.bias = self.add_weight(shape=(self.filters,),
#                                         initializer=self.bias_initializer,
#                                         name='bias',
#                                         regularizer=self.bias_regularizer,
#                                         constraint=self.bias_constraint)
#         else:
#             self.bias = None
#         # # Set input spec.
#         # self.input_spec = InputSpec(ndim=self.rank + 2,
#         #                             axes={channel_axis: input_dim})
#         self.built = True
#
#     def call(self, inputs):
#         attention = inputs[1]
#         inputs = inputs[0]
#         outputs = K.conv1d(
#             inputs,
#             self.kernel,
#             strides=self.strides[0],
#             padding=self.padding,
#             data_format=self.data_format,
#             dilation_rate=self.dilation_rate[0])
#
#         att = K.dot(attention, self.kernel_att)
#         outputs = outputs + att
#
#         if self.use_bias:
#             outputs = K.bias_add(
#                 outputs,
#                 self.bias,
#                 data_format=self.data_format)
#
#         if self.activation is not None:
#             return self.activation(outputs)
#         return outputs
#
#     def compute_output_shape(self, input_shape):
#         input_shape = input_shape[0]
#         if self.data_format == 'channels_last':
#             space = input_shape[1:-1]
#             new_space = []
#             for i in range(len(space)):
#                 new_dim = conv_utils.conv_output_length(
#                     space[i],
#                     self.kernel_size[i],
#                     padding=self.padding,
#                     stride=self.strides[i],
#                     dilation=self.dilation_rate[i])
#                 new_space.append(new_dim)
#             return (input_shape[0],) + tuple(new_space) + (self.filters,)
#         if self.data_format == 'channels_first':
#             space = input_shape[2:]
#             new_space = []
#             for i in range(len(space)):
#                 new_dim = conv_utils.conv_output_length(
#                     space[i],
#                     self.kernel_size[i],
#                     padding=self.padding,
#                     stride=self.strides[i],
#                     dilation=self.dilation_rate[i])
#                 new_space.append(new_dim)
#             return (input_shape[0], self.filters) + tuple(new_space)
#
#     def get_config(self):
#         config = {
#             'rank': self.rank,
#             'filters': self.filters,
#             'kernel_size': self.kernel_size,
#             'strides': self.strides,
#             'padding': self.padding,
#             'data_format': self.data_format,
#             'dilation_rate': self.dilation_rate,
#             'activation': activations.serialize(self.activation),
#             'use_bias': self.use_bias,
#             'kernel_initializer': initializers.serialize(self.kernel_initializer),
#             'bias_initializer': initializers.serialize(self.bias_initializer),
#             'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
#             'bias_regularizer': regularizers.serialize(self.bias_regularizer),
#             'activity_regularizer': regularizers.serialize(self.activity_regularizer),
#             'kernel_constraint': constraints.serialize(self.kernel_constraint),
#             'bias_constraint': constraints.serialize(self.bias_constraint)
#         }
#         base_config = super(Att_Conv, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))
#
#
# class AttConv1D(Att_Conv):
#
#     @interfaces.legacy_conv1d_support
#     def __init__(self, filters,
#                  kernel_size,
#                  strides=1,
#                  padding='valid',
#                  data_format='channels_last',
#                  dilation_rate=1,
#                  activation=None,
#                  use_bias=True,
#                  kernel_initializer='glorot_uniform',
#                  bias_initializer='zeros',
#                  kernel_regularizer=None,
#                  bias_regularizer=None,
#                  activity_regularizer=None,
#                  kernel_constraint=None,
#                  bias_constraint=None,
#                  **kwargs):
#         if padding == 'causal':
#             if data_format != 'channels_last':
#                 raise ValueError('When using causal padding in `Conv1D`, '
#                                  '`data_format` must be "channels_last" '
#                                  '(temporal data).')
#         super(AttConv1D, self).__init__(
#             filters=filters,
#             kernel_size=kernel_size,
#             strides=strides,
#             padding=padding,
#             data_format=data_format,
#             dilation_rate=dilation_rate,
#             activation=activation,
#             use_bias=use_bias,
#             kernel_initializer=kernel_initializer,
#             bias_initializer=bias_initializer,
#             kernel_regularizer=kernel_regularizer,
#             bias_regularizer=bias_regularizer,
#             activity_regularizer=activity_regularizer,
#             kernel_constraint=kernel_constraint,
#             bias_constraint=bias_constraint,
#             **kwargs)
#
#     def get_config(self):
#         config = super(AttConv1D, self).get_config()
#         config.pop('rank')
#         return config
#
#
# class Att_Matching_Conv(Layer):
#     def __init__(self,
#                  filters,
#                  kernel_size,
#                  rank=1,
#                  strides=1,
#                  padding='valid',
#                  data_format=None,
#                  dilation_rate=1,
#                  activation=None,
#                  use_bias=True,
#                  kernel_initializer='glorot_uniform',
#                  bias_initializer='zeros',
#                  kernel_regularizer=None,
#                  bias_regularizer=None,
#                  activity_regularizer=None,
#                  kernel_constraint=None,
#                  bias_constraint=None,
#                  **kwargs):
#         super(Att_Matching_Conv, self).__init__(**kwargs)
#         self.rank = rank
#         self.filters = filters
#         self.kernel_size = conv_utils.normalize_tuple(kernel_size, rank, 'kernel_size')
#         self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
#         self.padding = conv_utils.normalize_padding(padding)
#         self.data_format = K.normalize_data_format(data_format)
#         self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, rank, 'dilation_rate')
#         self.activation = activations.get(activation)
#         self.use_bias = use_bias
#         self.kernel_initializer = initializers.get(kernel_initializer)
#         self.bias_initializer = initializers.get(bias_initializer)
#         self.kernel_regularizer = regularizers.get(kernel_regularizer)
#         self.bias_regularizer = regularizers.get(bias_regularizer)
#         self.activity_regularizer = regularizers.get(activity_regularizer)
#         self.kernel_constraint = constraints.get(kernel_constraint)
#         self.bias_constraint = constraints.get(bias_constraint)
#         # self.input_spec = InputSpec(ndim=self.rank + 2)
#
#     def build(self, input_shape):
#         matching_shape = input_shape[-1]
#         att_shape = input_shape[1]
#         input_shape = input_shape[0]
#         if self.data_format == 'channels_first':
#             channel_axis = 1
#         else:
#             channel_axis = -1
#         if input_shape[channel_axis] is None:
#             raise ValueError('The channel dimension of the inputs '
#                              'should be defined. Found `None`.')
#         input_dim = input_shape[channel_axis]
#         kernel_shape = self.kernel_size + (input_dim, self.filters)
#
#         self.kernel = self.add_weight(shape=kernel_shape,
#                                       initializer=self.kernel_initializer,
#                                       name='kernel',
#                                       regularizer=self.kernel_regularizer,
#                                       constraint=self.kernel_constraint)
#         self.kernel_att = self.add_weight(shape=(att_shape[-1], self.filters),
#                                           initializer=self.kernel_initializer,
#                                           name='kernel_att',
#                                           regularizer=self.kernel_regularizer,
#                                           constraint=self.kernel_constraint)
#         self.kernel_matching = self.add_weight(shape=(matching_shape[-1], self.filters),
#                                           initializer=self.kernel_initializer,
#                                           name='kernel_matching',
#                                           regularizer=self.kernel_regularizer,
#                                           constraint=self.kernel_constraint)
#
#         if self.use_bias:
#             self.bias = self.add_weight(shape=(self.filters,),
#                                         initializer=self.bias_initializer,
#                                         name='bias',
#                                         regularizer=self.bias_regularizer,
#                                         constraint=self.bias_constraint)
#         else:
#             self.bias = None
#         # # Set input spec.
#         # self.input_spec = InputSpec(ndim=self.rank + 2,
#         #                             axes={channel_axis: input_dim})
#         self.built = True
#
#     def call(self, inputs):
#         matching = inputs[-1]
#         attention = inputs[1]
#         inputs = inputs[0]
#         outputs = K.conv1d(
#             inputs,
#             self.kernel,
#             strides=self.strides[0],
#             padding=self.padding,
#             data_format=self.data_format,
#             dilation_rate=self.dilation_rate[0])
#
#         att = K.dot(attention, self.kernel_att)
#         matching = K.dot(matching, self.kernel_matching)
#         outputs = outputs + att + matching
#
#         if self.use_bias:
#             outputs = K.bias_add(
#                 outputs,
#                 self.bias,
#                 data_format=self.data_format)
#
#         if self.activation is not None:
#             return self.activation(outputs)
#         return outputs
#
#     def compute_output_shape(self, input_shape):
#         input_shape = input_shape[0]
#         if self.data_format == 'channels_last':
#             space = input_shape[1:-1]
#             new_space = []
#             for i in range(len(space)):
#                 new_dim = conv_utils.conv_output_length(
#                     space[i],
#                     self.kernel_size[i],
#                     padding=self.padding,
#                     stride=self.strides[i],
#                     dilation=self.dilation_rate[i])
#                 new_space.append(new_dim)
#             return (input_shape[0],) + tuple(new_space) + (self.filters,)
#         if self.data_format == 'channels_first':
#             space = input_shape[2:]
#             new_space = []
#             for i in range(len(space)):
#                 new_dim = conv_utils.conv_output_length(
#                     space[i],
#                     self.kernel_size[i],
#                     padding=self.padding,
#                     stride=self.strides[i],
#                     dilation=self.dilation_rate[i])
#                 new_space.append(new_dim)
#             return (input_shape[0], self.filters) + tuple(new_space)
#
#     def get_config(self):
#         config = {
#             'rank': self.rank,
#             'filters': self.filters,
#             'kernel_size': self.kernel_size,
#             'strides': self.strides,
#             'padding': self.padding,
#             'data_format': self.data_format,
#             'dilation_rate': self.dilation_rate,
#             'activation': activations.serialize(self.activation),
#             'use_bias': self.use_bias,
#             'kernel_initializer': initializers.serialize(self.kernel_initializer),
#             'bias_initializer': initializers.serialize(self.bias_initializer),
#             'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
#             'bias_regularizer': regularizers.serialize(self.bias_regularizer),
#             'activity_regularizer': regularizers.serialize(self.activity_regularizer),
#             'kernel_constraint': constraints.serialize(self.kernel_constraint),
#             'bias_constraint': constraints.serialize(self.bias_constraint)
#         }
#         base_config = super(Att_Matching_Conv, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))
#
# class AttMatchingConv1D(Att_Matching_Conv):
#
#     @interfaces.legacy_conv1d_support
#     def __init__(self, filters,
#                  kernel_size,
#                  strides=1,
#                  padding='valid',
#                  data_format='channels_last',
#                  dilation_rate=1,
#                  activation=None,
#                  use_bias=True,
#                  kernel_initializer='glorot_uniform',
#                  bias_initializer='zeros',
#                  kernel_regularizer=None,
#                  bias_regularizer=None,
#                  activity_regularizer=None,
#                  kernel_constraint=None,
#                  bias_constraint=None,
#                  **kwargs):
#         if padding == 'causal':
#             if data_format != 'channels_last':
#                 raise ValueError('When using causal padding in `Conv1D`, '
#                                  '`data_format` must be "channels_last" '
#                                  '(temporal data).')
#         super(AttMatchingConv1D, self).__init__(
#             filters=filters,
#             kernel_size=kernel_size,
#             strides=strides,
#             padding=padding,
#             data_format=data_format,
#             dilation_rate=dilation_rate,
#             activation=activation,
#             use_bias=use_bias,
#             kernel_initializer=kernel_initializer,
#             bias_initializer=bias_initializer,
#             kernel_regularizer=kernel_regularizer,
#             bias_regularizer=bias_regularizer,
#             activity_regularizer=activity_regularizer,
#             kernel_constraint=kernel_constraint,
#             bias_constraint=bias_constraint,
#             **kwargs)
#
#     def get_config(self):
#         config = super(AttMatchingConv1D, self).get_config()
#         config.pop('rank')
#         return config
#
