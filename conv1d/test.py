from keras import backend as K
from keras.layers import Layer
from keras.models import Sequential
from keras.layers import Input
from keras.layers import Dense
import tensorflow as tf
import keras.backend as K

# import keras.backend.cntk as C
from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.engine.base_layer import Layer
from keras.engine.base_layer import InputSpec
from keras.utils import conv_utils
from keras.utils.generic_utils import transpose_shape
from keras.legacy import interfaces
from keras.engine.base_layer import InputSpec

# def _get_dynamic_axis_num(x):
#     if hasattr(x, 'dynamic_axes'):
#         return len(x.dynamic_axes)
#     else:
#         return 0
# def temporal_padding(x, padding=(1, 1)):
#     assert len(padding) == 2
#     num_dynamic_axis = _get_dynamic_axis_num(x)
#     assert len(x.shape) == 3 - (1 if num_dynamic_axis > 0 else 0)
#     return pad(x, [padding], 'channels_last', num_dynamic_axis)
# def normalize_data_format(value):
#     """Checks that the value correspond to a valid data format.

#     # Arguments
#         value: String or None. `'channels_first'` or `'channels_last'`.

#     # Returns
#         A string, either `'channels_first'` or `'channels_last'`

#     # Example
#     ```python
#         >>> from keras import backend as K
#         >>> K.normalize_data_format(None)
#         'channels_first'
#         >>> K.normalize_data_format('channels_last')
#         'channels_last'
#     ```

#     # Raises
#         ValueError: if `value` or the global `data_format` invalid.
#     """
#     if value is None:
#         value = 'channels_last'
#     data_format = value.lower()
#     if data_format not in {'channels_first', 'channels_last'}:
#         raise ValueError('The `data_format` argument must be one of '
#                          '"channels_first", "channels_last". Received: ' +
#                          str(value))
#     return data_format

# def conv1d(x, kernel, strides=1, padding='valid',
#            data_format=None, dilation_rate=1):
#     data_format = normalize_data_format(data_format)

#     if padding == 'causal':
#         # causal (dilated) convolution:
#         left_pad = dilation_rate * (kernel.shape[0] - 1)
#         x = temporal_padding(x, (left_pad, 0))
#         padding = 'valid'

#     if data_format == 'channels_last':
#         x = C.swapaxes(x, 0, 1)

#     # As of Keras 2.0.0, all kernels are normalized
#     # on the format `(steps, input_depth, depth)`,
#     # independently of `data_format`.
#     # CNTK expects `(depth, input_depth, steps)`.
#     kernel = C.swapaxes(kernel, 0, 2)

#     padding = _preprocess_border_mode(padding)

#     if dev.type() == 0 and dilation_rate != 1:
#         raise ValueError(
#             'Dilated convolution on CPU is not supported by CNTK backend. '
#             'Please set `dilation_rate` to 1. You passed: %s' % (dilation_rate,))

#     dilation_rate = (1, dilation_rate)

#     x = C.convolution(
#         kernel,
#         x,
#         strides=strides,
#         auto_padding=[False, padding],
#         dilation=dilation_rate)

#     if data_format == 'channels_last':
#         x = C.swapaxes(x, 0, 1)
#     return x

class MyLayer(Layer):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=1,
                 padding='valid',
                 data_format=None,
                 dilation_rate=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(MyLayer, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, 1,
                                                      'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, 1, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = K.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, 1,
                                                        'dilation_rate')
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(ndim=1 + 2)

    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.filters)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        # Set input spec.
        self.input_spec = InputSpec(ndim=1 + 2,
                                    axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs):
        outputs = K.conv1d(
            inputs,
            self.kernel,
            strides=self.strides[0],
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate[0])


        if self.use_bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_last':
            space = input_shape[1:-1]
        elif self.data_format == 'channels_first':
            space = input_shape[2:]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i],
                self.kernel_size[i],
                padding=self.padding,
                stride=self.strides[i],
                dilation=self.dilation_rate[i])
            new_space.append(new_dim)
        if self.data_format == 'channels_last':
            return (input_shape[0],) + tuple(new_space) + (self.filters,)
        elif self.data_format == 'channels_first':
            return (input_shape[0], self.filters) + tuple(new_space)
    
# model = Sequential()
# model.add(Dense(32, input_dim=32))
# model.add(MyLayer(100))
# model.summary()