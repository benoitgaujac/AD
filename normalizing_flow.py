import numpy as np
import tensorflow.compat.v1 as tf

# import tensorflow as tf
# import tensorflow_probability as tfp
# from tensorflow.keras.layers import Layer, Dense, BatchNormalization, ReLU, Conv2D, Reshape
# from tensorflow.keras import Model
#
# from .case import Case
#
# from utils.train_utils import checkerboard
# tfd = tfp.distributions
# tfb = tfp.bijectors
# tfk = tf.keras
#
# tf.keras.backend.set_floatx('float32')
#
# print('tensorflow: ', tf.__version__)
# print('tensorflow-probability: ', tfp.__version__)

import ops._ops as _ops
import networks
import ops

import pdb

'''--------------------------------------------- mlp flow --------------------------------------------------------'''

def mlp(opts, inputs, scope=None, reuse=False):
    outputs = networks.mlp(opts, inputs, output_dim=2,
                            nlayers=opts['mlpflow_nlayers'],
                            init=opts['mlpflow_init'],
                            stddev=opts['mlpflow_init_std'],
                            bias=opts['mlpflow_init_bias'],
                            nonlinear=opts['mlpflow_nonlinear'],
                            eta1=opts['mlpflow_eta1'],
                            eta2=opts['mlpflow_eta2'],
                            scope=scope, reuse=reuse)

    return outputs


'''--------------------------------------------- Planar Flow --------------------------------------------------------'''

def planar(opts, inputs, scope=None, reuse=False):
    '''
    Implementation of Planar Flow.
    '''

    def m(x):
        return -1 + tf.math.softplus(x)

    # init flow parameters
    with tf.variable_scope(scope, reuse=reuse):
        u = tf.get_variable('u', [2,], tf.float32, tf.random_uniform_initializer(minval=-1., maxval=1))
        w = tf.get_variable('w', [2,], tf.float32, tf.random_uniform_initializer(minval=-1., maxval=1))
        b = tf.get_variable('b', [1,], tf.float32, tf.random_uniform_initializer(minval=-1., maxval=1))

    # follow Appendix A.1. of Jimenez Rezende, D. and Mohamed, S.
    wu = tf.linalg.matmul(tf.expand_dims(w,0), tf.expand_dims(u,-1)) #[1,1]
    wu = tf.reshape(wu, [1,]) #[1,1]
    w_norm = w / tf.reduce_sum(tf.square(w))
    u_hat = u + (m(wu) - wu) * w_norm #[2,]
    u_hat = tf.reshape(u_hat, [1,2,1]) #[1,2,1]
    # flow
    outputs = tf.linalg.matmul(tf.reshape(w, [1,1,-1]), tf.expand_dims(inputs, -1)) + tf.reshape(b, [1,1,-1]) #[batch, 1,1]
    outputs = tf.expand_dims(inputs, axis=-1) + u_hat * _ops.non_linear(outputs, 'tanh') #[batch, 2,1]

    return tf.reshape(outputs, [-1,2])


# '''--------------------------------------------- Radial Flow --------------------------------------------------------'''
#
# class RadialFlow(tfb.Bijector, tf.Module):
#     '''
#     To-do: Check implementation.
#     '''
#
#     def __init__(self, validate_args=False, event_ndims=0, name='radial'):
#         super(RadialFlow, self).__init__(
#           validate_args=validate_args,
#           forward_min_event_ndims=event_ndims,
#           name=name)
#         self.event_ndims = event_ndims
#         self.x0 = tf.Variable(np.random.uniform(0., 1., size=(1, self.event_ndims)), name='u', dtype=tf.float32)
#         self.alpha = tf.Variable(0.0, dtype=tf.float32)
#         self.beta = tf.Variable(0.0, dtype=tf.float32)
#
#     def _forward(self, x):
#         """
#         Given x, returns z and the log-determinant log|df/dx|.
#         """
#         r = tf.norm(x - self.x0)
#         h = 1/(tf.nn.relu(self.alpha) + r)
#         return x + self.beta*h*(x-self.x0)
#
#     def _inverse(self, y):
#         raise NotImplementedError('missing implementation of _inverse')
#
#     def _forward_log_det_jacobian(self, z):
#         raise NotImplementedError('missing implementation of _inverse')
#
#     def _inverse_log_det_jacobian(self, y):
#         raise NotImplementedError('missing implementation of _inverse_log_det_jacobian')
#         #return -self._forward_log_det_jacobian(self.inverse(y))
#
#
# '''--------------------------------------------- Real NVP -----------------------------------------------'''
#
# class NN(Layer):
#     """
#     Neural Network Architecture for calcualting s and t for Real-NVP
#
#     :param input_shape: shape of the data coming in the layer
#     :param hidden_units: Python list-like of non-negative integers, specifying the number of units in each hidden layer.
#     :param activation: Activation of the hidden units
#     """
#     def __init__(self, input_shape, n_hidden=[512, 512], activation="relu", name="nn"):
#         super(NN, self).__init__(name="nn")
#         layer_list = []
#         for i, hidden in enumerate(n_hidden):
#             layer_list.append(Dense(hidden, activation=activation))
#         self.layer_list = layer_list
#         self.log_s_layer = Dense(input_shape, activation="tanh", name='log_s')
#         self.t_layer = Dense(input_shape, name='t')
#
#     def call(self, x):
#         y = x
#         for layer in self.layer_list:
#             y = layer(y)
#         log_s = self.log_s_layer(y)
#         t = self.t_layer(y)
#         return log_s, t
#
# class RealNVP(tfb.Bijector):
#     """
#     Implementation of a Real-NVP for Denisty Estimation. L. Dinh “Density estimation using Real NVP,” 2016.
#     This implementation only works for 1D arrays.
#     :param input_shape: shape of the data coming in the layer
#     :param hidden_units: Python list-like of non-negative integers, specifying the number of units in each hidden layer.
#     """
#
#     def __init__(self, input_shape, n_hidden=[512, 512], forward_min_event_ndims=1, validate_args: bool = False, name="real_nvp"):
#         super(RealNVP, self).__init__(
#             validate_args=validate_args, forward_min_event_ndims=forward_min_event_ndims, name=name
#         )
#
#         assert input_shape % 2 == 0
#         input_shape = input_shape // 2
#         nn_layer = NN(input_shape, n_hidden)
#         x = tf.keras.Input(input_shape)
#         log_s, t = nn_layer(x)
#         self.nn = Model(x, [log_s, t], name="nn")
#
#     def _bijector_fn(self, x):
#         log_s, t = self.nn(x)
#         return tfb.affine_scalar.AffineScalar(shift=t, log_scale=log_s)
#
#     def _forward(self, x):
#         x_a, x_b = tf.split(x, 2, axis=-1)
#         y_b = x_b
#         y_a = self._bijector_fn(x_b).forward(x_a)
#         y = tf.concat([y_a, y_b], axis=-1)
#         return y
#
#     def _inverse(self, y):
#         y_a, y_b = tf.split(y, 2, axis=-1)
#         x_b = y_b
#         x_a = self._bijector_fn(y_b).inverse(y_a)
#         x = tf.concat([x_a, x_b], axis=-1)
#         return x
#
#     def _forward_log_det_jacobian(self, x):
#         x_a, x_b = tf.split(x, 2, axis=-1)
#         return self._bijector_fn(x_b).forward_log_det_jacobian(x_a, event_ndims=1)
#
#     def _inverse_log_det_jacobian(self, y):
#         y_a, y_b = tf.split(y, 2, axis=-1)
#         return self._bijector_fn(y_b).inverse_log_det_jacobian(y_a, event_ndims=1)
#
#
# '''--------------------------------------------- Neural Spline Flow --------------------------------------------------------'''
#
# class NN_Spline(Layer):
#     def __init__(self, layers, k_dim, remaining_dims, first_d_dims, activation="relu"):
#         super(NN_Spline, self).__init__(name="nn")
#         self.k_dim = k_dim
#         layer_list = []
#         layer_list.append(Dense(layers[0], activation=activation, input_dim=first_d_dims, dtype=tf.float32, name=f'0_layer'))
#         for i, hidden in enumerate(layers[1:]):
#             layer_list.append(Dense(hidden, activation=activation, dtype=tf.float32, name=f'{i+1}_layer'))
#         layer_list.append(Dense(remaining_dims*(3*k_dim-1), dtype=tf.float32, name='last_layer'))
#         self.layer_list = layer_list
#
#     def call(self, x):
#         y = x
#         for layer in self.layer_list:
#             y = layer(y)
#         return y
#
# class NeuralSplineFlow(tfb.Bijector):
#     """
#     Implementation of a Neural Spline Flows by Durkan et al. [1].
#     :param n_dims: The dimension of the vector-sized input. Each individual input should be a vector with d_dim dimensions.
#     :param number_of_bins: Number of bins to create the spline
#     :param nn_layers: Python list-like of non-negative integers, specifying the number of units in each hidden layer.
#     :param b_interval: Interval to define the spline function. Spline function behaves as identity outside of the interval
#     :param d_dim: The number of dimensions to create the parameters of the spline. (d_dim-1) dims are used to create the parameters as in paper.
#     :param simetric_interval: If this is true we have a interval of [-b_interval, b_interval]. [0, 2*b_interval] if false.
#     """
#
#     def __init__(self,input_dim, d_dim, b_interval, number_of_bins=5, nn_layers = [16, 16], n_dims=1, simetric_interval: bool = True, validate_args: bool = False, name="neural_spline_flow"):
#         super(NeuralSplineFlow, self).__init__(
#             validate_args=validate_args, forward_min_event_ndims=1, name=name
#         )
#
#         self.event_ndims = 1
#         self.total_input_dim = input_dim
#         self.first_d_dims = d_dim-1
#         self.remaining_dims = input_dim-self.first_d_dims
#         self.number_of_bins = number_of_bins
#         self.number_of_knots= number_of_bins+1
#         self.b_interval = tf.constant(b_interval, dtype=tf.float32)
#         self.nn = NN_Spline(layers= nn_layers, k_dim = self.number_of_bins, first_d_dims= self.first_d_dims, remaining_dims= self.remaining_dims)
#         x = tf.keras.Input(self.first_d_dims, dtype=tf.float32)
#         output = self.nn(x)
#         self.min_bin_width = 1e-3 #maximum number of bins 1/1e-3 then...
#         self.nn_model = Model(x, output, name="nn")
#         self.simetric_interval = simetric_interval
#
#     # some calculation could be done in one-line of code but it was preferred to explicitly write them
#     # for easy debugging purposes during the development and also to give an understanding of the implementations of the terms in the paper
#     # to the reader
#     def _produce_thetas(self, x):
#         thetas = self.nn_model(x)
#         thetas = tf.reshape(thetas, [tf.shape(x)[0], self.remaining_dims, 3*self.number_of_bins-1])
#         return thetas
#
#     def _get_thetas(self, thetas, input_mask_indexes):
#         thetas_for_input = tf.gather_nd(thetas, input_mask_indexes)
#         thetas_1 = thetas_for_input[:, :self.number_of_bins]
#         thetas_2 = thetas_for_input[:, self.number_of_bins:2*self.number_of_bins]
#         thetas_3 = thetas_for_input[:, 2*self.number_of_bins:]
#         return thetas_1, thetas_2, thetas_3
#
#
#     def _bins(self, thetas, intervals):
#         normalized_widths = tf.math.softmax(thetas)
#         normalized_widths_filled = self.min_bin_width + (1 - self.min_bin_width * self.number_of_bins) * normalized_widths
#         expanded_widths = normalized_widths_filled * 2 * tf.expand_dims(intervals,1)
#         return expanded_widths
#
#     def _knots(self, bins, intervals):
#         interval = -1 * tf.expand_dims(intervals,1)
#         b = tf.concat([tf.zeros((tf.shape(bins)[0],1), dtype=tf.float32),  tf.dtypes.cast((tf.math.cumsum(bins, axis=1)),tf.float32)], 1) + tf.dtypes.cast(interval,tf.float32) if self.simetric_interval else tf.concat([tf.zeros((tf.shape(bins)[0],1), dtype=tf.float32),  tf.dtypes.cast((tf.math.cumsum(bins, axis=1)),tf.float32)], 1)
#         return b
#
#     def _derivatives(self, thetas):
#         inner_derivatives = tf.math.softplus(thetas)
#         c = tf.concat([tf.ones((tf.shape(inner_derivatives)[0],1), dtype=tf.float32), inner_derivatives, tf.ones((tf.shape(inner_derivatives)[0],1),dtype=tf.float32)], 1)
#         return c + self.min_bin_width
#
#     def _s_values(self, y_bins, x_bins):
#         y = y_bins / x_bins
#         return y
#
#     def _knots_locations(self, x, knot_xs):
#         x_binary_mask = tf.cast((tf.expand_dims(x,1) > knot_xs), tf.int32)
#         knot_xs = tf.reduce_sum(x_binary_mask, axis=1)
#         return knot_xs
#
#     def _indices(self, locations):
#         row_indices = tf.range(tf.shape(locations)[0], dtype=tf.int32)
#         z = tf.transpose(tf.stack([row_indices, locations]))
#         return z
#
#     def _xi_values(self, x , knot_xs, x_bin_sizes, ind):
#         f = (tf.transpose(x) - tf.gather_nd(knot_xs, ind)) / tf.gather_nd(x_bin_sizes, ind)
#         return f
#
#     def _g_function(self, x, bin_ind, knot_ind, xi_values, s_values, y_bin_sizes, derivatives, knot_ys):
#         xi_times_1_minus_xi = xi_values * (1 - xi_values)
#         s_k = tf.gather_nd(s_values, bin_ind)
#         y_kplus1_minus_y_k = tf.gather_nd(y_bin_sizes, bin_ind)
#         xi_square = xi_values**2
#         d_k = tf.gather_nd(derivatives, bin_ind)
#         d_kplus1 = tf.gather_nd(derivatives, knot_ind)
#         y_k = tf.gather_nd(knot_ys,bin_ind)
#         second_term_nominator = y_kplus1_minus_y_k * (s_k * xi_square + d_k * xi_times_1_minus_xi)
#         second_term_denominator =  s_k + (d_kplus1 + d_k - 2*s_k) * xi_times_1_minus_xi
#         forward_val = y_k + second_term_nominator / second_term_denominator
#         return forward_val
#
#     def _inverse_g_function(self, input_for_inverse, floor_indices, ceil_indices, s_values, y_bin_sizes, derivatives, knot_ys, knot_xs, x_bin_sizes):
#         y_minus_y_k = tf.dtypes.cast(tf.transpose(input_for_inverse), tf.float32) - tf.dtypes.cast(tf.gather_nd(knot_ys, floor_indices), tf.float32)
#         s_k = tf.gather_nd(s_values,floor_indices)
#         y_kplus1_minus_y_k = tf.gather_nd(y_bin_sizes, floor_indices)
#         d_k = tf.gather_nd(derivatives,floor_indices)
#         d_kplus1 = tf.gather_nd(derivatives,ceil_indices)
#         common_term = y_minus_y_k*(d_kplus1 + d_k - 2*s_k)
#         a = y_kplus1_minus_y_k * (s_k - d_k) + common_term
#         b = y_kplus1_minus_y_k * d_k - common_term
#         c = -1 * s_k * y_minus_y_k
#         b_squared_minus_4ac = b**2 - 4 * a * c
#         sqrt_b_squared_minus_4ac = tf.math.sqrt(b_squared_minus_4ac)
#         denominator = (-1 * b - sqrt_b_squared_minus_4ac)
#         xi_x_d_to_D = 2 * c / denominator
#         x_d_to_D = xi_x_d_to_D * tf.gather_nd(x_bin_sizes, floor_indices) + tf.gather_nd(knot_xs, floor_indices)
#         return x_d_to_D
#
#     def _derivative_of_g_func(self, x, floor_indices, ceil_indices, xi_values, s_values, derivatives):
#         one_minus_xi = (1 - xi_values)
#         xi_times_1_minus_xi = xi_values * one_minus_xi
#         s_k = tf.gather_nd(s_values, floor_indices)
#         one_minus_xi_square = one_minus_xi**2
#         d_k = tf.gather_nd(derivatives, floor_indices)
#         d_kplus1 = tf.gather_nd(derivatives, ceil_indices)
#         nominator = s_k**2 * (d_kplus1*(xi_values**2) + 2*s_k*xi_times_1_minus_xi + d_k*one_minus_xi_square)
#         denominator = (s_k + (d_kplus1 + d_k - 2*s_k)*xi_times_1_minus_xi)**2
#         derivative_result = nominator/denominator
#         return derivative_result
#
#
#     def _data_mask(self, x_d_to_D, interval):
#         less_than_right_limit_mask = tf.less(x_d_to_D, interval)
#         bigger_than_left_limit_mask = tf.greater(x_d_to_D, -1.0 * interval)
#         input_mask = less_than_right_limit_mask & bigger_than_left_limit_mask
#         return input_mask
#
#     def _forward(self, x):
#         x_1_to_d, x_d_to_D = x[:,:self.first_d_dims], x[:,self.first_d_dims:]
#         x_d_to_D = tf.constant(x_d_to_D, dtype=tf.float32)
#         x_1_to_d = tf.constant(x_1_to_d, dtype=tf.float32)
#         _, intervals_for_func = self.b_interval[:self.first_d_dims], self.b_interval[self.first_d_dims:]
#         y_1_to_d = x_1_to_d
#         input_mask = self._data_mask(x_d_to_D, intervals_for_func)
#
#         def return_identity(): return x
#
#         def return_result():
#             output = tf.zeros(tf.shape(x_d_to_D))
#             input_mask_indexes = tf.where(input_mask)
#             neg_input_mask_indexes = tf.where(~input_mask)
#             thetas = self._produce_thetas(x_1_to_d)
#             thetas_1, thetas_2, thetas_3 = self._get_thetas(thetas, input_mask_indexes)
#             interval_indices = input_mask_indexes[:,1]
#
#             input_for_spline = x_d_to_D[input_mask]
#             intervals_for_input = tf.gather(intervals_for_func, interval_indices)
#             x_bin_sizes = self._bins(thetas_1,intervals_for_input)
#             knot_xs = self._knots(x_bin_sizes, intervals_for_input)
#             y_bin_sizes = self._bins(thetas_2,intervals_for_input)
#             knot_ys = self._knots(y_bin_sizes, intervals_for_input)
#             derivatives = self._derivatives(thetas_3)
#             locs = self._knots_locations(input_for_spline, knot_xs)
#             floor_indices = self._indices(locs-1)
#             ceil_indices = self._indices(locs)
#             xi_values = self._xi_values(input_for_spline , knot_xs, x_bin_sizes, floor_indices)
#             s_values = self._s_values(y_bin_sizes, x_bin_sizes)
#             forward_val = self._g_function(input_for_spline, floor_indices, ceil_indices, xi_values, s_values, y_bin_sizes, derivatives, knot_ys)
#             output = tf.tensor_scatter_nd_update(tf.dtypes.cast(tf.expand_dims(output,2), dtype=tf.float32), input_mask_indexes, tf.expand_dims(tf.dtypes.cast(tf.transpose(forward_val),dtype=tf.float32), 1))
#             output = tf.tensor_scatter_nd_update(output, neg_input_mask_indexes, tf.expand_dims(x_d_to_D[~input_mask],1))
#             return output
#
#         #these conditions are used in order to be able to use tf.function however
#         #it didn't work with tf.function.
#         r = tf.cond(tf.equal(tf.reduce_any(input_mask), tf.constant(False)), return_identity, return_result)
#         y = tf.concat([y_1_to_d, tf.squeeze(r,-1)], axis=-1)
#         return y
#
#     def _inverse(self, y):
#         y_1_to_d, y_d_to_D = y[:,:self.first_d_dims], y[:,self.first_d_dims:]
#         _, intervals_for_func = self.b_interval[:self.first_d_dims], self.b_interval[self.first_d_dims:]
#         x_1_to_d = y_1_to_d
#         input_mask = self._data_mask(y_d_to_D, intervals_for_func)
#
#         def return_identity():
#             return y
#
#         def return_result():
#             output = tf.zeros(tf.shape(y_d_to_D), dtype=tf.float32)
#             input_mask_indexes = tf.where(input_mask)
#             neg_input_mask_indexes = tf.where(~input_mask)
#             thetas = self._produce_thetas(y_1_to_d)
#             thetas_1, thetas_2, thetas_3 = self._get_thetas(thetas, input_mask_indexes)
#             input_for_inverse = y_d_to_D[input_mask]
#             interval_indices = input_mask_indexes[:,1]
#
#             intervals_for_input = tf.gather(intervals_for_func, interval_indices)
#             x_bin_sizes = self._bins(thetas_1, intervals_for_input)
#             knot_xs = self._knots(x_bin_sizes, intervals_for_input)
#             y_bin_sizes = self._bins(thetas_2, intervals_for_input)
#             knot_ys = self._knots(y_bin_sizes, intervals_for_input)
#             derivatives = self._derivatives(thetas_3)
#             locs = self._knots_locations(input_for_inverse, knot_ys)
#             floor_indices = self._indices(locs-1)
#             ceil_indices = self._indices(locs)
#             s_values = self._s_values(y_bin_sizes, x_bin_sizes)
#
#             inverse_val = self._inverse_g_function(input_for_inverse, floor_indices, ceil_indices, s_values, y_bin_sizes, derivatives, knot_ys, knot_xs, x_bin_sizes)
#             output = tf.tensor_scatter_nd_update(tf.dtypes.cast(tf.expand_dims(output,2), dtype=tf.float32), input_mask_indexes, tf.expand_dims(tf.dtypes.cast(tf.transpose(inverse_val), dtype=tf.float32),1))
#             output = tf.tensor_scatter_nd_update(tf.dtypes.cast(output, dtype=tf.float32), neg_input_mask_indexes, tf.dtypes.cast(tf.expand_dims(y_d_to_D[~input_mask],1), tf.float32))
#             return tf.concat([tf.dtypes.cast(y_1_to_d, tf.float32), tf.dtypes.cast(tf.squeeze(output,-1), tf.float32)], axis=-1)
#
#         return tf.cond(tf.equal(tf.reduce_any(input_mask), tf.constant(False)), return_identity, return_result)
#
#
#     def _forward_log_det_jacobian(self, x, thetas=None):
#         x_1_to_d, x_d_to_D = x[:,:self.first_d_dims], x[:,self.first_d_dims:]
#         _, intervals_for_func = self.b_interval[:self.first_d_dims], self.b_interval[self.first_d_dims:]
#         input_mask = self._data_mask(x_d_to_D, intervals_for_func)
#         def return_identity_log_det(): return tf.constant(0.0, dtype=tf.float32)
#
#         def return_result_log_det():
#             input_mask_indexes = tf.where(input_mask)
#             neg_input_mask_indexes = tf.where(~input_mask)
#             thetas =  self._produce_thetas(x_1_to_d)
#             thetas_1, thetas_2, thetas_3 = self._get_thetas(thetas, input_mask_indexes)
#             interval_indices = input_mask_indexes[:,1]
#             intervals_for_input = tf.gather(intervals_for_func, interval_indices)
#             input_for_derivative = x_d_to_D[input_mask]
#             x_bin_sizes = self._bins(thetas_1, intervals_for_input)
#             knot_xs = self._knots(x_bin_sizes, intervals_for_input)
#             y_bin_sizes = self._bins(thetas_2, intervals_for_input)
#             knot_ys = self._knots(y_bin_sizes, intervals_for_input)
#             derivatives = self._derivatives(thetas_3)
#             locs = self._knots_locations(input_for_derivative, knot_xs)
#             floor_indices = self._indices(locs-1)
#             ceil_indices = self._indices(locs)
#             s_values = self._s_values(y_bin_sizes, x_bin_sizes)
#             xi_values = self._xi_values(input_for_derivative, knot_xs, x_bin_sizes, floor_indices)
#             dervs = self._derivative_of_g_func(input_for_derivative, floor_indices, ceil_indices, xi_values, s_values, derivatives)
#             output = tf.ones(tf.shape(x), dtype=tf.float32)
#             squeezed = tf.tensor_scatter_nd_update(tf.dtypes.cast(tf.expand_dims(output,2), dtype=tf.float32), input_mask_indexes, tf.expand_dims(tf.transpose(tf.dtypes.cast(dervs,dtype=tf.float32)),1))
#             output = tf.squeeze(squeezed)
#             log_dervs = tf.math.log(output)
#             log_det_sum = tf.reduce_sum(log_dervs, axis=1)
#             return log_det_sum
#
#         r = tf.cond(tf.equal(tf.reduce_any(input_mask), tf.constant(False)), return_identity_log_det, return_result_log_det)
#         return r
#
#     def _inverse_log_det_jacobian(self, y):
#         neg_for_log_det = -1*self._forward_log_det_jacobian(self._inverse(y))
#         return neg_for_log_det
#
#
# '''--------------------------------------- Masked Autoregressive Flow -----------------------------------------------'''
#
# class Made(tfk.layers.Layer):
#     """
#     Implementation of a Masked Autoencoder for Distribution Estimation (MADE) [Germain et al. (2015)].
#     The existing TensorFlow bijector "AutoregressiveNetwork" is used. The output is reshaped to output one shift vector
#     and one log_scale vector.
#     :param params: Python integer specifying the number of parameters to output per input.
#     :param event_shape: Python list-like of positive integers (or a single int), specifying the shape of the input to this layer, which is also the event_shape of the distribution parameterized by this layer. Currently only rank-1 shapes are supported. That is, event_shape must be a single integer. If not specified, the event shape is inferred when this layer is first called or built.
#     :param hidden_units: Python list-like of non-negative integers, specifying the number of units in each hidden layer.
#     :param activation: An activation function. See tf.keras.layers.Dense. Default: None.
#     :param use_bias: Whether or not the dense layers constructed in this layer should have a bias term. See tf.keras.layers.Dense. Default: True.
#     :param kernel_regularizer: Regularizer function applied to the Dense kernel weight matrices. Default: None.
#     :param bias_regularizer: Regularizer function applied to the Dense bias weight vectors. Default: None.
#     """
#
#     def __init__(self, params, event_shape=None, hidden_units=None, activation=None, use_bias=True,
#                  kernel_regularizer=None, bias_regularizer=None, name="made"):
#
#         super(Made, self).__init__(name=name)
#
#         self.params = params
#         self.event_shape = event_shape
#         self.hidden_units = hidden_units
#         self.activation = activation
#         self.use_bias = use_bias
#         self.kernel_regularizer = kernel_regularizer
#         self.bias_regularizer = bias_regularizer
#
#         self.network = tfb.AutoregressiveNetwork(params=params, event_shape=event_shape, hidden_units=hidden_units,
#                                                  activation=activation, use_bias=use_bias, kernel_regularizer=kernel_regularizer,
#                                                  bias_regularizer=bias_regularizer)
#
#     def call(self, x):
#         shift, log_scale = tf.unstack(self.network(x), num=2, axis=-1)
#
#         return shift, tf.math.tanh(log_scale)
#
#
# '''------------------------------------- Batch Normalization Bijector -----------------------------------------------'''
#
# class BatchNorm(tfb.Bijector):
#     """
#     Implementation of a Batch Normalization layer for use in normalizing flows according to [Papamakarios et al. (2017)].
#     The moving average of the layer statistics is adapted from [Dinh et al. (2016)].
#     :param eps: Hyperparameter that ensures numerical stability, if any of the elements of v is near zero.
#     :param decay: Weight for the update of the moving average, e.g. avg = (1-decay)*avg + decay*new_value.
#     """
#
#     def __init__(self, eps=1e-5, decay=0.95, validate_args=False, name="batch_norm"):
#         super(BatchNorm, self).__init__(
#             forward_min_event_ndims=1,
#             inverse_min_event_ndims=1,
#             validate_args=validate_args,
#             name=name)
#
#         self._vars_created = False
#         self.eps = eps
#         self.decay = decay
#
#     def _create_vars(self, x):
#         # account for 1xd and dx1 vectors
#         if len(x.get_shape()) == 1:
#             n = x.get_shape().as_list()[0]
#         if len(x.get_shape()) == 2:
#             n = x.get_shape().as_list()[1]
#
#         self.beta = tf.compat.v1.get_variable('beta', [1, n], dtype=tf.float32)
#         self.gamma = tf.compat.v1.get_variable('gamma', [1, n], dtype=tf.float32)
#         self.train_m = tf.compat.v1.get_variable(
#             'mean', [1, n], dtype=tf.float32, trainable=False)
#         self.train_v = tf.compat.v1.get_variable(
#             'var', [1, n], dtype=tf.float32, trainable=False)
#
#         self._vars_created = True
#
#     def _forward(self, u):
#         if not self._vars_created:
#             self._create_vars(u)
#         return (u - self.beta) * tf.exp(-self.gamma) * tf.sqrt(self.train_v + self.eps) + self.train_m
#
#     def _inverse(self, x):
#         # Eq. 22 of [Papamakarios et al. (2017)]. Called during training of a normalizing flow.
#         if not self._vars_created:
#             self._create_vars(x)
#
#         # statistics of current minibatch
#         m, v = tf.nn.moments(x, axes=[0], keepdims=True)
#
#         # update train statistics via exponential moving average
#         self.train_v.assign_sub(self.decay * (self.train_v - v))
#         self.train_m.assign_sub(self.decay * (self.train_m - m))
#
#         # normalize using current minibatch statistics, followed by BN scale and shift
#         return (x - m) * 1. / tf.sqrt(v + self.eps) * tf.exp(self.gamma) + self.beta
#
#     def _inverse_log_det_jacobian(self, x):
#         # at training time, the log_det_jacobian is computed from statistics of the
#         # current minibatch.
#         if not self._vars_created:
#             self._create_vars(x)
#
#         _, v = tf.nn.moments(x, axes=[0], keepdims=True)
#         abs_log_det_J_inv = tf.reduce_sum(
#             self.gamma - .5 * tf.math.log(v + self.eps))
#         return abs_log_det_J_inv
#
#
# '''---------------------------------------------- Trainable Variables -----------------------------------------------'''
#
# def get_trainable_variables(flow):
#     """
#     Returns the number of trainable variables/weights of a flow.
#     :param flow: A normalizing flow in the form of a TensorFlow Transformed Distribution.
#     :return: n_trainable_variables
#     """
#     # number of trainable variables
#     n_trainable_variables = 0
#     for weights in flow.trainable_variables:
#         n_trainable_variables = n_trainable_variables + np.prod(weights.shape)
#
#     return n_trainable_variables


'''---------------------------------------------- main -----------------------------------------------'''

def flow(opts, inputs, scope=None, reuse=False):
    if opts['flow']=='identity':
        return inputs
    elif opts['flow']=='mlp':
        outputs = mlp(opts, inputs, scope, reuse)
    elif opts['flow']=='planar':
        outputs = planar(opts, inputs, scope, reuse)
    elif opts['flow']=='realNVP':
        outputs = realNVP(opts, inputs, scope, reuse)
    elif opts['flow']=='glow':
        outputs = glow(opts, inputs, scope, reuse)
    else:
        raise ValueError('Unknown {} flow' % flow_type)

    return outputs
