import tensorflow as tf


class batch_norm(object):
  def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
    with tf.compat.v1.variable_scope(name):
      self.epsilon  = epsilon
      self.momentum = momentum
      self.name = name

  def __call__(self, x, train=True):
    return tf.keras.layers.BatchNormalization(
             name="BatchNorm", 
            momentum=self.momentum,
             epsilon=self.epsilon,
              scale=True,
               )(x,training=train)


# Leaky Relu
def lrelu(x, alpha = 0.2, name='lrelu'):
    return tf.maximum(x, alpha*x)

def dense(x, inp_dim, out_dim, name = 'dense'):
    """
    Used to create a dense layer.
    :param x: input tensor to the dense layer
    :param inp_dim: no. of input neurons
    :param out_dim: no. of output neurons
    :param name: name of the entire dense layer.i.e, variable scope name.
    :return: tensor with shape [batch_size, out_dim]
    """
    with tf.compat.v1.variable_scope(name, reuse=None):
        weights = tf.compat.v1.get_variable("weights", shape=[inp_dim, out_dim],
                                  initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"),use_resource=False)
        bias = tf.compat.v1.get_variable("bias", shape=[out_dim], initializer=tf.compat.v1.constant_initializer(0.0),use_resource=False)
        out = tf.add(tf.matmul(x, weights), bias, name='matmul')
        return out