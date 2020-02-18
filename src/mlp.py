import tensorflow as tf
from typing import Union


class Dense(tf.Module):
    def __init__(self, input_dim, output_dim, activation: Union[any, None] = tf.nn.leaky_relu, name="dense_layer"):
        super(Dense, self).__init__(name=name)
        self.w = tf.Variable(tf.initializers.GlorotUniform()((input_dim, output_dim)))
        self.b = tf.Variable(tf.zeros((1, output_dim)))
        self.activation = activation

    @tf.function
    def __call__(self, x):
        y = tf.matmul(x, self.w) + self.b
        if self.activation:
            y = self.activation(y)
        return y


class MultiLayerPerceptron(tf.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, hidden_activation: Union[any, None] = tf.nn.leaky_relu,
                 final_activation: Union[any, None] = None, name="mlp"):
        super(MultiLayerPerceptron, self).__init__(name=name)
        self.layers = []
        from_dim = input_dim
        for hidden_dim in hidden_dims:
            to_dim = hidden_dim
            self.layers.append(Dense(from_dim, to_dim, hidden_activation))
            from_dim = to_dim
        self.layers.append(Dense(from_dim, output_dim, final_activation))

    @tf.function
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
