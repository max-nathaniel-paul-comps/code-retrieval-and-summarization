import tensorflow as tf
from typing import Union


class Dense(tf.Module):
    def __init__(self, input_dim, output_dim, activation: Union[any, None] = tf.nn.leaky_relu, name="dense_layer"):
        super(Dense, self).__init__(name=name)
        self.w = tf.Variable(tf.initializers.GlorotUniform()((input_dim, output_dim)))
        self.b = tf.Variable(tf.zeros((1, output_dim)))
        self.activation = activation

    def __call__(self, x):
        y = tf.matmul(x, self.w) + self.b
        if self.activation:
            y = self.activation(y)
        return y


class MultiLayerPerceptron(tf.Module):
    def __init__(self, input_dim, output_dim, hidden_dims,
                 hidden_activation: Union[any, None] = tf.nn.leaky_relu,
                 final_activation: Union[any, None] = None, name="mlp"):
        super(MultiLayerPerceptron, self).__init__(name=name)
        self.layers = []
        if type(input_dim) == tuple:
            input_dim = input_dim[0] * input_dim[1]
            self.reshape_input = True
        else:
            self.reshape_input = False
        from_dim = input_dim
        for hidden_dim in hidden_dims:
            to_dim = hidden_dim
            self.layers.append(Dense(from_dim, to_dim, hidden_activation))
            from_dim = to_dim
        if type(output_dim) == tuple:
            self.output_sub_dim = output_dim[1]
            output_dim = output_dim[0] * output_dim[1]
            self.reshape_output = True
        else:
            self.reshape_output = False
        to_dim = output_dim
        self.layers.append(Dense(from_dim, to_dim, final_activation))

    def __call__(self, x):
        if self.reshape_input:
            y = tf.reshape(x, (x.shape[0], x.shape[1] * x.shape[2]))
        else:
            y = x
        for layer in self.layers:
            y = layer(y)
        if self.reshape_output:
            y = tf.reshape(y, (x.shape[0], int(y.shape[1] / self.output_sub_dim), self.output_sub_dim))
        return y
