import numpy as np
from keras import regularizers
from keras.activations import relu
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.initializers import RandomNormal
from keras.layers import Input, Embedding, Conv1D, GlobalMaxPool1D, MaxPooling1D, LSTM, Bidirectional, Dense, Dropout, \
    PReLU, BatchNormalization, Lambda, CuDNNGRU, Flatten
from keras.layers.merge import add
from keras.models import Model
from keras.optimizers import SGD, Adam, Adagrad, Adadelta
import math
import tensorflow as tf


'''
class KMaxPooling(Layer):
    """
    K-max pooling layer that extracts the k-highest activations from a sequence (2nd dimension).
    TensorFlow backend.
    """
    def __init__(self, k=1, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = InputSpec(ndim=3)
        self.k = k

    def compute_output_shape(self, input_shape):
        return (input_shape[0], (input_shape[2] * self.k))

    def call(self, inputs):

        # swap last two dimensions since top_k will be applied along the last dimension
        shifted_input = tf.transpose(inputs, [0, 2, 1])

        # extract top_k, returns two tensors [values, indices]
        top_k = tf.nn.top_k(shifted_input, k=self.k, sorted=True, name=None)[0]

        # return flattened output
        return Flatten()(top_k)

class DynamicKMaxPooling(Layer):

    def __init__(self, **kwargs):
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(MyLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return K.dot(x, self.kernel)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
'''

### Statistical Models ###
class nbsvm():
    def pr(self, y_i, y, data):
        #print y.shape, type(y), y_i, y==y_i, type(data), data[y==y_i]
        p = data[y==y_i].sum(0)
        return (p+1) / ((y==y_i).sum()+1)

    def fit(self, data, y):
        y = y.values
        #print 'y shape', y.shape
        r = sparse.csr_matrix(np.log(self.pr(1,y, data) / self.pr(0,y, data)))
        m = LogisticRegression(C=4, dual=True)
        x_nb = data.multiply(r)
        #np.multiply(r.reshape(1, len(r)), data.values)
        #print np.isnan(x_nb).any(), np.isnan(y).any(), np.isinf(x_nb).any(), np.isinf(y).any()
        self.model = (m.fit(x_nb, y), r)
        return self.model

    def predict_proba(self, data):
        m, r = self.model
        #return m.predict_proba(np.multiply(r.reshape(1, len(r)), data.values))
        return m.predict_proba(data.multiply(r))

### Statistical Models ###
### Neural Network based model ###
#Building things with pre-activation
def _convolutional_block(filter_count, kernel_size, l2_reg_convo):
    def f(x):
        x =  Lambda(relu(x))
        convLayer = Conv1D(filter_count, kernel_size,
                        kernel_initializer = RandomNormal(mean=0.0, stddev=0.001),
                        kernel_regularizer =  keras.regularizers.l2(l2_reg_convo))(x)
        return convLayer
    return f

def _shape_matching_layer(filter_nr, l2_reg):
    def f(x):
        x = Conv1D(filter_nr, kernel_size=1, padding='same', activation='linear',
                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
                   kernel_regularizer=regularizers.l2(l2_reg))(x)
        x = Lambda(relu)(x)
        return x

    return f

def _dpcnnBlock(filter_count, kernel_size, l2_reg_convo):
    def f(x):
        x = MaxPooling1D(pool_size=3, strides=2)(x)
        conv_1 = _convolutional_block(filter_count, kernel_size, l2_reg_convo)(x)
        conv_2 = _convolutional_block(filter_count, kernel_size, l2_reg_convo)(conv_1)
        ret_x = add([conv_2, x])
        return ret_x
    return f

def _dense_block(dense_size, dropout, l2_reg):
    def f(x):
        x = Dense(dense_size, activation='linear',
                  kernel_regularizer=regularizers.l2(l2_reg))(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)
        return x

    return f

def DPCNN(embedding_matrix, embedding_size, input_len, max_features, filter_count,
          kernel_size, repeat_block, dropout_convo, dense_size, repeat_dense,
          dropout_dense, l2_reg_convo, l2_reg_dense):
    inputText = Input((input_len, ))
    embedding = Embedding(max_features, embedding_size, weights = [embedding_matrix], trainable=True)(inputText)
    convo = _convolutional_block(filter_count, kernel_size, l2_reg_convo)(embedding)
    convo = _convolutional_block(filter_count, kernel_size, l2_reg_convo)(convo)
    if filter_count == embedding_size:
        x = add([convo, embedding])
    else:
        embedding_reshaped = _shape_matching_layer(filter_count, l2_reg_convo)(embedding)
        x = add([convo, embedding_reshaped])

    for _ in range(repeat_block):
        x = _dpcnn_block(filter_nr, kernel_size, l2_reg_convo)(x)

    x = GlobalMaxPool1D()(x)
    for _ in range(repeat_dense):
        x = _dense_block(dense_size, dropout_dense, l2_reg_dense)(x)

    predictions = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=input_text, outputs=predictions)
    return model



'''
#Dynamic CNN implementation
class DCNN(object):
    def __init__(self, num_classes, vocab_size, embedding_size, k_top,
                 filter_sizes, num_filters, dropout_convo, dense_size,
                 repeat_dense, dropout_dense, l2_reg_convo, l2_reg_dense,
                 embedding_matrix):
        #implementation
        self.input_x = tf.placeholder(tf.int32, [None, None], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            W = tf.Variable(embedding_matrix, name="W")
            self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        convoInput = self.embedded_chars_expanded
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-dynamicMaxKpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv1d(
                    convoInput,
                    W,
                    strides=[1, 1, 1],
                    padding="SAME",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Need to change it to dynamic pooling
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")


    def getKValue(self, k_top, total_layers, currentLayer, inputLength):
        return max(k_top, math.ceil((float(total_layers - currentLayer)/total_layers)*inputLength))

    def DynamicKMaxPooling(self, input_x, k_top, total_layers, currentLayer):
        #We have to figure out the padding requirement as well

        #Shifting to as topk will work only on the last dimension
        shifted_input = tf.transpose(input_x, [0, 2, 1])
        # extract top_k, returns two tensors [values, indices]
        top_k = tf.nn.top_k(shifted_input, k=self.k, sorted=True, name=None)[0]

'''

class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes, embedding_matrix,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(embedding_matrix, name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
            self.embedded_chars_expanded = tf.cast(self.embedded_chars_expanded, tf.float32)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
