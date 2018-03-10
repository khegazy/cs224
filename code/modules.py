# Copyright 2018 Stanford University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This file contains some basic model components"""

import tensorflow as tf
from tensorflow.python.ops.rnn_cell import DropoutWrapper
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import rnn_cell


class RNNEncoder(object):
    """
    General-purpose module to encode a sequence using a RNN.
    It feeds the input through a RNN and returns all the hidden states.

    Note: In lecture 8, we talked about how you might use a RNN as an "encoder"
    to get a single, fixed size vector representation of a sequence
    (e.g. by taking element-wise max of hidden states).
    Here, we're using the RNN as an "encoder" but we're not taking max;
    we're just returning all the hidden states. The terminology "encoder"
    still applies because we're getting a different "encoding" of each
    position in the sequence, and we'll use the encodings downstream in the model.

    This code uses a bidirectional GRU, but you could experiment with other types of RNN.
    """

    def __init__(self, hidden_size, keep_prob):
        """
        Inputs:
          hidden_size: int. Hidden size of the RNN
          keep_prob: Tensor containing a single scalar that is the keep probability (for dropout)
        """
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob
        self.rnn_cell_fw = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_fw = DropoutWrapper(self.rnn_cell_fw, input_keep_prob=self.keep_prob)
        self.rnn_cell_bw = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_bw = DropoutWrapper(self.rnn_cell_bw, input_keep_prob=self.keep_prob)

    def build_graph(self, inputs, masks):
        """
        Inputs:
          inputs: Tensor shape (batch_size, seq_len, input_size)
          masks: Tensor shape (batch_size, seq_len).
            Has 1s where there is real input, 0s where there's padding.
            This is used to make sure tf.nn.bidirectional_dynamic_rnn doesn't iterate through masked steps.

        Returns:
          out: Tensor shape (batch_size, seq_len, hidden_size*2).
            This is all hidden states (fw and bw hidden states are concatenated).
        """
        with vs.variable_scope("RNNEncoder"):
            input_lens = tf.reduce_sum(masks, reduction_indices=1) # shape (batch_size)

            # Note: fw_out and bw_out are the hidden states for every timestep.
            # Each is shape (batch_size, seq_len, hidden_size).
            (fw_out, bw_out), _ = tf.nn.bidirectional_dynamic_rnn(self.rnn_cell_fw, self.rnn_cell_bw, inputs, input_lens, dtype=tf.float32)

            # Concatenate the forward and backward hidden states
            out = tf.concat([fw_out, bw_out], 2)

            # Apply dropout
            out = tf.nn.dropout(out, self.keep_prob)

            return out


class SimpleSoftmaxLayer(object):
    """
    Module to take set of hidden states, (e.g. one for each context location),
    and return probability distribution over those states.
    """

    def __init__(self):
        pass

    def build_graph(self, inputs, masks):
        """
        Applies one linear downprojection layer, then softmax.

        Inputs:
          inputs: Tensor shape (batch_size, seq_len, hidden_size)
          masks: Tensor shape (batch_size, seq_len)
            Has 1s where there is real input, 0s where there's padding.

        Outputs:
          logits: Tensor shape (batch_size, seq_len)
            logits is the result of the downprojection layer, but it has -1e30
            (i.e. very large negative number) in the padded locations
          prob_dist: Tensor shape (batch_size, seq_len)
            The result of taking softmax over logits.
            This should have 0 in the padded locations, and the rest should sum to 1.
        """
        with vs.variable_scope("SimpleSoftmaxLayer"):

            # Linear downprojection layer
            logits = tf.contrib.layers.fully_connected(inputs, num_outputs=1, activation_fn=None) # shape (batch_size, seq_len, 1)
            logits = tf.squeeze(logits, axis=[2]) # shape (batch_size, seq_len)

            # Take softmax over sequence
            masked_logits, prob_dist = masked_softmax(logits, masks, 1)

            return masked_logits, prob_dist


class BasicAttn(object):
    """Module for basic attention.

    Note: in this module we use the terminology of "keys" and "values" (see lectures).
    In the terminology of "X attends to Y", "keys attend to values".

    In the baseline model, the keys are the context hidden states
    and the values are the question hidden states.

    We choose to use general terminology of keys and values in this module
    (rather than context and question) to avoid confusion if you reuse this
    module with other inputs.
    """

    def __init__(self, keep_prob, key_vec_size, value_vec_size):
        """
        Inputs:
          keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
          key_vec_size: size of the key vectors. int
          value_vec_size: size of the value vectors. int
        """
        self.keep_prob = keep_prob
        self.key_vec_size = key_vec_size
        self.value_vec_size = value_vec_size

    def build_graph(self, values, values_mask, keys):
        """
        Keys attend to values.
        For each key, return an attention distribution and an attention output vector.

        Inputs:
          values: Tensor shape (batch_size, num_values, value_vec_size).
          values_mask: Tensor shape (batch_size, num_values).
            1s where there's real input, 0s where there's padding
          keys: Tensor shape (batch_size, num_keys, value_vec_size)

        Outputs:
          attn_dist: Tensor shape (batch_size, num_keys, num_values).
            For each key, the distribution should sum to 1,
            and should be 0 in the value locations that correspond to padding.
          output: Tensor shape (batch_size, num_keys, hidden_size).
            This is the attention output; the weighted sum of the values
            (using the attention distribution as weights).
        """
        print("\n\nbuilding basicattention")
        with vs.variable_scope("BasicAttn"):

            # Calculate attention distribution
            values_t = tf.transpose(values, perm=[0, 2, 1]) # (batch_size, value_vec_size, num_values)
            attn_logits = tf.matmul(keys, values_t) # shape (batch_size, num_keys, num_values)
            attn_logits_mask = tf.expand_dims(values_mask, 1) # shape (batch_size, 1, num_values)
            _, attn_dist = masked_softmax(attn_logits, attn_logits_mask, 2) # shape (batch_size, num_keys, num_values). take softmax over values

            # Use attention distribution to take weighted sum of values
            output = tf.matmul(attn_dist, values) # shape (batch_size, num_keys, value_vec_size)

            # Apply dropout
            output = tf.nn.dropout(output, self.keep_prob)

            return attn_dist, output



class bidirectionalAttn(object):

    def __init__(self, keep_prob, num_keys, key_vec_size, num_vals, value_vec_size):
        """
        Inputs:
          keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
          key_vec_size: size of the key vectors. int
          value_vec_size: size of the value vectors. int
        """
        self.keep_prob = keep_prob
        self.num_keys = num_keys
        self.key_vec_size = key_vec_size
        self.num_vals = num_vals
        self.value_vec_size = value_vec_size

    def build_graph(self, values, values_mask, keys, keys_mask):
        """
        Keys attend to values.
        For each key, return an attention distribution and an attention output vector.

        Inputs:
          values: Tensor shape (batch_size, num_values, value_vec_size).
          values_mask: Tensor shape (batch_size, num_values).
            1s where there's real input, 0s where there's padding
          keys: Tensor shape (batch_size, num_keys, value_vec_size)

        Outputs:
          attn_dist: Tensor shape (batch_size, num_keys, num_values).
            For each key, the distribution should sum to 1,
            and should be 0 in the value locations that correspond to padding.
          output: Tensor shape (batch_size, num_keys, hidden_size).
            This is the attention output; the weighted sum of the values
            (using the attention distribution as weights).
        """
        print("\n\nbuilding bidirattention")
        with vs.variable_scope("bidirectionalAttn"):

            Wvals = tf.get_variable(name="simVals", shape=(self.key_vec_size),
                dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            Wkeys = tf.get_variable(name="simKeys", shape=(self.key_vec_size),
                dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            Wovrlp = tf.get_variable(name="simOvrlp", shape=(self.key_vec_size),
                dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())


            #(batch_size, num_keys)
            simKeys   = tf.reduce_sum(tf.multiply(keys, Wkeys), axis=2)
            #(batch_size, num_values)
            simVals   = tf.reduce_sum(tf.multiply(values, Wvals), axis=2)

            #(batch_size, num_keys, num_values)
            simOvrlp = tf.stack([tf.reduce_sum(
                                    tf.multiply(
                                      tf.expand_dims(keys[:,i,:], 1), values), 
                                    axis=2) 
                                  for i in range(self.num_keys)], 
                                axis=1)
            print("simOverlp", simOvrlp.shape.as_list())

            #(batch_size, num_keys, num_values, 3*vec_size)
            S         = tf.tile(tf.expand_dims(simKeys, 2), tf.stack([1,1,self.num_vals])) +\
                        tf.tile(tf.expand_dims(simVals, 1), tf.stack([1,self.num_keys,1])) + simOvrlp
            print("s",S.shape.as_list())
            
            mask = tf.expand_dims(values_mask, 1)
            print("mask",mask.shape.as_list())
            _, alpha = masked_softmax(S, mask, 2)
            print("alpha",alpha.shape.as_list())

            a = tf.stack([tf.reduce_sum(
                            tf.multiply(
                              tf.expand_dims(alpha[:,i,:], 2), values), axis=1) 
                            for i in range(self.num_keys)], 
                          axis=1)
            print("a",a.shape.as_list())

            m = tf.reduce_max(S, axis=2)
            print("m",m.shape.as_list())
            _, beta = masked_softmax(m, keys_mask, 1)
            print("b",beta.shape.as_list())
            beta = tf.expand_dims(beta, 2)
            c = tf.multiply(beta, keys)
            print("c",c.shape.as_list())
            c = tf.reduce_sum(c, axis=1)
            print("c",c.shape.as_list())

            cTile = tf.tile(tf.expand_dims(c, 1), [1,self.num_keys,1])
            output = tf.concat([keys, a, cTile], axis=2)
            print("output",output.shape.as_list())

            # Apply dropout
            output = tf.nn.dropout(output, self.keep_prob)

            return output



class coattention(object):

    def __init__(self, keep_prob, batch_size, num_keys, key_vec_size, num_vals, value_vec_size):
        """
        Inputs:
          keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
          key_vec_size: size of the key vectors. int
          value_vec_size: size of the value vectors. int
        """
        self.keep_prob = keep_prob
        self.batch_size = batch_size
        self.num_keys = num_keys
        self.key_vec_size = key_vec_size
        self.num_vals = num_vals
        self.value_vec_size = value_vec_size

    def build_graph(self, values, values_mask, keys, keys_mask):
        """
        Keys attend to values.
        For each key, return an attention distribution and an attention output vector.

        Inputs:
          values: Tensor shape (batch_size, num_values, value_vec_size).
          values_mask: Tensor shape (batch_size, num_values).
            1s where there's real input, 0s where there's padding
          keys: Tensor shape (batch_size, num_keys, value_vec_size)

        Outputs:
          attn_dist: Tensor shape (batch_size, num_keys, num_values).
            For each key, the distribution should sum to 1,
            and should be 0 in the value locations that correspond to padding.
          output: Tensor shape (batch_size, num_keys, hidden_size).
            This is the attention output; the weighted sum of the values
            (using the attention distribution as weights).
        """
        print("\n\nbuilding coattention")
        with vs.variable_scope("coattention"):

            vS = tf.get_variable(name="valSentinel", shape=(1,1,self.value_vec_size),
                dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            kS = tf.get_variable(name="keySentinel", shape=(1,1,self.key_vec_size),
                dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            mask = tf.ones(name="addMask", shape=(1,1), dtype=tf.int32)


            valTanh   = tf.contrib.layers.fully_connected(values, self.value_vec_size, activation_fn=tf.nn.tanh) 
            valSent   = tf.concat([valTanh,tf.tile(vS, [self.batch_size,1,1])], axis=1)
            keySent   = tf.concat([keys,tf.tile(kS, [self.batch_size,1,1])], axis=1)

            L = tf.stack([tf.reduce_sum(
                            tf.multiply(
                              tf.expand_dims(keySent[:,i,:], 1), valSent), 
                            axis=2) 
                          for i in range(self.num_keys+1)], 
                        axis=1)
           
            print("L",L.shape.as_list())
            valMask = tf.concat([values_mask,tf.tile(mask,[self.batch_size,1])], axis=1)
            keyMask = tf.concat([keys_mask,tf.tile(mask,[self.batch_size,1])], axis=1)
            print("vmask",valMask.shape.as_list())
            print("kmask",keyMask.shape.as_list())

            _, alpha = masked_softmax(L, tf.expand_dims(valMask, 1), 2)
            print("alpha",alpha.shape.as_list())
            a = tf.stack([tf.reduce_sum(
                            tf.multiply(
                              tf.expand_dims(alpha[:,i,:], 2), valSent), axis=1) 
                            for i in range(self.num_keys+1)], 
                          axis=1)

            print("a",a.shape.as_list())

            _, beta = masked_softmax(L, tf.expand_dims(keyMask, 2), 1)
            print("beta",beta.shape.as_list())
            #beta = tf.expand_dims(beta, 2)
            b = tf.stack([tf.reduce_sum(
                            tf.multiply(
                              tf.expand_dims(beta[:,:,i], 2), keySent), axis=1) 
                            for i in range(self.num_vals+1)], 
                          axis=1)

            print("b",b.shape.as_list())
            s = tf.stack([tf.reduce_sum(
                            tf.multiply(
                              tf.expand_dims(alpha[:,i,:], 2), b), axis=1) 
                            for i in range(self.num_keys+1)], 
                          axis=1)

            print("s",s.shape.as_list())


            with vs.variable_scope("lstm"):
              lstmInput = tf.concat([s[:,:-1,:],a[:,:-1,:]], axis=2)
              print("inp",lstmInput.shape.as_list())

              #lstmCell = tf.nn.rnn_cell.LSTMCell(1);
              lstmCell = tf.contrib.rnn.LSTMCell(self.key_vec_size);
              outputs,states = tf.nn.dynamic_rnn(lstmCell, lstmInput, dtype=tf.float32)
            #cTile = tf.tile(tf.expand_dims(c, 1), [1,num_keys,1])
            #output = tf.concat([keys, a, cTile], axis=2)
            print("output",outputs.shape.as_list())

            # Apply dropout
            output = tf.nn.dropout(outputs, self.keep_prob)

            return output

class get_attn_weights(object):
  
    def __init__(self, NattnModels, batch_size, question_len, context_len, hidden_size):
      	self.NattnModels = NattnModels
        self.batch_size = batch_size
        self.question_len = question_len
        self.context_len = context_len
        self.hidden_size = hidden_size

    def build_graph(self, question_hiddens, attentions, isTraining):
        with vs.variable_scope("attenWeights"):
          	#lstmCell = tf.contrib.rnn.LSTMCell(self.hidden_size//2, name="questionCell")
          	#outputs,states = tf.nn.dynamic_rnn(lstmCell, question_hiddens, dtype=tf.float32)
          	#lstmOut = tf.reshape(outputs, shape=(self.batch_size, self.question_len*self.hidden_size))
            qSize = self.question_len*self.hidden_size*2
            qHiddens  = tf.reshape(question_hiddens, shape=(self.batch_size, qSize))
            qLinear1   = tf.layers.dense(qHiddens, units=qSize//5)
            #qNorm1     = tf.layers.batch_normalization(qLinear1, training=isTraining)
            qNLinear1  = tf.nn.relu(qLinear1)
            qLinear2   = tf.layers.dense(qNLinear1, units=2*self.hidden_size)
            #qNorm2     = tf.layers.batch_normalization(qLinear2, training=isTraining)
            qNLinear2  = tf.nn.relu(qLinear2)
            qSummary  = tf.layers.dense(qNLinear2, units=self.hidden_size)

            aSize     = attentions.shape.as_list()[2] 
            inpMask   = tf.concat([attentions, 
                            tf.tile(tf.expand_dims(qSummary, 1), [1, self.context_len, 1])],
                            axis=2)
            mLinear1  = tf.layers.dense(inpMask, units=aSize*1.1)
            #mNorm1    = tf.layers.batch_normalization(mLinear1, training=isTraining)
            mNLinear1 = tf.nn.relu(mLinear1)
            mLinear2  = tf.layers.dense(mNLinear1, units=aSize)
            #mNorm2    = tf.layers.batch_normalization(mLinear2, training=isTraining)
            mNLinear2 = tf.nn.relu(mLinear2)
            mLinear3  = tf.layers.dense(mNLinear2, units=aSize)
            return tf.sigmoid(mLinear3)




def masked_softmax(logits, mask, dim):
    """
    Takes masked softmax over given dimension of logits.

    Inputs:
      logits: Numpy array. We want to take softmax over dimension dim.
      mask: Numpy array of same shape as logits.
        Has 1s where there's real data in logits, 0 where there's padding
      dim: int. dimension over which to take softmax

    Returns:
      masked_logits: Numpy array same shape as logits.
        This is the same as logits, but with 1e30 subtracted
        (i.e. very large negative number) in the padding locations.
      prob_dist: Numpy array same shape as logits.
        The result of taking softmax over masked_logits in given dimension.
        Should be 0 in padding locations.
        Should sum to 1 over given dimension.
    """
    exp_mask = (1 - tf.cast(mask, 'float')) * (-1e30) # -large where there's padding, 0 elsewhere
    masked_logits = tf.add(logits, exp_mask) # where there's padding, set logits to -large
    prob_dist = tf.nn.softmax(masked_logits, dim)
    return masked_logits, prob_dist
