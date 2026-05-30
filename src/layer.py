from __future__ import absolute_import
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import activations, constraints, initializers, regularizers
from tensorflow.keras.layers import Layer
import numpy as np


class NR_GraphAttention(Layer):

    def __init__(self,
                 node_size,
                 rel_size,
                 triple_size,
                 depth=1,
                 use_w=False,
                 attn_heads=1,
                 attn_heads_reduction='concat',
                 activation=None,
                 use_bias=False,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 attn_kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 attn_kernel_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 attn_kernel_constraint=None,
                 **kwargs):

        if attn_heads_reduction not in {'concat', 'average'}:
            raise ValueError('Possible reduction methods: concat, average')

        self.node_size = node_size
        self.rel_size = rel_size
        self.triple_size = triple_size
        self.attn_heads = attn_heads
        self.attn_heads_reduction = attn_heads_reduction
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.use_w = use_w
        self.depth = depth

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.attn_kernel_initializer = initializers.get(attn_kernel_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.attn_kernel_regularizer = regularizers.get(attn_kernel_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.attn_kernel_constraint = constraints.get(attn_kernel_constraint)
        self.supports_masking = False

        self.biases = []
        self.attn_kernels = []

        super(NR_GraphAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        node_F = input_shape[0][-1]
        self.ent_F = node_F

        self.gate_kernel = self.add_weight(
            shape=(self.ent_F * (self.depth + 1), self.ent_F * (self.depth + 1)),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='gate_kernel'
        )

        self.proxy = self.add_weight(
            shape=(64, node_F * (self.depth + 1)),
            initializer=self.attn_kernel_initializer,
            regularizer=self.attn_kernel_regularizer,
            constraint=self.attn_kernel_constraint,
            name='proxy'
        )

        if self.use_bias:
            self.bias = self.add_weight(
                shape=(1, self.ent_F * (self.depth + 1)),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='bias'
            )

        for l in range(self.depth):
            self.attn_kernels.append([])
            for head in range(self.attn_heads):
                attn_kernel = self.add_weight(
                    shape=(node_F, 1),
                    initializer=self.attn_kernel_initializer,
                    regularizer=self.attn_kernel_regularizer,
                    constraint=self.attn_kernel_constraint,
                    name=f'attn_kernel_self_{l}_{head}'
                )
                self.attn_kernels[l].append(attn_kernel)

        self.built = True

    def call(self, inputs, training=None):
        outputs = []
        features = inputs[0]
        rel_emb = inputs[1]

        adj_indices_raw = inputs[2]
        sparse_indices_raw = inputs[3]
        sparse_val_raw = inputs[4]

        adj_indices = tf.cast(tf.squeeze(adj_indices_raw, axis=0), dtype=tf.int64)
        sparse_indices = tf.cast(tf.squeeze(sparse_indices_raw, axis=0), dtype=tf.int64)
        sparse_val = tf.cast(tf.squeeze(sparse_val_raw, axis=0), dtype=tf.float32)

        if self.use_w:
            features = features * self.gcn_kernel
        features = self.activation(features)
        outputs.append(features)

        try:
            adj_st = tf.SparseTensor(
                indices=adj_indices,
                values=tf.ones(tf.shape(adj_indices)[0], dtype=tf.float32),
                dense_shape=[self.node_size, self.node_size]
            )
            adj_st = tf.sparse.reorder(adj_st)
            use_sparse = True
        except Exception:
            use_sparse = False

        for l in range(self.depth):
            features_list = []
            for head in range(self.attn_heads):
                attention_kernel = self.attn_kernels[l][head]

                row_indices = tf.cast(sparse_indices[:, 0], tf.int32)
                col_indices = tf.cast(sparse_indices[:, 1], tf.int32)

                rel_gathered = tf.gather(rel_emb, col_indices)
                weighted_rel = rel_gathered * tf.expand_dims(sparse_val, axis=-1)

                rels_sum = tf.math.unsorted_segment_sum(
                    weighted_rel, row_indices, self.triple_size
                )

                adj_indices_int32 = tf.cast(adj_indices, tf.int32)
                neighs = tf.gather(features, adj_indices_int32[:, 1])

                rels_sum = tf.math.l2_normalize(rels_sum, axis=-1)

                dot_product = tf.reduce_sum(neighs * rels_sum, axis=1, keepdims=True)
                neighs = neighs - 2.0 * dot_product * rels_sum

                att = tf.squeeze(tf.matmul(rels_sum, attention_kernel), axis=-1)

                if use_sparse:
                    try:
                        att_st = tf.SparseTensor(
                            indices=adj_indices,
                            values=att,
                            dense_shape=[self.node_size, self.node_size]
                        )
                        att_st = tf.sparse.reorder(att_st)
                        att_st = tf.sparse.softmax(att_st)
                        att = att_st.values
                    except Exception:
                        att = self._segment_softmax(att, adj_indices_int32[:, 0], self.node_size)
                else:
                    att = self._segment_softmax(att, adj_indices_int32[:, 0], self.node_size)

                weighted_neighs = neighs * tf.expand_dims(att, axis=-1)
                new_features = tf.math.unsorted_segment_sum(
                    weighted_neighs, adj_indices_int32[:, 0], self.node_size
                )
                features_list.append(new_features)

            if self.attn_heads_reduction == 'concat':
                features = tf.concat(features_list, axis=-1)
            else:
                features = tf.reduce_mean(tf.stack(features_list), axis=0)

            features = self.activation(features)
            outputs.append(features)

        outputs = tf.concat(outputs, axis=-1)

        outputs_norm = tf.math.l2_normalize(outputs, axis=-1)
        proxy_norm = tf.math.l2_normalize(self.proxy, axis=-1)

        proxy_att = tf.matmul(outputs_norm, tf.transpose(proxy_norm))
        proxy_att = tf.nn.softmax(proxy_att, axis=-1)
        proxy_feature = outputs - tf.matmul(proxy_att, self.proxy)

        if self.use_bias:
            gate_rate = tf.sigmoid(tf.matmul(proxy_feature, self.gate_kernel) + self.bias)
        else:
            gate_rate = tf.sigmoid(tf.matmul(proxy_feature, self.gate_kernel))

        outputs = gate_rate * outputs + (1.0 - gate_rate) * proxy_feature

        if self.use_w:
            return [outputs] + [self.gcn_kernel]
        else:
            return outputs

    def _segment_softmax(self, values, segment_ids, num_segments):
        max_vals = tf.math.unsorted_segment_max(values, segment_ids, num_segments)
        max_vals = tf.gather(max_vals, segment_ids)
        exp_vals = tf.exp(values - max_vals)
        sum_exp = tf.math.unsorted_segment_sum(exp_vals, segment_ids, num_segments)
        sum_exp = tf.gather(sum_exp, segment_ids)
        return exp_vals / (sum_exp + 1e-10)

    def compute_output_shape(self, input_shape):
        node_shape = (self.node_size, input_shape[0][-1] * (self.depth + 1))
        if not self.use_w:
            return node_shape
        else:
            return [node_shape] + [self.gcn_kernel.shape]

    def get_config(self):
        config = super().get_config()
        config.update({
            'node_size': self.node_size,
            'rel_size': self.rel_size,
            'triple_size': self.triple_size,
            'depth': self.depth,
            'use_w': self.use_w,
            'attn_heads': self.attn_heads,
            'attn_heads_reduction': self.attn_heads_reduction,
            'use_bias': self.use_bias,
        })
        return config
