import numpy as np
import tensorflow as tf
import keras
import keras.backend as K
from keras.layers import Input, Dropout, Lambda, Concatenate
from ..layer import NR_GraphAttention


def _sparse_softmax(st):
    if hasattr(tf, "sparse_softmax"):
        return tf.sparse_softmax(st)
    return tf.sparse.softmax(st)


def _sparse_dense_matmul(st, dense):
    if hasattr(tf, "sparse_tensor_dense_matmul"):
        return tf.sparse_tensor_dense_matmul(st, dense)
    return tf.sparse.sparse_dense_matmul(st, dense)


class TokenEmbedding(keras.layers.Embedding):
    def compute_output_shape(self, input_shape):
        return self.input_dim, self.output_dim

    def compute_mask(self, inputs, mask=None):
        return None

    def call(self, inputs):
        return self.embeddings


def build_duala_models(
    *,
    node_size: int,
    rel_size: int,
    triple_size: int,
    node_hidden: int = 128,
    rel_hidden: int = 128,
    dropout_rate: float = 0.3,
    gamma: float = 1.0,
    lr: float = 0.005,
    depth: int = 2,
):
    adj_input = Input(shape=(None, 2))
    index_input = Input(shape=(None, 2), dtype='int64')
    val_input = Input(shape=(None,))
    rel_adj = Input(shape=(None, 2))
    ent_adj = Input(shape=(None, 2))

    ent_emb = TokenEmbedding(node_size, node_hidden, trainable=True)(val_input)
    rel_emb = TokenEmbedding(rel_size, node_hidden, trainable=True)(val_input)

    def avg(tensor, size: int):
        adj = K.cast(K.squeeze(tensor[0], axis=0), dtype="int64")
        st = tf.SparseTensor(
            indices=adj,
            values=tf.ones_like(adj[:, 0], dtype='float32'),
            dense_shape=(node_size, size),
        )
        st = _sparse_softmax(st)
        return _sparse_dense_matmul(st, tensor[1])

    opt = [rel_emb, adj_input, index_input, val_input]
    ent_feature = Lambda(avg, arguments={'size': node_size})([ent_adj, ent_emb])
    rel_feature = Lambda(avg, arguments={'size': rel_size})([rel_adj, rel_emb])

    e_encoder = NR_GraphAttention(
        node_size, activation="tanh", rel_size=rel_size,
        use_bias=True, depth=depth, triple_size=triple_size,
    )
    r_encoder = NR_GraphAttention(
        node_size, activation="tanh", rel_size=rel_size,
        use_bias=True, depth=depth, triple_size=triple_size,
    )

    out_feature = Concatenate(-1)([
        e_encoder([ent_feature] + opt),
        r_encoder([rel_feature] + opt),
    ])
    out_feature = Dropout(dropout_rate)(out_feature)

    alignment_input = Input(shape=(None, 2))

    def align_loss(tensor):
        def squared_dist(x):
            A, B = x
            row_norms_A = tf.reduce_sum(tf.square(A), axis=1)
            row_norms_A = tf.reshape(row_norms_A, [-1, 1])
            row_norms_B = tf.reduce_sum(tf.square(B), axis=1)
            row_norms_B = tf.reshape(row_norms_B, [1, -1])
            return row_norms_A + row_norms_B - 2 * tf.matmul(A, B, transpose_b=True)

        emb = tensor[1]
        l = K.cast(tensor[0][0, :, 0], 'int32')
        r = K.cast(tensor[0][0, :, 1], 'int32')
        l_emb = K.gather(reference=emb, indices=l)
        r_emb = K.gather(reference=emb, indices=r)

        pos_dis = K.sum(K.square(l_emb - r_emb), axis=-1, keepdims=True)

        r_neg_dis = squared_dist([r_emb, emb])
        l_neg_dis = squared_dist([l_emb, emb])

        l_loss = pos_dis - l_neg_dis + gamma
        l_loss = l_loss * (1 - K.one_hot(indices=l, num_classes=node_size) - K.one_hot(indices=r, num_classes=node_size))

        r_loss = pos_dis - r_neg_dis + gamma
        r_loss = r_loss * (1 - K.one_hot(indices=l, num_classes=node_size) - K.one_hot(indices=r, num_classes=node_size))

        r_loss = (r_loss - K.stop_gradient(K.mean(r_loss, axis=-1, keepdims=True))) / K.stop_gradient(K.std(r_loss, axis=-1, keepdims=True))
        l_loss = (l_loss - K.stop_gradient(K.mean(l_loss, axis=-1, keepdims=True))) / K.stop_gradient(K.std(l_loss, axis=-1, keepdims=True))

        lamb, tau = 30.0, 10.0
        l_loss = tf.reduce_logsumexp(lamb * l_loss + tau, axis=-1)
        r_loss = tf.reduce_logsumexp(lamb * r_loss + tau, axis=-1)
        return K.mean(l_loss + r_loss)

    loss = Lambda(align_loss)([alignment_input, out_feature])

    inputs = [adj_input, index_input, val_input, rel_adj, ent_adj]
    train_model = keras.Model(inputs=inputs + [alignment_input], outputs=out_feature)
    train_model.add_loss(loss)
    train_model.compile(optimizer=tf.keras.optimizers.legacy.RMSprop(learning_rate=lr))

    feature_model = keras.Model(inputs=inputs, outputs=out_feature)
    return train_model, feature_model


def predict_embeddings(feature_model, *, adj_edges, r_index, r_val, rel_matrix, ent_matrix):
    inputs = [adj_edges, r_index, r_val, rel_matrix, ent_matrix]
    inputs = [np.expand_dims(x, axis=0) for x in inputs]
    vec = feature_model.predict_on_batch(inputs)
    return vec
