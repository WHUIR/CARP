# -*- coding: utf-8 -*-
import numpy as np
import math, heapq, random, os, sys
import tensorflow as tf
from time import time
from auxiliaryTools.ExtractData import Dataset
from auxiliaryTools.Evaluation import get_test_list_mask
epsilon = 1e-9

def ini_word_embed(num_words, latent_dim):
    word_embeds = np.random.rand(num_words, latent_dim)
    return word_embeds

def get_train_instance(train):
    user_input, item_input, rates = [], [], []

    for (u, i) in train.keys():
        # positive instance
        user_input.append(u)
        item_input.append(i)
        rates.append(train[u,i])
    return user_input, item_input, rates

def ini_cap(dim1_tr, dim2, dim3):
    return np.zeros([dim1_tr, dim2, dim3])

def get_pos_neg(thrhld, rate):
    if rate > thrhld:
        return 1.0
    else:
        return 0.0

def get_train_instance_batch(count, batch_size, user_input, item_input, ratings, user_reviews, item_reviews, user_masks, item_masks):
    users_batch, items_batch, user_input_batch, item_input_batch, user_mask_batch, item_mask_batch, labels_batch, dir_batch = [], [], [], [], [], [], [], []

    for idx in range(batch_size):
        index = (count*batch_size + idx) % len(user_input)
        users_batch.append(user_input[index])
        items_batch.append(item_input[index])
        user_input_batch.append(user_reviews.get(user_input[index]))
        item_input_batch.append(item_reviews.get(item_input[index]))
        user_mask_batch.append(user_masks.get(user_input[index]))
        item_mask_batch.append(item_masks.get(item_input[index]))
        labels_batch.append([ratings[index]])
        dir_batch.append(get_pos_neg(rating_thrhld, rateings[index]))

    return users_batch, items_batch, user_input_batch, item_input_batch, user_mask_batch, item_mask_batch, labels_batch, dir_batch


def conv_layer(doc_representations, conv_W, conv_b):
    doc_representations_expnd = tf.expand_dims(doc_representations, -1)
    conv = tf.nn.conv2d(doc_representations_expnd, conv_W, strides=[1, 1, word_latent_dim, 1], padding='SAME') + conv_b
    h = tf.nn.relu(conv)
    return tf.squeeze(h, -2)

def asps_gate(word_dim, contextual_words, asps_embeds, W_word, W_asps, b_gate, drop_out):
    W_word = tf.nn.dropout(W_word, drop_out)
    W_asps = tf.nn.dropout(W_asps, drop_out)
    contextual_words_reshape = tf.reshape(contextual_words, [-1, num_filters])
    asps_gate = tf.reshape(
        tf.nn.sigmoid(tf.matmul(contextual_words_reshape, W_word) + tf.matmul(asps_embeds, W_asps) + b_gate),
        [-1, word_dim, num_filters])
    gated_contextual_words = contextual_words * asps_gate
    return gated_contextual_words

def asps_prj(word_num, gated_contextual_embeds, W_prj):
    gated_contextual_embeds_reshape = tf.reshape(gated_contextual_embeds, [-1, num_filters])
    return tf.reshape(tf.matmul(gated_contextual_embeds_reshape, W_prj), [-1, word_num, num_filters])

def squash(vector):
    '''Squashing function corresponding to Eq. 1
    Args:
        vector: A tensor with shape [batch_size, 1, num_caps, vec_len, 1] or [batch_size, num_caps, vec_len, 1].
    Returns:
        A tensor with the same shape as vector but squashed in 'vec_len' dimension.
    '''
    vec_squared_norm = tf.reduce_sum(tf.square(vector), -2, keep_dims=True)
    scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm + epsilon)
    vec_squashed = scalar_factor * vector  # element-wise
    return (vec_squashed)

def simple_self_attn(word_dim, u_hat, input_masks, caps_b, b_ij, cap_1_b_inputs):
    third_dim = int(u_hat.get_shape()[2])

    u_hat = tf.reshape(u_hat, shape=[-1, word_dim, int(third_dim / num_filters), num_filters, 1])
    u_hat_stopped = tf.stop_gradient(u_hat)

    b_ij = tf.expand_dims(tf.expand_dims(cap_1_b_inputs * b_ij, -1), -1)

    for r_itr in range(itr_0):
        if (r_itr > 0):
            b_ij = b_ij * tf.expand_dims(tf.expand_dims(tf.expand_dims(input_masks, -1), -1), -1)
        c_ij = tf.nn.softmax(b_ij, dim=1)

        if r_itr == itr_0 - 1:
            s_j = tf.multiply(c_ij, u_hat)
            s_j = tf.reduce_sum(s_j, axis=1, keep_dims=True) + caps_b
            v_j = squash(s_j)

        elif r_itr < itr_0 - 1:
            s_j = tf.multiply(c_ij, u_hat_stopped)
            s_j = tf.reduce_sum(s_j, axis=1, keep_dims=True) + caps_b
            v_j = squash(s_j)

            v_j_tiled = tf.tile(v_j, [1, word_dim, 1, 1, 1])
            u_produce_v = tf.reduce_sum(u_hat_stopped * v_j_tiled, axis=3,
                                        keep_dims=True)

            b_ij += u_produce_v

    return tf.squeeze(v_j, [1, -1])

def pack_vects(vect_1, vect_2, type):
    if type == "mul":
        return tf.multiply(vect_1, vect_2)
    elif type == "sub":
        return tf.subtract(vect_1, vect_2)
    elif type == "both":
        return tf.concat([tf.multiply(vect_1, vect_2), tf.subtract(vect_1, vect_2)], 2)


#Bi-Agreement based Capsule
def caps_layer_2(num_latent, input_vects, caps_W, caps_b, b_ij, cap_1_b_inputs):
    third_dim = int(caps_W.get_shape()[2])
    input_vects_exp = tf.expand_dims(tf.expand_dims(input_vects, 2), -1)
    input_vects_exp = tf.tile(input_vects_exp, [1, 1, third_dim, 1, 1])

    u_hat = tf.reduce_sum(caps_W * input_vects_exp, axis=3, keep_dims=True)
    u_hat = tf.reshape(u_hat, shape=[-1, num_latent, int(third_dim / latent_dim), latent_dim, 1])

    u_hat_stopped = tf.stop_gradient(u_hat)
    b_ij = tf.expand_dims(tf.expand_dims(cap_1_b_inputs * b_ij, -1), -1)

    for r_itr in range(itr_1):
        c_ij_intra = tf.nn.softmax(b_ij, dim=1)
        c_ij_inter = tf.nn.softmax(b_ij, dim=2)
        c_mul = tf.sqrt(tf.multiply(c_ij_intra, c_ij_inter))
        c_sum = tf.reduce_sum(c_mul, axis=1, keep_dims=True)
        c_ij = c_mul / (c_sum + _EPSILON)


        if r_itr == itr_1 - 1:
            s_j = tf.multiply(c_ij, u_hat)
            s_j = tf.reduce_sum(s_j, axis=1, keep_dims=True) + caps_b
            v_j = squash(s_j)

        elif r_itr < itr_1 - 1:
            s_j = tf.multiply(c_ij, u_hat_stopped)
            s_j = tf.reduce_sum(s_j, axis=1, keep_dims=True) + caps_b
            v_j = squash(s_j)

            v_j_tiled = tf.tile(v_j, [1, num_latent, 1, 1, 1])
            u_produce_v = tf.reduce_sum(u_hat_stopped * v_j_tiled, axis=3,
                                        keep_dims=True)
            b_ij += u_produce_v

    return tf.squeeze(v_j, [1, -1])

#Vanilla Routing based Capsule
def caps_layer_1(num_latent, input_vects, caps_W, caps_b, b_ij, cap_1_b_inputs):
    third_dim = int(caps_W.get_shape()[2])
    input_vects_exp = tf.expand_dims(tf.expand_dims(input_vects, 2), -1)
    input_vects_exp = tf.tile(input_vects_exp, [1, 1, third_dim, 1, 1])

    u_hat = tf.reduce_sum(caps_W * input_vects_exp, axis=3, keep_dims=True)
    u_hat = tf.reshape(u_hat, shape=[-1, num_latent, int(third_dim / latent_dim), latent_dim, 1])

    u_hat_stopped = tf.stop_gradient(u_hat)
    b_ij = tf.expand_dims(tf.expand_dims(cap_1_b_inputs * b_ij, -1), -1)

    for r_itr in range(itr_1):
        c_ij = tf.nn.softmax(b_ij, dim=2)

        if r_itr == itr_1 - 1:
            s_j = tf.multiply(c_ij, u_hat)
            s_j = tf.reduce_sum(s_j, axis=1, keep_dims=True) + caps_b
            v_j = squash(s_j)

        elif r_itr < itr_1 - 1:
            s_j = tf.multiply(c_ij, u_hat_stopped)
            s_j = tf.reduce_sum(s_j, axis=1, keep_dims=True) + caps_b
            v_j = squash(s_j)

            v_j_tiled = tf.tile(v_j, [1, num_latent, 1, 1, 1])
            u_produce_v = tf.reduce_sum(u_hat_stopped * v_j_tiled, axis=3,
                                        keep_dims=True)
            b_ij += u_produce_v

    return tf.squeeze(v_j, [1, -1])

def highway(W_trans, b_trans, W_gate, b_gate, embed, dropout, on_gate):
    W_trans = tf.nn.dropout(W_trans, dropout)
    high_embed = tf.nn.tanh(tf.matmul(embed, W_trans) + b_trans)
    if not on_gate:
        print("gate is closed")
        return high_embed
    W_gate = tf.nn.dropout(W_gate, dropout)
    gate = tf.sigmoid(tf.matmul(embed, W_gate) + b_gate)
    return tf.multiply(high_embed, gate) + tf.multiply(embed, (1 - gate))

def cal_degree(embedding, W, b, dropout_rate):
    W = tf.nn.dropout(W, dropout_rate)
    predict_rate = (tf.matmul(embedding, W) + b)
    return predict_rate

def rescale_sigmoid(point, low_bound, up_bound):
    return low_bound + tf.sigmoid(point) * (up_bound - low_bound)

def train_model():
    users = tf.placeholder(tf.int32, shape=[None])
    items = tf.placeholder(tf.int32, shape=[None])
    users_inputs = tf.placeholder(tf.int32, shape=[None, max_doc_length])
    items_inputs = tf.placeholder(tf.int32, shape=[None, max_doc_length])
    users_masks_inputs = tf.placeholder(tf.float32, shape=[None, max_doc_length])
    items_masks_inputs = tf.placeholder(tf.float32, shape=[None, max_doc_length])
    ratings = tf.placeholder(tf.float32, shape=[None, 1])
    labels_input = tf.placeholder(tf.int32, shape=[None])
    cap_1_b_inputs = tf.placeholder(tf.float32, shape=[None, 1, 1])
    cap_2_b_inputs = tf.placeholder(tf.float32, shape=[None, 1, 2])
    T_c = tf.one_hot(labels_input, depth=2, axis=-1, on_value=0.0, off_value=1.0, dtype=tf.float32)
    dropout_rate = tf.placeholder(tf.float32)

    text_embedding = tf.Variable(word_embedding_mtrx, dtype=tf.float32)
    padding_embedding = tf.Variable(np.zeros([1, word_latent_dim]), dtype=tf.float32)

    word_embeddings = tf.concat([text_embedding, padding_embedding], 0)

    # interaction embedding
    user_bias = tf.Variable(tf.random_normal([num_users, 1], mean=0, stddev=0.02))
    item_bias = tf.Variable(tf.random_normal([num_items, 1], mean=0, stddev=0.02))
    user_bs = tf.nn.embedding_lookup(user_bias, users)
    item_bs = tf.nn.embedding_lookup(item_bias, items)

    # embedding layer
    user_reviews_representation = tf.nn.embedding_lookup(word_embeddings, users_inputs)
    item_reviews_representation = tf.nn.embedding_lookup(word_embeddings, items_inputs)

    user_reviews_representation = user_reviews_representation * tf.expand_dims(users_masks_inputs, -1)
    item_reviews_representation = item_reviews_representation * tf.expand_dims(items_masks_inputs, -1)

    # CNN layer
    W_conv_u = tf.Variable(
        tf.truncated_normal([window_size, word_latent_dim, 1, num_filters], stddev=0.3))
    W_conv_u = tf.nn.dropout(W_conv_u, dropout_rate)
    b_conv_u = tf.Variable(tf.constant(0.1, shape=[num_filters]))
    W_conv_i = tf.Variable(tf.truncated_normal([window_size, word_latent_dim, 1, num_filters], stddev=0.3))
    W_conv_i = tf.nn.dropout(W_conv_i, dropout_rate)
    b_conv_i = tf.Variable(tf.constant(0.1, shape=[num_filters]))

    user_contextual_embeds = conv_layer(user_reviews_representation, W_conv_u, b_conv_u)
    item_contextual_embeds = conv_layer(item_reviews_representation, W_conv_i, b_conv_i)

    user_word_dim = int(user_contextual_embeds.get_shape()[1])
    item_word_dim = int(item_contextual_embeds.get_shape()[1])

    #gate mechanism
    user_viewpoint_embeddings = [
        tf.Variable(tf.truncated_normal([1, num_filters], stddev=0.3), name="user_aspect_embedding_{}".format(i)) for i
        in range(num_aspect)]
    item_aspect_embeddings = [
        tf.Variable(tf.truncated_normal([1, num_filters], stddev=0.3), name="user_aspect_embedding_{}".format(i)) for i
        in range(num_aspect)]

    W_word_gate_u = [
        tf.Variable(tf.truncated_normal([num_filters, num_filters], stddev=0.1), name="user_word_gate_u_{}".format(i))
        for i in range(num_aspect)]
    W_asps_gate_u = [
        tf.Variable(tf.truncated_normal([num_filters, num_filters], stddev=0.1), name="user_asps_gate_u_{}".format(i))
        for i in range(num_aspect)]
    W_word_gate_i = [
        tf.Variable(tf.truncated_normal([num_filters, num_filters], stddev=0.1), name="item_word_gate_i_{}".format(i))
        for i in range(num_aspect)]
    W_asps_gate_i = [
        tf.Variable(tf.truncated_normal([num_filters, num_filters], stddev=0.1), name="item_asps_gate_i_{}".format(i))
        for i in range(num_aspect)]
    b_gate_u = [tf.Variable(tf.constant(0.1, shape=[num_filters]), name="gate_b_u_{}".format(i)) for i in
                range(num_aspect)]
    b_gate_i = [tf.Variable(tf.constant(0.1, shape=[num_filters]), name="gate_b_i_{}".format(i)) for i in
                range(num_aspect)]

    user_pre_attn_contextual_embeds, item_pre_attn_contextual_embeds = [], []
    for i in range(num_aspect):
        user_pre_attn_contextual_embeds.append(
            asps_gate(user_word_dim, user_contextual_embeds, user_viewpoint_embeddings[i], W_word_gate_u[i],
                      W_asps_gate_u[i], b_gate_u[i], dropout_rate))
        item_pre_attn_contextual_embeds.append(
            asps_gate(item_word_dim, item_contextual_embeds, item_aspect_embeddings[i], W_word_gate_i[i],
                      W_asps_gate_i[i], b_gate_i[i], dropout_rate))

    # self-attn layer
    W_prj_u = [tf.Variable(tf.truncated_normal([num_filters, num_filters], mean=0, stddev=0.3),
                            name="user_project_W_{}".format(i)) for i in range(num_aspect)]
    b_prj_u = [tf.Variable(tf.constant(0.1, shape=[1, 1, 1, num_filters, 1]), name="user_project_b_{}".format(i)) for i in
               range(num_aspect)]
    b_ij_u = [tf.constant(np.zeros([1, max_doc_length, 1], dtype=np.float32), name="user_bij_{}".format(i)) for i in
              range(num_aspect)]

    W_prj_i = [tf.Variable(tf.truncated_normal([num_filters, num_filters], mean=0, stddev=0.3),
                            name="item_project_W_{}".format(i)) for i in range(num_aspect)]
    b_prj_i = [tf.Variable(tf.constant(0.1, shape=[1, 1, 1, num_filters, 1]), name="item_project_b_{}".format(i)) for i in
               range(num_aspect)]
    b_ij_i = [tf.constant(np.zeros([1, max_doc_length, 1], dtype=np.float32), name="item_bij_{}".format(i)) for i in
              range(num_aspect)]

    user_viewpoint_words = []
    item_viewpoint_words = []
    for i in range(num_aspect):
        user_viewpoint_words.append(asps_prj(user_word_dim, user_pre_attn_contextual_embeds[i], W_prj_u[i]))
        item_viewpoint_words.append(asps_prj(item_word_dim, item_pre_attn_contextual_embeds[i], W_prj_i[i]))

    user_viewpoint_embeddings = []
    item_aspect_embeddings = []
    for i in range(len(W_prj_u)):
        u_viewpoint_v = simple_self_attn(user_word_dim, user_viewpoint_words[i], users_masks_inputs,
                                                        b_prj_u[i], b_ij_u[i], cap_1_b_inputs)
        user_viewpoint_embeddings.append(u_viewpoint_v)
    for i in range(len(W_prj_i)):
        i_asps_v = simple_self_attn(item_word_dim, item_viewpoint_words[i], items_masks_inputs,
                                                        b_prj_i[i], b_ij_i[i], cap_1_b_inputs)
        item_aspect_embeddings.append(i_asps_v)

    #pack logic unit
    aspect_embeds_pack = []
    for i in range(num_aspect):
        user_asps_embed = user_viewpoint_embeddings[i]
        for j in range(num_aspect):
            item_asps_embed = item_aspect_embeddings[j]
            aspect_embeds_pack.append(pack_vects(user_asps_embed, item_asps_embed, "both"))
    logic_units = tf.concat(aspect_embeds_pack, 1)
    num_intrn = int(logic_units.get_shape()[1])
    cap1_latent_dim = int(logic_units.get_shape()[-1])

    #sentiment capsule
    W_caps_sent = tf.Variable(tf.truncated_normal([1, num_intrn, latent_dim * 2, cap1_latent_dim, 1], mean=0, stddev=0.3))
    b_caps_sent = tf.Variable(tf.constant(0.1, shape=[1, 1, 2, latent_dim, 1]))
    b_ij_sent = tf.constant(np.zeros([1, num_aspect * num_aspect, 2], dtype=np.float32))
    v_sent = caps_layer_2(num_intrn, logic_units, W_caps_sent, b_caps_sent, b_ij_sent, cap_2_b_inputs)

    caps_len = tf.squeeze(tf.sqrt(tf.reduce_sum(tf.square(v_sent), axis=-1, keep_dims=True) + epsilon), -1)

    max_p = tf.square(tf.maximum(0., 0.8 - caps_len))
    max_n = tf.square(tf.maximum(0., caps_len - 0.2))
    L_c = T_c * max_p + lambda_1 * (1 - T_c) * max_n
    margin_loss = tf.reduce_mean(tf.reduce_sum(L_c, axis=1))#sentiment classification

    pos_mask = tf.expand_dims(tf.constant([[1.0], [0.0]]), 0)
    neg_mask = tf.expand_dims(tf.constant([[0.0], [1.0]]), 0)

    #get sentiment probability
    v_pos = tf.reduce_sum(v_sent * pos_mask, axis=1)
    pos_len = tf.reduce_sum(caps_len * tf.squeeze(pos_mask, -1), axis=1, keep_dims=True)
    v_neg = tf.reduce_sum(v_sent * neg_mask, axis=1)
    neg_len = tf.reduce_sum(caps_len * tf.squeeze(neg_mask, -1), axis=1, keep_dims=True)

    #highway layer
    W_trans_pos = tf.Variable(tf.truncated_normal([latent_dim, latent_dim], stddev=0.3), name="W_trans_pos".format(i))
    b_trans_pos = tf.Variable(tf.constant(0.1, shape=[latent_dim]), name="b_trans_pos".format(i))
    W_high_pos = tf.Variable(tf.truncated_normal([latent_dim, latent_dim], stddev=0.3), name="W_high_pos".format(i))
    b_high_pos = tf.Variable(tf.constant(0.1, shape=[latent_dim]), name="b_high_pos".format(i))

    abs_vect_pos = highway(W_trans_pos, b_trans_pos, W_high_pos, b_high_pos, v_pos, dropout_rate, True)

    W_trans_neg = tf.Variable(tf.truncated_normal([latent_dim, latent_dim], stddev=0.3), name="W_trans_neg".format(i))
    b_trans_neg = tf.Variable(tf.constant(0.1, shape=[latent_dim]), name="b_trans_neg".format(i))
    W_high_neg = tf.Variable(tf.truncated_normal([latent_dim, latent_dim], stddev=0.3), name="W_high_neg".format(i))
    b_high_neg = tf.Variable(tf.constant(0.1, shape=[latent_dim]), name="b_high_neg".format(i))

    abs_vect_neg = highway(W_trans_neg, b_trans_neg, W_high_neg, b_high_neg, v_neg, dropout_rate, True)

    W = [tf.Variable(tf.truncated_normal([latent_dim, 1], stddev=0.3), name="W_{}".format(i)) for i in range(2)]
    b_0 = tf.Variable(tf.constant(3.0, shape=[1]), name="b_0")
    b_1 = tf.Variable(tf.constant(1.0, shape=[1]), name="b_1")

    #get sentiment degree
    predict_pos_rate = cal_degree(abs_vect_pos, W[0], b_0, dropout_rate)
    predict_neg_rate = cal_degree(abs_vect_neg, W[1], b_1, dropout_rate)
    predict_ratings = tf.concat([predict_pos_rate, predict_neg_rate], 1)
    predict_rating = rescale_sigmoid(pos_len * predict_pos_rate - neg_len * predict_neg_rate, 1.0,
                                     5.0) + user_bs + item_bs #predict rating based on a rescaled sigmoid function

    loss = tf.reduce_mean(tf.squared_difference(predict_rating, ratings))#rating prediction
    total_loss = gama * loss + (1 - gama) * margin_loss #multi-task learning fusion

    train_step = tf.train.RMSPropOptimizer(learn_rate).minimize(total_loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for e in range(epochs):
            t = time()
            loss_total = 0.0
            margin_total = 0.0
            count = 0.0
            for i in range(int(math.ceil(len(user_input) / float(batch_size)))):
                user_batch, item_batch, user_input_batch, item_input_batch, user_mask_batch, item_mask_batch, rates_batch, labels_batch = get_train_instance_batch(
                    i, batch_size, user_input, item_input, rateings,
                    user_reviews, item_reviews, user_masks, item_masks)
                _, loss_val, margin_loss_val = sess.run([train_step, loss, margin_loss],
                                                             feed_dict={users: user_batch, items: item_batch,
                                                                        users_inputs: user_input_batch,
                                                                        items_inputs: item_input_batch,
                                                                        users_masks_inputs: user_mask_batch,
                                                                        items_masks_inputs: item_mask_batch,
                                                                        ratings: rates_batch,
                                                                        labels_input: labels_batch,
                                                                        cap_1_b_inputs: cap_1_b_train,
                                                                        cap_2_b_inputs: cap_2_b_train,
                                                                        dropout_rate: drop_out})
                loss_total += loss_val
                margin_total += margin_loss_val
                count += 1.0
            t1 = time()
            mses, maes = [], []
            for i in range(len(user_input_test)):
                cap_1_b_test = ini_cap(len(user_input_test[i]), 1, 1)
                cap_2_b_test = ini_cap(len(user_input_test[i]), 1, 2)
                eval_model(users, items, users_inputs, items_inputs, users_masks_inputs, items_masks_inputs,
                           dropout_rate, cap_1_b_inputs, cap_2_b_inputs, predict_rating, sess, user_test[i],
                           item_test[i], user_input_test[i],
                           item_input_test[i], user_mask_test[i], item_mask_test[i], rating_input_test[i], cap_1_b_test,
                           cap_2_b_test, mses, maes)
            mse = np.array(mses).mean()
            mae = np.array(maes).mean()
            t2 = time()
            print("epoch%d  time: %.3fs  loss: %.3f  margin_loss: %.3f  test time: %.3fs  mse: %.3f  mae: %.3f" % (e, (t1 - t), loss_total / count, margin_total / count, (t2 - t1), mse, mae))

def eval_model(users, items, users_inputs, items_inputs, users_masks_inputs, items_masks_inputs, dropout_rate,
               cap_1_b_inputs, cap_2_b_inputs, predict_rating, sess, user_tests, item_tests, user_input_tests,
               item_input_tests, user_mask_tests, item_mask_tests, rate_tests, cap_1_b_test, cap_2_b_test, rmses, maes):
    predicts = sess.run(predict_rating, feed_dict={users: user_tests, items: item_tests, users_inputs: user_input_tests,
                                                   items_inputs: item_input_tests, users_masks_inputs: user_mask_tests,
                                                   items_masks_inputs: item_mask_tests, cap_1_b_inputs: cap_1_b_test,
                                                   cap_2_b_inputs: cap_2_b_test, dropout_rate: 1.0})
    row, col = predicts.shape
    for r in range(row):
        rmses.append(pow((predicts[r, 0] - rate_tests[r][0]), 2))
        maes.append(abs(predicts[r, 0] - rate_tests[r][0]))
    return rmses, maes


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    latent_dim = 25#denote as k
    word_latent_dim = 300#denote as d in paper
    num_filters = 50#number of filters of convolution operation
    window_size = 3#denote as c in paper
    max_doc_length = 300
    learn_rate = 0.001
    epochs = 150
    batch_size = 100
    num_aspect = 5#denote as M in paper
    rating_thrhld = 3.0#the thrhld which regard the lower and equal rating as negative
    _EPSILON = 10e-9
    itr_0 = 2#number of iteration of self attention, we use 2
    itr_1 = 3#denote as tao, the number of iteration of Dynamic Routing
    lambda_1 = 0.8
    gama = 0.5#denote as λ in paper
    drop_out = 0.9

    # loading data
    firTime = time()
    dataSet = Dataset(max_doc_length, "/the parent directory of the training files/")
    word_dict, user_reviews, item_reviews, user_masks, item_masks, train, testRatings = dataSet.word_id_dict, dataSet.userReview_dict, dataSet.itemReview_dict, dataSet.userMask_dict, dataSet.itemMask_dict, dataSet.trainMtrx, dataSet.testRatings
    secTime = time()

    num_users, num_items = train.shape
    print "load data: %.3fs" % (secTime - firTime)
    print num_users, num_items

    word_embedding_mtrx = ini_word_embed(len(word_dict), word_latent_dim)

    # get train instances
    user_input, item_input, rateings = get_train_instance(train)
    #get test batch instances
    user_test, item_test, user_input_test, item_input_test, user_mask_test, item_mask_test, rating_input_test = get_test_list_mask(
        200, testRatings, user_reviews, item_reviews, user_masks, item_masks)

    cap_1_b_train = ini_cap(batch_size, 1, 1)
    cap_2_b_train = ini_cap(batch_size, 1, 2)

    train_model()