import tensorflow as tf
import numpy as np


def read_data():
    t_train = np.load('Data/t_train.npy')
    s_train = np.load('Data/s_train.npy')
    y_train = np.load('Data/y_train.npy')
    t_val = np.load('Data/t_val.npy')
    s_val = np.load('Data/s_val.npy')
    y_val = np.load('Data/y_val.npy')
    return t_train, s_train, y_train, t_val, s_val, y_val


def build_graph(batch_size=10, feature_size=20, time_series=7, scalar_size=2):
    t = tf.placeholder(tf.float32, [batch_size, feature_size, time_series])
    s = tf.placeholder(tf.float32, [batch_size, scalar_size])
    y = tf.placeholder(tf.float32, [batch_size, 1])

    t_trans = tf.transpose(t, [2, 0, 1])

    t_gru_cell = tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(20, )
    initial_state = t_gru_cell.zero_state(batch_size, dtype=tf.float32)
    t_output, state = tf.nn.dynamic_rnn(t_gru_cell, t_trans, initial_state=initial_state, time_major=True,
                                        dtype=tf.float32)
    t_output = t_output[-1, :, :]
    # batch, feature
    merge = tf.concat([t_output, s], 1)

    # batch, feature
    dense_1 = tf.layers.dense(merge, 64, tf.nn.leaky_relu, kernel_initializer=tf.initializers.random_normal)
    # dense_2 = tf.layers.dense(dense_1, 32, tf.nn.leaky_relu, kernel_initializer=tf.initializers.random_normal())
    output_layer = tf.layers.dense(dense_1, 1, kernel_initializer=tf.initializers.random_normal)

    loss = tf.norm(output_layer - y)
    opt = tf.train.AdamOptimizer()
    update = opt.minimize(loss)

    tf.summary.scalar('loss', loss)

    log_op = tf.summary.merge_all()
    return output_layer, loss, log_op, update, {'t': t, 's': s, 'y': y}


class data_flow():
    def __init__(self, data_t, data_s, data_y, placeholder):
        self.t = data_t
        self.s = data_s
        self.y = data_y
        self.p = placeholder
        self.state = 0

    def select_batch(self, size=10):
        batch_ind = np.random.choice(np.arange(self.t.shape[0]), size)
        feed_dict = {self.p['t']: self.t[batch_ind], self.p['s']: self.s[batch_ind], self.p['y']: self.y[batch_ind]}
        return feed_dict

    def seqeuence_batch(self, size=10):
        valid = True if self.state+size <= self.t.shape[0] else False
        batch_ind = np.arange(self.state, self.state + size) if valid else np.arange(self.state, self.t.shape[0])
        if not valid:
            batch_ind = np.concatenate((batch_ind, np.arange(0, size - batch_ind.size)), axis=0)
        self.state += size
        feed_dict = {self.p['t']: self.t[batch_ind], self.p['s']: self.s[batch_ind], self.p['y']: self.y[batch_ind]}
        return feed_dict, valid


def train(sess, loss_op, log_op, update_op, logger, feed_dict, eps=10000, saver=None, save_path=None):
    for i in range(int(eps)):
        loss_total = 0
        loss, _ = sess.run([loss_op, update_op], feed_dict=feed_dict)
        loss_total += loss
        if i % 100 == 0:
            loss, log, _ = sess.run([loss_op, log_op, update_op], feed_dict=feed_dict)
            loss_total += loss
            logger.add_summary(log, i)
            print("EPS: {:5d}, loss: {:.4f}".format(i, loss))
            if saver is not None:
                saver.save(sess, save_path)
    return loss_total / data.t.shape[0]


def test(sess, loss_op, log_op, logger, data):
    feed_dict, valid = data.seqeuence_batch(10)
    loss_total = 0
    i = 0
    while valid:
        i += 1
        loss, log = sess.run([loss_op, log_op], feed_dict=feed_dict)
        loss_total += loss
        logger.add_summary(log, i)
        print("Val_loss: {:.4f}".format(loss))
        feed_dict, valid = data.seqeuence_batch(10)
    return loss_total / data.t.shape[0]


if __name__ == '__main__':
    t_train, s_train, y_train, t_val, s_val, y_val = read_data()
    save_path = '/model/sub-2.ckpt'
    batch_size = 10
    time_series_feature = t_train.shape[1]
    time_series_length = t_train.shape[2]
    scalar_feature = s_train.shape[1]

    # train
    eps = 20000
    tf.reset_default_graph()
    output_all, loss_all, log_all, update_all, placeholder = build_graph(batch_size, time_series_feature, time_series_length, scalar_feature)
    data = data_flow(t_train, s_train, y_train, placeholder)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        logger = tf.summary.FileWriter('Result/train', sess.graph)
        train(sess, loss_all, log_all, update_all, logger, eps=int(eps), feed_dict=data.select_batch(10), saver=saver,
              save_path=save_path)
    #
    # test
    tf.reset_default_graph()
    _, loss_all, log_all, _, placeholder = build_graph(batch_size, time_series_feature, time_series_length, scalar_feature)
    data = data_flow(t_train, s_train, y_train, placeholder)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, save_path)
        logger = tf.summary.FileWriter('Result/test', sess.graph)
        avg_loss = test(sess, loss_all, log_all, logger, data)
        print(avg_loss)

    # val
    tf.reset_default_graph()
    _, loss_all, log_all, _, placeholder = build_graph(batch_size, time_series_feature, time_series_length, scalar_feature)
    data = data_flow(t_val, s_val, y_val, placeholder)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, save_path)
        logger = tf.summary.FileWriter('Result/val', sess.graph)
        avg_loss = test(sess, loss_all, log_all, logger, data)
        print(avg_loss)
