# -*- coding:utf-8 -*-

import tensorflow as tf

model_path = '../runs/gp_dsa/1555228636/checkpoints'

class Sim_Model(object):
    def __init__(self):
        graph = tf.Graph()
        with graph.as_default():
            session_conf = tf.ConfigProto(allow_soft_placement=True)
            self.sess = tf.Session(config=session_conf)
            with self.sess.as_default():
                checkpoint_file = tf.train.latest_checkpoint(model_path)
                saver = tf.train.import_meta_graph(checkpoint_file + '.meta')
                saver.restore(self.sess, checkpoint_file)

                self.input_left = graph.get_tensor_by_name('input_left:0')  # [batch, max_len]
                self.input_right = graph.get_tensor_by_name('input_right:0')
                self.keep_prob = graph.get_tensor_by_name('dropout_keep_prob:0')
                self.scores = graph.get_tensor_by_name('output/scores:0')  # [batch, 2]


    def inference(self, query, docs):
        # 定义你的输入输出以及计算图

        scores_ = self.sess.run(self.scores, feed_dict={self.input_left:query, self.input_right:docs,
                                                   self.keep_prob:1.0})
        return scores_[:, -1].tolist()

if __name__ == '__main__':
    input_left = [[1, 2, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    input_right = [[1, 2, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    model = Sim_Model()

    result = model.inference(input_left, input_right)

    print(result.tolist())

