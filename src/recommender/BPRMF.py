import numpy as np
from time import time
from recommender.Evaluator import Evaluator
import os
import logging

from util.read import find_checkpoint

np.random.seed(0)
logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from recommender.RecommenderModel import RecommenderModel

print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))


class BPRMF(RecommenderModel):

    def __init__(self, data, path_output_rec_result, path_output_rec_weight, args):
        super(BPRMF, self).__init__(data, path_output_rec_result, path_output_rec_weight, args.rec)
        self.embedding_size = args.embed_size
        self.learning_rate = args.lr
        self.reg = args.reg
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.verbose = args.verbose
        self.restore_epochs = args.restore_epochs
        self.evaluator = Evaluator(self, data, args.k)

        self.embedding_P = tf.Variable(
            tf.random.truncated_normal(shape=[self.num_users, self.embedding_size], mean=0.0, stddev=0.01),
            name='embedding_P', dtype=tf.dtypes.float32)  # (users, embedding_size)
        self.embedding_Q = tf.Variable(
            tf.random.truncated_normal(shape=[self.num_items, self.embedding_size], mean=0.0, stddev=0.01),
            name='embedding_Q', dtype=tf.dtypes.float32)  # (items, embedding_size)
        self.h = tf.constant(1.0, tf.float32, [self.embedding_size, 1], name="h")

        self.optimizer = tf.keras.optimizers.Adagrad(learning_rate=self.learning_rate)

    def get_inference(self, user_input, item_input_pos):
        self.embedding_p = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_P, user_input), 1)
        self.embedding_q = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_Q, item_input_pos), 1)

        return tf.matmul(self.embedding_p * self.embedding_q,
                         self.h), self.embedding_p, self.embedding_q  # (b, embedding_size) * (embedding_size, 1)

    def get_full_inference(self):
        return tf.matmul(self.embedding_P, tf.transpose(self.embedding_Q))

    def _train_step(self, batches):
        user_input, item_input_pos, item_input_neg = batches

        for batch_idx in range(len(user_input)):
            with tf.GradientTape() as t:
                t.watch([self.embedding_P, self.embedding_Q])

                self.output_pos, embed_p_pos, embed_q_pos = self.get_inference(user_input[batch_idx],
                                                                               item_input_pos[batch_idx])
                self.output_neg, embed_p_neg, embed_q_neg = self.get_inference(user_input[batch_idx],
                                                                               item_input_neg[batch_idx])
                self.result = tf.clip_by_value(self.output_pos - self.output_neg, -80.0, 1e8)

                # self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

                loss = tf.reduce_sum(tf.nn.softplus(-self.result)) + self.reg * tf.reduce_mean(
                    tf.square(embed_p_pos) + tf.square(embed_q_pos) + tf.square(embed_q_neg))

            gradients = t.gradient(loss, [self.embedding_P, self.embedding_Q])
            self.optimizer.apply_gradients(zip(gradients, [self.embedding_P, self.embedding_Q]))

    def train(self):

        saver_ckpt = tf.train.Checkpoint(optimizer=self.optimizer, model=self)

        if self.restore_epochs > 1:
            # Restore the model at the args
            # saver_ckpt.restore(tf.train.latest_checkpoint(self.path_output_rec_weight))
            # TODO
            # We should pass the basic model
            try:
                checkpoint_file = find_checkpoint(self.path_output_rec_weight, self.restore_epochs, self.rec)
                saver_ckpt.restore(checkpoint_file)
            except:
                self.restore_epochs = 1
                print("Training from scratch...")

        for epoch in range(self.restore_epochs, self.epochs + 1):
            start = time()
            batches = self.data.shuffle(self.batch_size)
            self._train_step(batches)

            if epoch % self.verbose == 0 or epoch == 1:
                saver_ckpt.save('{0}/weights-{1}'.format(self.path_output_rec_weight, epoch))
                results = self.evaluator.eval()
                hr, ndcg, auc = np.swapaxes(results, 0, 1)[-1]
                print("Epoch %d\tHR: %.4f\tnDCG: %.4f\tAUC: %.4f [Sec %.2f]" % (epoch, hr, ndcg, auc, time() - start))
                start = time()

        self.evaluator.store_recommendation()
