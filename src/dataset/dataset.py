"""
Created on April 1, 2020
Processing datasets.
@author Felice Antonio Merra (felice.merra@poliba.it)
Inspired by @author: Xiangnan He (xiangnanhe@gmail.com)
"""
import scipy.sparse as sp
import numpy as np
from multiprocessing import Pool
from multiprocessing import cpu_count
from scipy.sparse import dok_matrix

np.random.seed(0)


class DataLoader(object):
    """
    Load train and test dataset
    """

    def __init__(self, path_train_data, path_test_data):
        """
        Constructor of DataLoader
        :param path_train_data: relative path for train file
        :param path_test_data: relative path for test file
        """
        self.load_train_file(path_train_data)
        self.load_train_file_as_list(path_train_data)
        self.load_test_file(path_test_data)

    def load_train_file(self, filename):
        """
        Read /data/dataset_name/train file and Return the matrix.
        """
        # Get number of users and items
        self.num_users, self.num_items = 0, 0
        with open(filename, "r") as f:
            line = f.readline()
            while line is not None and line != "":
                arr = line.split("\t")
                u, i = int(arr[0]), int(arr[1])
                self.num_users = max(self.num_users, u)
                self.num_items = max(self.num_items, i)
                line = f.readline()

        # Construct URM
        self.train = sp.dok_matrix((self.num_users + 1, self.num_items + 1), dtype=np.float32)
        with open(filename, "r") as f:
            line = f.readline()
            while line is not None and line != "":
                arr = line.split("\t")
                user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
                if rating > 0:
                    self.train[user, item] = 1.0
                line = f.readline()

        self.num_users = self.train.shape[0]
        self.num_items = self.train.shape[1]

    def load_train_file_as_list(self, filename):
        # Get number of users and items
        u_ = 0
        self.train_list, items = [], []
        with open(filename, "r") as f:
            line = f.readline()
            index = 0
            while line is not None and line != "":
                arr = line.split("\t")
                u, i = int(arr[0]), int(arr[1])
                if u_ < u:
                    index = 0
                    self.train_list.append(items)
                    items = []
                    u_ += 1
                index += 1
                items.append(i)
                line = f.readline()
        self.train_list.append(items)

    def load_test_file(self, filename):
        self.test = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item = int(arr[0]), int(arr[1])
                self.test.append([user, item])
                line = f.readline()

    def sampling(self):
        _user_input, _item_input_pos = [], []
        for (u, i) in self.train.keys():
            # positive instance
            _user_input.append(u)
            _item_input_pos.append(i)
        return _user_input, _item_input_pos

    def shuffle(self, batch_size=512):
        """
        Shuffle dataset to create batch with batch size
        :param batch_size: default 512
        :return: set of all generated random batches
        """
        self._batch_size = batch_size
        self._user_input, self._item_input_pos = self.sampling()
        self._index = range(len(self._user_input))

        _num_batches = len(self._user_input) // self._batch_size
        pool = Pool(cpu_count() - 1)
        res = pool.map(self._get_train_batch, range(_num_batches))
        pool.close()
        pool.join()

        user_input = [r[0] for r in res]
        item_input_pos = [r[1] for r in res]
        item_input_neg = [r[2] for r in res]
        return user_input, item_input_pos, item_input_neg

    def _get_train_batch(self, i):
        """
        Generation of a batch in multiprocessing
        :param i: index to control the batch generayion
        :return:
        """
        user_batch, item_pos_batch, item_neg_batch = [], [], []
        begin = i * self._batch_size
        for idx in range(begin, begin + self._batch_size):
            user_batch.append(self._user_input[self._index[idx]])
            item_pos_batch.append(self._item_input_pos[self._index[idx]])
            j = np.random.randint(self.num_items)
            while j in self.train[self._user_input[self._index[idx]]]:
                j = np.random.randint(self.num_items)
            item_neg_batch.append(j)
        return np.array(user_batch)[:, None], np.array(item_pos_batch)[:, None], np.array(item_neg_batch)[:, None]
