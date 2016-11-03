# -*- coding: utf-8 -*-
import numpy as np
import random
BATCH_SIZE = 100

EPOCH_SIZE = 100

class Dataset:
    def __init__(self,x,y_):
        self.x              = x # [3000,300] みたいな input data
        self.y_             = y_ # [3000,19] みたいな label data
        self.badge_lst      = range(0,len(self.x)) # インデックスのリストをつくる
        self.max_index      = len(self.badge_lst) # 3000このデータです
        # random mini badegeにしたかったらここをアンコメント
        #random.shuffle(self.badge_lst)

        self.batch_size     = BATCH_SIZE # ばっじさいずを指定
        self.start_index    = 0
        self.number_of_iter = (self.max_index / self.batch_size) if (self.max_index % self.batch_size == 0) else (self.max_index/self.batch_size)+1 # 3000個をバッジサイズで割る(あまりがあったら+1する)
        self.batch_indices  = [] # 学習したいインデックスを入れる(ミニバッジのこと)
        # self.epoch_times    = 0 # 何回データを総なめするか

        print ("x shape is...\t"),
        print (self.x.shape)
        print ("y_ shape is...\t"),
        print (self.y_.shape)

    def get_next_minibatch(self):
        if (self.max_index <= (self.start_index + self.batch_size)) or (self.number_of_iter == 1):
            self.batch_indices  = self.badge_lst[self.start_index:]
            self.start_index = 0
            self.badge_lst = range(0,self.max_index)
            #random.shuffle(self.badge_lst)
        else:
            self.batch_indices = self.badge_lst[self.start_index:self.start_index+self.batch_size]
            self.start_index += self.batch_size
        return self.x[self.batch_indices], self.y_[self.batch_indices]

    def get_all(self):
        return self.x, self.y_


'''
if __name__ == "__main__":
    x     = np.arange(900000).reshape(3000,300) # training input data
    y_    = np.arange(57000).reshape(3000,19) # test input data
    aiueo = Dataset(x, y_)

    for i in range(10):#EPOCH_SIZE:
        for s in (1,aiueo.number_of_iter):
            x, y_ = aiueo.get_next_minibatch()
            print (y_ )
'''
