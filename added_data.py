

# -*- coding: utf-8 -*-
import numpy as np

def main(word2vec_models,data_list,knn):
    #data_list=[(['Still', 'getting', 'use', 'to', 'Window', '8', '.'], np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])),(['Still', 'getting', 'use', 'to', 'Window', '8', '.'], np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])),['Still', 'getting', 'use', 'to', 'Window', '8', '.'], np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])]
    #data_tuple=[(['Still', 'getting', 'use', 'to', 'Window', '8', '.'], np.array([0, 1])),(['Still', 'getting', 'use', 'to', 'Window', '8', '.'], np.array([0, 1])),(['Still', 'getting', 'use', 'to', 'Window', '8', '.'], np.array([0, 1]))]

    added_list=[]
    for data_tuple in data_list:
        glob_list=[]
        #print ('the length of dataset is',len((data_tuple))
        for word in data_tuple[0]:
            #print(word)
            try:
                cand_word_list=word2vec_models.most_similar(word)
                #print(cand_word_list)
            except: continue
            #print(cand_word_list)
            #print (cand_word_list[0])

            tempo_list=[]

            for i in range(knn):
                #近傍の単語５つをリストで出力する。
                tempo_list.append(cand_word_list[i][0])
            glob_list.append(tempo_list)

        #print(len(glob_list))

        for s in range(knn):
            #print (len(glob_list))
            # print (s)
            ttmp_list=[]
            for list_ in glob_list:
                ttmp_list.append(list_[s])
            added_list.append((ttmp_list,data_tuple[1]))

    return (added_list)
