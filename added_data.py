

# -*- coding: utf-8 -*-
import numpy as np

def main(word2vec_models,data_list,knn):
    added_list=[]
   
      for data_tuple in data_list:
        glob_list=[]

        for word in data_tuple[0]:
            try:
                cand_word_list=word2vec_models.most_similar(word)
            except: continue

            tempo_list=[]

            for i in range(knn):
                tempo_list.append(cand_word_list[i][0])
            glob_list.append(tempo_list)

        for s in range(knn):
            ttmp_list=[]

            for list_ in glob_list:
                ttmp_list.append(list_[s])
            added_list.append((ttmp_list,data_tuple[1]))

    return (added_list)
