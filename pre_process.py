import xml.etree.ElementTree as ET
from nltk.tokenize import word_tokenize
from collections import Counter
from collections import defaultdict
import numpy as np
from numpy.random import *

def xml_parse(file_name):
    train_data=[]
    sentences=[]

    reviews = ET.parse(file_name).getroot().findall('Review')
    for r in reviews:
        sentences += r.find('sentences').getchildren()

    for e in sentences:
        categories=[]
        opinions=e.find('Opinions')
        if opinions==None:
            categories.append("NONE")
        else:
            for opinion in opinions:
                categories.append(opinion.attrib['category'])

        text=e.find('text').text
        temp_tuple=(word_tokenize(text),categories)
        train_data.append(temp_tuple)

    return train_data

def common_labels_counter(data_list):
    common_labels=[]
    whole_labels=[]

    for data_tuple in data_list:
        whole_labels+=data_tuple[1]

    counterA=Counter(whole_labels)
    tempo_list=counterA.most_common(18)
    for c_tap in tempo_list:
        common_labels.append(c_tap[0])

    return common_labels

def label_proc(data_list,cm_labels,normalized):
    tempo_labels=cm_labels
    for data_tuple in data_list:
        category_list=data_tuple[1]
        for cat in category_list:
            if cat not in tempo_labels:
                data_tuple[1][category_list.index(cat)]='OTHER'

    tempo_labels.append('OTHER')

    common_dict=defaultdict(int)
    for feature in tempo_labels:
        if not feature in common_dict:
            common_dict[feature] = len(common_dict)

    output_data_list=[]
    for data_tuple in data_list:
        label_vect=[0]*len(common_dict)
        category_list=data_tuple[1]

        for cat in category_list:
            if normalized==True:
                label_vect[common_dict[cat]]=1/float(len(set(category_list)))
            else:
                label_vect[common_dict[cat]]=1

        output_data_list.append((data_tuple[0],np.array(label_vect)))

    return output_data_list

def load_word2vec(word2vec, words):
    words    = list(words)
    screened_words = list()
    for word in words:
        if word in word2vec:
            screened_words.append(word)
    weights  = np.zeros([len(screened_words), 300])
    screened_words = sorted(screened_words)
    w2idx    = {w:i for i, w in enumerate(screened_words)}
    for word,index in sorted(w2idx.items()):
        weights[index, :] = word2vec[word]
    return weights, screened_words, w2idx

def make_Dataset(train_data_list,word2vec_models):
    train_list=[]
    train_labels=[]
    for data_tuple in train_data_list:
        train_labels.append(data_tuple[1])

        sent_list=[]
        embedding, vocabs, word2index = load_word2vec(word2vec_models, data_tuple[0])
        for word in vocabs:
            w_id = word2index[word]
            sent_list.append(embedding[w_id])
        sent_matrix=np.array(sent_list)

        if sent_matrix.shape[0]==0:
            train_list.append(np.mean(randint(0,10,(5,300)),axis=0))

        else:
            train_list.append(np.mean(sent_matrix,axis=0))

    train_matrix=np.array(train_list)
    label_matrix=np.array(train_labels)

    #print (label_matrix.shape)
    return (train_matrix,label_matrix)


'''
file_name='/home/cl/shinnosuke-n/Wang_category/data/ABSA15_Laptops_Test.xml'
train_file='/home/cl/shinnosuke-n/Wang_category/data/ABSA-15_Laptops_Train_Data.xml'
a_data=pp.xml_parse(file_name)
b_labels=pp.common_labels_counter(a_data)
test_data=pp.label_proc(a_data,b_labels,normalized=False)

raw_train_data=pp.xml_parse(train_file)
train_data=pp.label_proc(raw_train_data,b_labels,normalized=True)

train_sent,train_labels=pp.make_Dataset(train_data,word2vec_models)
test_sent,test_labels=pp.make_Dataset(test_data,word2vec_models)


#print c_data


#def make_Dataset()
'''
