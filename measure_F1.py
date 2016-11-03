import numpy as np

#y_true = np.array([[0, 1, 0, 0, 1, 1],[0, 1, 0, 0, 1, 1],[0, 1, 0, 0, 1, 1]])
#y_pred = np.array([[0, 0, 1, 0, 0, 1],[0, 0, 1, 0, 0, 1],[0, 0, 1, 0, 0, 1]])

def perf_measure(y_true_matrix, y_pred_matrix):
    F1_list=[]
    for y_actual, y_hat in zip(y_true_matrix, y_pred_matrix):

        tp = 0
        fp = 0
        tn = 0
        fn = 0

        for i in range(len(y_hat)):
            if y_actual[i]==y_hat[i]==1:
               tp += 1
        for i in range(len(y_hat)):
            if y_actual[i]==0 and y_hat[i]==1:
               fp += 1
        for i in range(len(y_hat)):
            if y_actual[i]==y_hat[i]==0:
               tn += 1
        for i in range(len(y_hat)):
            if y_actual[i]==1 and y_hat[i]==0:
               fn += 1

        #print(tp, fp, tn, fn)

        recall=(tp / float(tp + fn))
        precision=(tp / float(tp + fp))
        try:
            F1 =2 * (precision * recall) / float(precision + recall)
        except:
            continue

        acc=((tp+tn)/float(tp+fp+tn+fn))

        F1_list.append(F1)

    return(np.array(F1_list).mean())

#print(perf_measure(y_true, y_pred))
#print (y_true)
