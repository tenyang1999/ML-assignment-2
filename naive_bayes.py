import numpy as np
import pandas as pd

def conditional_probability(train, target_feature, feature_names):

    all_len = len(train)
    all_prob = {}
    for i in train[target_feature].unique():
        sub_train = train[train[target_feature] == i]
        
        target_prob = {}
        for j in feature_names:
            feature = j
            
            feature_prob = {}
            for k in train[feature].unique():

                k_len = len(sub_train[sub_train[feature] == k]) 
                prob = k_len/all_len
                feature_prob[k] = prob  

            target_prob[j] = feature_prob

        all_prob[i] = target_prob
    return all_prob

def naive_bayes_train(train, target_feature, feature_names):
    
    all_len = len(train)
    
    # step1 :compute the probability of y
    y_prob = {}
    for i in train[target_feature].unique():
        y_len = len(train[train[target_feature] == i])
        prob = y_len/all_len
        y_prob[i] = prob  
    
    # step2 :compute the probability of x_1 to x_n
    x_prob = {}
    for j in feature_names:
        feature = j
            
        feature_prob = {}
        for k in train[feature].unique():

            k_len = len(train[train[feature] == k]) 
            prob = k_len/all_len
            feature_prob[k] = prob  

        x_prob[j] = feature_prob
    
    # step3 :compute the conditional probability of x_1 to x_n under y
    x_y_prob = conditional_probability(train,target_feature,feature_names)
    
    all_prob = {'y_prob':y_prob,"x_prob":x_prob,"x_y_prob":x_y_prob}
    
    return all_prob

def naive_bayes_test(x,feature_names,prob):
    total_prob = []
    for i in ([0,1]):
        y_prob = prob['y_prob'][i]

        x_y_prob = 1
        tar = prob["x_y_prob"][i]
        for j in feature_names:
            try :
                x_y_prob *= tar[j][x[j]]
            except:
                continue
        # 因每個類別都要除以相同的x_prob，故省略計算，即可直接比較
        total = y_prob*x_y_prob
        total_prob.append(total)

    pred = np.argmax(total_prob)
    return pred