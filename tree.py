import numpy as np
import math


def entropy(probs):
    ### take a list of probabilities and calculate their entropy values
    if probs[0] == 0 or probs[1] == 0:
        return 0
    else:
        list_entropy = -(probs[0])*math.log(probs[0], 2)-(probs[1])*math.log(probs[1], 2)
        return list_entropy
    

def entropy_of_list(a_list):
    ### take a list of items with discrete values (e.g., class A , class B ) and return the list of entropy values for those items
    a_list = np.where(a_list == a_list.iloc[0], -1, 1)
    counts = [np.count_nonzero(a_list == -1),np.count_nonzero(a_list == 1)]
    probs = [counts[0]/sum(counts),counts[1]/sum(counts)]
    
    return entropy(probs)


def information_gain(df, split_feature_name, target_feature_name):
    
    ### take a dataFrame of features, and quantify the entropy of a target feature
    ### after performing a split along the values of another feature and calculate the corresponding information gain

    ori_entropy =entropy_of_list(df[target_feature_name])
    sum_entropy = 0
    fea_value = np.unique(df[split_feature_name])
    
    for i in fea_value:
        fea_val  = df[df[split_feature_name] == i]
        if fea_val.empty:
            continue
        prob_of_fea_val = len(fea_val)/len(df)
        
        fea_entropy =entropy_of_list(fea_val[target_feature_name])
        sum_entropy += fea_entropy*prob_of_fea_val
        
    info_gain = ori_entropy - sum_entropy
    return info_gain


def id3(df, target_feature_name, feature_names, default_class = None):
    
    ## counting for the target feature
    from collections import Counter
    cnt = Counter(x for x in df[feature_names])
    
    # First check : to check if there is only one class left
    check_split = np.unique(df[target_feature_name])
    
    # Second check: to check if there is only one feature left
    if len(cnt) == 1:
        return np.argmax(df[target_feature_name].value_counts())
    
    ## Third check: is this split of the dataset empty?
    # if yes, return a default value
    elif df.empty : #or (not feature_names)
        return default_class 
    
    # if yes, return the only one class
    elif len(check_split) == 1:
        return  check_split[0]
    
    ## Otherwise: this dataset is ready to be split!
    else:
        ### step 1: get the default value for next recursive call of this function
        index_of_max = list(cnt.values()).index(max(cnt.values())) 
        default_class = list(cnt.keys())[index_of_max] # most common value of target feature in dataset
        
        ### step 2: choose the best feature to split on
        best_gain,best_split = 0 , 0 
        for i in cnt.keys():
            gain = information_gain(df, i,  target_feature_name)
            if gain > best_gain:
                best_gain = gain
                best_split = i
        
        # if there is no any other better split,then return the class with the largest number of
        if best_split == 0:
            return np.argmax(df[target_feature_name].value_counts())
        
        ### step 3: create an empty tree, to be populated in a moment
        
        # your code here
        tree = { best_split :{}}
        
        ### Step 4: split dataset
        # on each split, recursively call this "id3" function
        # populate the empty tree with subtrees, which are the result of the recursive call
        
        feature_names.remove(best_split)
        for i in np.unique(df[best_split]):
            fea_val  = df[df[best_split] == i]
            tree[best_split][i] = id3(fea_val, target_feature_name, feature_names, default_class = None)
            
        return tree

def classify(instance, tree, default = None):
    feature = list(tree.keys())[0]
    
    if instance[feature] in list(tree[feature].keys()):
        result = tree[feature][instance[feature]]
        if isinstance(result, dict): # this is a tree, delve deeper
            return classify(instance, result)
        else:
            return result # this is a label
    else:
        return default
        