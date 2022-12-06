import tree
import numpy as np
import pandas as pd

def bagging(x):
    return np.argmax(x.value_counts())

def random_forest_train(train,target_feature_name, n_estimators = 10,):
        forest = []
        for i in range(n_estimators):
            
            ### step1 sampling feature
            feature_names = list(train.columns)
            feature_names.remove(target_feature_name)
            sampling_features = np.random.choice(feature_names,replace = False, size = len(train.columns)//2)
            sampling_features = list(sampling_features)

            ### step2 sampling sample - bootstrap
            sampling_data = train.sample(n=len(train)//10,replace=True, random_state=1)

            ### step3 built tree 
            built_tree = tree.id3(sampling_data, target_feature_name, sampling_features)
            forest.append(built_tree)
        return forest
    
def random_forest_test(test,forest,n_estimators =10):
    df_result = pd.DataFrame(index=test.index)
    for i in range(n_estimators):
        
        # step4 use tree to pred testing data
        pred = test.apply(tree.classify, axis=1, args=(forest[i],0))
        df_result[i] = pred
    
    # step5 bagging for pred    
    result= df_result.apply(bagging, axis=1)
    
    return result   