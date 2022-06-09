import os
import numpy as np
import pandas as pd
import collections
import pickle

def exchange_dict(a):
    return dict((v,k) for k,v in a.items())

def get_classname(x):
    x = np.argmax(x)

def sub_fuse(logits_list, idx_to_class):
    ratio_list = [0.2,0.2,0.2,0.2,0.2]
    logits_sum = np.zeros(100)
    count = 0
    #print(logits_list)
    for i in logits_list:
        np_logits = np.array(i)
        #print(np_logits)
        logits_sum += np_logits*ratio_list[count]
        count += 1
        
    
    return idx_to_class[np.argmax(logits_sum)]
        

def fuse_logits(pk_list, idx_to_class,outname):
    #make dict
    whole_data = collections.OrderedDict()
    for pk in pk_list:
        data = None
        with open(pk, 'rb') as f:
            data = pickle.load(f)
        #print(data[list(data.keys())[0]])
        for i in sorted(list(data.keys())):
            if i in whole_data:
                whole_data[i].append(data[i])
            else:
                whole_data[i] = [data[i]]
        #print(whole_data[list(data.keys())[0]])
    #fuse
    for i in whole_data:
        whole_data[i] = sub_fuse(whole_data[i],idx_to_class)
    
    #make csv  
    imagename = []
    the_class = []
    for i in sorted(list(whole_data.keys())):
        imagename.append(i)
        the_class.append(whole_data[i])
        
    imagename = np.expand_dims(imagename, axis=1)
    the_class = np.expand_dims(the_class, axis=1)
    
    #new_data = np.concatenate((shop_id_list,fengfudu_list), axis=0)
    new_data = np.hstack((imagename,the_class))
    
    df = pd.DataFrame(new_data, columns = ['filename','cultivar'])
    df.to_csv(outname,index=None)


        

if __name__ == '__main__':

    
    logits_list = ['logits/model1.pk','logits/model2','logits/model3','logits/model4'] 
    
    outname = 'logits/output/fuse_result.csv' 

    directory = '/path/save/'
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    idx_to_class = exchange_dict(class_to_idx)
                        
                        
    fuse_logits(logits_list, idx_to_class, outname)