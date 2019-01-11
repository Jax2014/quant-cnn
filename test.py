import os
import json

def read(name):
    f = open('./nndata/'+name+'.data','r')
    result = json.loads(f.read())
    f.close()
    return result
    
    
    
list = read('list')
label = read('label')
print(str(len(list))+'     '+str(len(label)))