import urllib
import json
import os
import zipfile


PATH_DATA = 'D:\quant\python\data'
PATH_ZIP = 'D:\quant\python\zip'
PATH_NNDATA = 'D:\\quant\\python\\nndata\\'

DATA_ORIGINAL = []
DATA_FINAL = []
DATA_FINAL_SHAPE = 48  #数据矩阵的形状
DEPTH = 19  #市场深度的长度

PERCENT = 0.005 #百分之多少算暴涨暴跌
RATIO = 0.5  #占未来的概率
LABEL_LENGT = 1000  #标签数据的长度

'''
将store 目录下的压缩的数据文件移入zip目录下，并创建nndata目录。
zip 目录下的文件是从okex抓取的价格以及市场深度的原始数据，用于加工成可卷积的数据 48*48 的二维矩阵 包括原料数据和标签数据
最后存入nndata   目录下
'''



#解压文件
def unzip(zip_path,data_path):
    list = os.listdir(zip_path) 
    for i in range(0,len(list)):
        path = os.path.join(zip_path,list[i])
        if os.path.isfile(path):
            z = zipfile.ZipFile(path, 'r')
            z.extractall(path=r''+data_path)
            z.close()

#过滤异常数据
def filter(str):
    str = str.replace('\n[',',')
    return json.loads(str)


def write(path,data):
    f = open(path, 'w')
    f.write(str(data))
    f.close()


def getData(data_path):
    list = os.listdir(data_path) 
    result = []
    for i in range(0,len(list)):
        path = os.path.join(data_path,list[i])
        if os.path.isfile(path): 
            f = open(path,'r')
            result = result + filter(f.read())
            f.close()
    return result


#获得ma数据    
def ma(list,size,index,key):
    start = 0
    num = index  + 1
    total = 0
    if size<index :
        start = index-size
        num = size
        
    arr =  list[start:index]
    for i in range(0,len(arr)):
        total = total + arr[i][key]
    return round(total/num,2)

#最终的数据装载
def final(data,height):
    result = []
    for i in range(height-1,len(data)):
        if 'bids' in data[i].keys():
            it = data[i]
            pix = []
            for a in range(0,DEPTH):
                bids = it['bids'][a]
                pix.append(round(bids[0]*bids[1],2))
            for b in range(0,DEPTH):
                asks = it['asks'][b]
                pix.append(round(asks[0]*asks[1],2))
            pix.append(it['last'])
            pix.append(it['best_bid'])
            pix.append(it['best_ask'])
            pix.append(it['high_24h'])
            pix.append(it['low_24h'])
            pix.append(it['volume_24h'])
            pix.append(ma(data, 7,i,'last'))
            pix.append(ma(data,10,i,'last'))
            pix.append(ma(data,20,i,'last'))
            pix.append(ma(data,50,i,'last'))
            result.append(pix)
        else:
            print('数据异常')
            continue

    return result
    
def nnData(height,data,ratio,length):
    label = []
    list = []
    for i in range(height-1,len(data)):
        if i+length<len(data): 
            item = []
            up = 0
            down = 0
            lab = [1,0,0,0]  # 1的位置 -> 0 代表什么都不是 1暴涨暴跌  2暴涨 3暴跌
            for j in range(0,height):
                item = item + data[i+j-length+1]
            for j in range(0,length):
            
                if data[i+j][40]/data[i][40]>=1.01:
                    up +=1
                elif data[i+j][40]/data[i][40]<=0.99:
                    down +=1
                if j==length-1:
                    if up/length>=ratio:
                        lab = [0,0,1,0]
                    elif down/length>=ratio:
                        lab = [0,0,0,1]
                    elif up/length>=ratio and down/length>=ratio:
                        lab = [0,1,0,0]
                
                    
            list.append(item)
            label.append(lab)
    return {"list":list,"label":label}

    
def main():
        
    unzip(PATH_ZIP,PATH_DATA)
    DATA_ORIGINAL = getData(PATH_DATA)
    #print(len(DATA_ORIGINAL))
    d = final(DATA_ORIGINAL,DATA_FINAL_SHAPE)

'''
回测代码,忽略
    up = 0
    down = 0
    price = 0
    target = 'down'
    my = 1 
    price = DATA_ORIGINAL[0]['last']
    for j in range(len(DATA_ORIGINAL)):
        depth = DATA_ORIGINAL[j]
        bids = depth['bids'] 
        asks = depth['asks']
        bidsArea = 0
        asksArea = 0
        bidsNum = 0
        asksNum = 0
        
        for i in range(0,len(bids)):
            bidsArea = bidsArea + bids[i][0]*bids[i][1]
            bidsNum = bidsNum + bids[i][1]
        for i in range(0,len(asks)):
            asksArea = asksArea + asks[i][0]*asks[i][1]
            asksNum = asksNum + asks[i][1]
        if(round(my,2)<0.001):
            return print(j)
        if bidsArea/asksArea>1.5:
            
            if target=='up':
                my = ((bids[2][0] -price)/price*10+1)*my*0.997
                print('up:' +str(my) +'   '+str(bidsArea/asksArea))
                target = 'down'
                price = depth['last']
                down+=1
        elif bidsArea/asksArea<1.35:
            
            if target=='down':
                my = ((price - asks[2][0])/price*10+1)*my*0.997
                print('down:' +str(my) +'   '+str(bidsArea/asksArea))
                
                target = 'up'
                price = depth['last']
                up+=1
            
        
    print('up:'+str(up)+' down:'+str(down)+'  my:'+str(round(my,2)))
'''
    
    #print('结束!!!!!!!!!!!!正在生成神经网络的数据')

    
    train = nnData(DATA_FINAL_SHAPE,d,RATIO,LABEL_LENGT)
    write(PATH_NNDATA+'list.data',train['list'])
    write(PATH_NNDATA+'label.data',train['label'])

    print('神经网络的数据成功，共生成list:'+str(len(train['list'])) + ' lab:'+str(len(train['label'])))
    #return train
    
main()















