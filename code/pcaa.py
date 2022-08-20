import numpy as np

def pcaa(dataMat, percent):
    newData, meanVal = zeroMean(dataMat)
    print("newData: ",newData)
    print("meanVal: ",meanVal)
    print("Percentage: ", percent)

    # 求協方差矩陣
    covMat = np.cov(newData, rowvar=0) #cov為計算協方差，
    print("協方差矩陣的shape: ", covMat.shape)
    print("以下為協方差矩陣:")
    print(covMat)

    # 求特征值和特征向量
    eigVals, eigVects = np.linalg.eig(np.mat(covMat))
    print("eigVals: ", eigVals.shape)
    print(eigVals)
    print("\n")
    print("eigVects: ", eigVects.shape)
    print(eigVects)
    print("\n")
    

    # 抽取前n個特征向量
    n = percentage2n(eigVals, percent) #透過方差百分比決定 n 值
    print("數據降低到：" + str(n) + '維')

    # 將特征值按從小到大排序
    eigValIndice = np.argsort(eigVals)
    print("特徵值由小至大排序: ", eigValIndice)
    # 取最大的n個特征值的下標
    n_eigValIndice = eigValIndice[-1:-(n + 1):-1]
    # 取最大的n個特征值的特征向量
    print("決定選取的特徵值: ", n_eigValIndice)
    n_eigVect = eigVects[:, n_eigValIndice]
    # 取得降低到n維的數據
    lowDataMat = newData * n_eigVect

    ##---------------------------------###
    # 抽取前n+1個特征向量
    n1 = percentage2n(eigVals, percent) + 1 #透過方差百分比決定 n+1 值
    print("選擇n+1，數據降低到：" + str(n1) + '維')

    # 將特征值按從小到大排序
    eigValIndice_1 = np.argsort(eigVals)
    print("特徵值由小至大排序: ", eigValIndice_1)
    # 取最大的n+1個特征值的下標
    n_eigValIndice_1 = eigValIndice_1[-1:-(n + 2):-1]
    # 取最大的n+1個特征值的特征向量
    print("決定選取的特徵值: ", n_eigValIndice_1)
    n_eigVect_1 = eigVects[:, n_eigValIndice_1]

    # 取得降低到n+1維的數據
    lowDataMat_1 = newData * n_eigVect_1

    reconMat_1 = (lowDataMat_1 * n_eigVect_1.T) + meanVal

    return lowDataMat, lowDataMat_1, eigVals

def zeroMean(dataMat):
    # 求列均值
    meanVal = np.mean(dataMat, axis=0)
    # 求列差值
    newData = dataMat - meanVal
    return newData, meanVal

# 通過方差百分比確定抽取的特征向量的個數
def percentage2n(eigVals, percentage):
    # 按降序排序
    sortArray = np.sort(eigVals)[-1::-1]
    # 求和
    arraySum = sum(sortArray)

    tempSum = 0
    num = 0
    for i in sortArray:
        tempSum += i
        num += 1
        if tempSum >= arraySum * percentage:
            return num

def covariance_bar(eigVals):
    Index = np.argsort(eigVals)[-1::-1]
    Value = np.sort(eigVals)[-1::-1]
    arraySum = sum(Value)
    aaa = np.zeros(shape=len(eigVals))
    j = 0
    for i in Value:
        temp = 0
        temp = round((i/arraySum), 4)
        #aaa = np.append(aaa, temp)
        aaa[j] = temp
        j = j + 1
    return Index, aaa


def label_coding(label):
    dataset[label]= label_encoder.fit_transform(dataset[label]) 
    dataset[label].unique()
    
def sortingFunction(data):
    return data.shape[0]

def crop_dataset(len_dataset):
    for label in range(len_dataset):
        temp_dataframe=dataset[dataset['Label']==label]
        try:
            #if temp_dataframe.shape[0]>8:
                #_ ,temp_dataframe = train_test_split(temp_dataframe,test_size =.25)
            temp_train ,temp_test = train_test_split(temp_dataframe,test_size=0.25)
            list_train.append(temp_train)
            list_test.append(temp_test)
        except:
            print("Error for "+str(label))
            
def new_sample_generation(x,y,z):
    need=500-x if x<=500 else 0
    #print("n_sample: "+str(x)+" max_sample: "+str(y)+" need_to_create: "+str(need))
 