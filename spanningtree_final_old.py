# X =([[1, 8],
#     [ 0, 9],
#     [ 1, 9],
#     [0, 0],
#     [1, 0]])

# X =([[1, 8],
#     [ 1, 9],
#     [ 1, 19],
#     [1, 100],
#     [1, 6]])

# print(X)

data='/home/kuru/Desktop/adiusb/veri-split/veriN_fol/image_train/'
#data_src_out='/home/kuru/Desktop/adiusb/veri-split/veriN_fol/image_train_spanning/'


#data='/home/kuru/Desktop/adiusb/veri-split/train/'

data_src_out='/home/kuru/Desktop/adiusb/veri-split/veriNoise1_train_spanning_folder/'
data_src_out2='/home/kuru/Desktop/adiusb/veri-split/veriNoise1_train_spanning_subfolder/'
data_src_out3='/home/kuru/Desktop/adiusb/veri-split/veriNoise1_train_spanning_all'

# data=data='/home/kuru/Desktop/market_noise/image_train/'
# data_src_out='/home/kuru/Desktop/adiusb/veri-split/market_image_train_noise_spanning/'
import os
import pickle
import numpy as np
import cv2
pkl = {}
k=[]

path='/home/kuru/Desktop/gmscreate/gmsNoise776/'
#path = '/home/kuru/Desktop/veri-gms-master/gms/'
#path='/home/kuru/Desktop/market_noise/gms/'
entries = os.listdir(path)
for name in entries:
    f = open((path+name), 'rb')
    ccc=(path+name)
    #print(ccc)
    if name=='featureMatrix.pkl':
        s = name[0:13]
        #print(s)
        
    else:
        #s = name[0:3]
        s = name.split('.')[0]

        #print(s)
        k.append(s)
       
    #print(s)
    #with open (ccc,"rb") as ff:
    #    pkl[s] = pickle.load(ff)
    #print(len((pkl)))
    pkl[s] = pickle.load(f)
    f.close
n=0
countt=0
for dire in os.listdir(data):
    try:
        #print(countt)
        countt=countt+1


        #print(dire)

        k[n]=dire

        #print(k[n])
        #print(len(pkl[k[n]]))
        aa=np.array((pkl[k[n]]))
        maxa=np.max(aa)
        #print(np.max(aa))
        #print(k)
        #print(pkl[k[n]])
        #print(list(pkl)[0:5])

        X=pkl[k[n]]
        for i in range(0,len(X)):
            for j in range(0,len(X)):
                if i==j:
                    #print(i,j)
                    X[i,j]=maxa

        #print(X)
        #print(X[0])
        cut=1.4
        from mst_clustering import MSTClustering
        model = MSTClustering(cutoff_scale=maxa*cut, approximate=False)
        labels = model.fit_predict(X)
        #print(labels)

        # model2 = MSTClustering(cutoff_scale=maxa*0.9, approximate=False)
        # labels2 = model2.fit_predict(X)
        # print(labels2)


        data_src= data+k[n]
        #print(data_src)
        c=0
        for pic in os.listdir(data_src):

            #print(pic)
            img = cv2.imread(os.path.join(data_src, pic))
            #print(labels[c])
            my_folder=data_src_out + '/'+str(k[n])
            if not os.path.exists(my_folder):
                os.makedirs(my_folder)
            my_folder=data_src_out2 + '/'+str(k[n])+'/'+ str(labels[c])
            if not os.path.exists(my_folder):
                os.makedirs(my_folder)
            my_folder=data_src_out3 
            if not os.path.exists(my_folder):
                os.makedirs(my_folder)
            ce=str(list(labels).count(labels[c])).zfill(3)
            #print("ds",ce)
            #print(labels,labels[c],str(list(labels).count(labels[c])))
            pic= str(pic)[0:9]+'_'+str(labels[c]).zfill(3)+'_'+ce +'_'+str(pic)[10:]
            #print(pic)
            #pathout=data_src_out + '/'+str(k[n])+'/'+str(cut)+'/'+ pic
            pathout=data_src_out +str(k[n])+'/'+ pic
            pathout2=data_src_out2 +str(k[n])+'/'+ str(labels[c])+'/'+ pic
            pathout3=data_src_out3+'/'+ pic

            #print(pathout)
            cv2.imwrite(pathout, img) 
            cv2.imwrite(pathout2, img) 
            cv2.imwrite(pathout3, img) 

            #break
            c=c+1
        #break   
    except (ValueError,IndexError):
        print("Index Error")