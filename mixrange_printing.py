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

data_src_out='/home/kuru/Desktop/adiusb/veri-split/range_veriNoise1_train_spanning_folder/'
data_src_out2='/home/kuru/Desktop/adiusb/veri-split/range_veriNoise1_train_spanning_subfolder/'
data_src_out3='/home/kuru/Desktop/adiusb/veri-split/range_veriNoise1_train_spanning_all'

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


X=pkl['001']
print(pkl['001'])
aa=np.array(pkl['001'])
maxa=np.max(aa)
for i in range(0,len(X)):
    for j in range(0,len(X)):
        if i==j:
            #print(i,j)
            X[i,j]=maxa
print(X)

xx= maxa - X
print(xx)
adjmtrx=xx

import sys  # Library for INT_MAX

class Graph():

    def __init__(self, vertices):
        self.V = vertices
        self.graph = [[0 for column in range(vertices)]
                      for row in range(vertices)]
        #print(self.V,self.graph)

        # A utility function to print the constructed MST stored in parent[]

    def printMST(self, parent):
        #print(parent)
        matt=[]
        print("Edge \tWeight")
        #print(self.graph[1])
        for i in range(1, self.V):
            print(parent[i], "-", i, "\t", self.graph[i][parent[i]])
            matt.append((parent[i],i,self.graph[i][parent[i]]))
            #matt.append((i,parent[i],self.graph[i][parent[i]]))
        return matt

    # A utility function to find the vertex with
    # minimum distance value, from the set of vertices
    # not yet included in shortest path tree 
    def minKey(self, key, mstSet):

        min = sys.maxsize

        for v in range(self.V):
            if key[v] < min and mstSet[v] == False:
                min = key[v]
                min_index = v

        return min_index

    # Function to construct and print MST for a graph
    # represented using adjacency matrix representation
    def primMST(self):

        outMST = Graph(self.V)
        # Key values used to pick minimum weight edge in cut 
        key = [sys.maxsize] * self.V
        parent = [None] * self.V  # Array to store constructed MST
        # Make key 0 so that this vertex is picked as first vertex 
        key[0] = 0
        mstSet = [False] * self.V

        parent[0] = -1  # First node is always the root of

        for cout in range(self.V):

            # Pick the minimum distance vertex from  
            # the set of vertices not yet processed.  
            # u is always equal to src in first iteration 
            u = self.minKey(key, mstSet)

            # Put the minimum distance vertex in  
            # the shortest path tree 
            mstSet[u] = True

            # Update dist value of the adjacent vertices  
            # of the picked vertex only if the current  
            # distance is greater than new distance and 
            # the vertex in not in the shotest path tree 
            for v in range(self.V):
                # graph[u][v] is non zero only for adjacent vertices of m 
                # mstSet[v] is false for vertices not yet included in MST 
                # Update the key only if graph[u][v] is smaller than key[v] 
                if self.graph[u][v] > 0 and mstSet[v] == False and key[v] > self.graph[u][v]:
                    key[v] = self.graph[u][v]
                    parent[v] = u

        for v in range(self.V):
            outMST.graph[v][parent[v]] = self.graph[v][parent[v]]
            outMST.graph[parent[v]][v] = self.graph[parent[v]][v]

        outmat= self.printMST(parent)
        return outMST,outmat

vert=len(adjmtrx)
print(vert)
g = Graph(vert)
#print(g)

g.graph = adjmtrx
newg = Graph(g.V)
#print(newg)
newg,outmat = g.primMST()

print(outmat)
print(len(outmat))







def clustering(outmat,cut1):
    cluss1=[]
    remain=[]
    start=0


    for i in range(0,len(outmat)):
        #print(outmat[i][2])

        if outmat[i][2]<cut1:
            if start==0:
                #print(outmat[i]) 
                cluss1.append(outmat[i][0]) 
                cluss1.append(outmat[i][1]) 

                start=1

            elif start==1:
                #print((outmat[i][1] in cluss1),(outmat[i][0] in cluss1))
                if (outmat[i][1] in cluss1) or (outmat[i][0] in cluss1):
                    cluss1.append(outmat[i][0]) 
                    cluss1.append(outmat[i][1]) 
                    # if outmat[i] in outmatcopy:
                    #     outmatcopy.remove(outmat[i])

                    #print("cluster in the loop",cluss1)

                else:   
                    remain.append(outmat[i])
                    #print("remain in the loop",remain)

        cluss1=list(set(cluss1))

    return cluss1, remain

cut2=maxa* 0.9 
cut1=maxa * 0.8     
print(cut1,cut2)       

# cluss1, remain =clustering(outmat,cut1)
# cluss2, remain2 =clustering(outmat,cut2)
# remain= list(set(remain+remain2)) 
# cluss1=list(set(cluss1+cluss2))

# print("mixedremain",remain)
# print(clustering(remain,3))


# print(clustering(outmat,3))


# mergedclus=list(set(clustering(outmat,3)+clustering(outmat,2)))
# print(mergedclus)



clus=[]
while(len(outmat)>0):
    #print(clustering(outmat,cut1))
    #print(clustering(outmat,cut2))
    # cluss1, remain =clustering(outmat,cut1)



    cluss1, remain =clustering(outmat,cut1)
    cluss2, remain2 =clustering(outmat,cut2)
    remain= list(set(remain+remain2)) 
    cluss1=list(set(cluss1+cluss2))

    clus.append(cluss1)
    outmat=remain
    print("remaing",outmat)
    print("totalclus",clus)

print(sum(len(x) for x in clus))

def labelling(clus):
    labels={}
    for i in range(0,sum(len(x) for x in clus)):
        labels[i]=[]
        for l,e in enumerate(clus):
            if i in e:
                print(i,l,e)

                labels[i].append(l)
        if labels[i]==[]:
            labels[i]=[sum(len(x) for x in clus)]
    print(labels)
    print(set(labels))

    return labels
labelling(clus)


n=0
countt=0
for dire in os.listdir(data):
    try:
        #print(countt)
        countt=countt+1



        print(dire)

        k[n]=dire

        X=pkl[k[n]]
        #X=pkl['001']
        print(X)
        aa=np.array(X)
        maxa=np.max(aa)
        for i in range(0,len(X)):
            for j in range(0,len(X)):
                if i==j:
                    #print(i,j)
                    X[i,j]=maxa
        print(X)

        xx= maxa - X
        print(xx)
        adjmtrx=xx

        #print(k[n])
        #print(len(pkl[k[n]]))
        #aa=np.array((pkl[k[n]]))
        #maxa=np.max(aa)
        #print(np.max(aa))
        #print(k)
        #print(pkl[k[n]])
        #print(list(pkl)[0:5])

        vert=len(adjmtrx)
        print(vert)
        g = Graph(vert)
        #print(g)

        g.graph = adjmtrx
        newg = Graph(g.V)
        #print(newg)
        newg,outmat = g.primMST()

        print(outmat)
        print(len(outmat))


        #print(X)
        #print(X[0])
        cut2=maxa* 0.9 
        cut1=maxa * 0.8     
        print(cut1,cut2)  
        clus=[]
        while(len(outmat)>0):
            print(clustering(outmat,cut1))
            print(clustering(outmat,cut2))
            # cluss1, remain =clustering(outmat,cut1)



            cluss1, remain =clustering(outmat,cut1)
            cluss2, remain2 =clustering(outmat,cut2)
            remain= list(set(remain+remain2)) 
            cluss1=list(set(cluss1+cluss2))

            clus.append(cluss1)
            outmat=remain
            print("remaing",outmat)
            print("totalclus",clus)
        total_length=sum(len(x) for x in clus)
        print(sum(len(x) for x in clus))
        labels = labelling(clus)
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
            print(labels[c])
            for i in labels[c]:

                ce=str(i).zfill(3)
                print("ds",ce)
                my_folder=data_src_out + '/'+str(k[n])
                if not os.path.exists(my_folder):
                    os.makedirs(my_folder)
                my_folder=data_src_out2 + '/'+str(k[n])+'/'+ ce
                if not os.path.exists(my_folder):
                    os.makedirs(my_folder)
                my_folder=data_src_out3 
                if not os.path.exists(my_folder):
                    os.makedirs(my_folder)

            #print(labels,labels[c],str(list(labels).count(labels[c])))
                pic= str(pic)[0:9]+'_'+ce +'_'+str(pic)[10:]
                print(pic)
                #pathout=data_src_out + '/'+str(k[n])+'/'+str(cut)+'/'+ pic
                pathout=data_src_out +str(k[n])+'/'+ pic
                pathout2=data_src_out2 +str(k[n])+'/'+ce+'/'+ pic
                pathout3=data_src_out3+'/'+ pic

                print(pathout2)
                cv2.imwrite(pathout, img) 
                cv2.imwrite(pathout2, img) 
                cv2.imwrite(pathout3, img) 

            #break
            c=c+1
        #break   
    except (ValueError,IndexError,KeyError):
        print("Index Error")