import os
def opti():
     from copy import deepcopy
     from sklearn.metrics import silhouette_score
     import csv
     from sklearn.datasets import load_iris
     import numpy as np
     import matplotlib.ticker as ticker
     import pandas as pd
     import pickle
     from matplotlib import pyplot as plt
     from mpl_toolkits.mplot3d import Axes3D
     from sklearn.cluster import KMeans
     plt.rcParams['figure.figsize'] = (9, 5)
     data = pd.read_csv('input.csv')
     print("Input Data and Shape")
     print(data.shape)
     data.head()
     f1 = data['Latitude'].values
     f2 = data['Longitude'].values
     f3 = data['Date'].values
     f4 = data['Time'].values
     f5 = data['Message'].values
     mm=[]
     sil=[]
     cnt=0
     X = np.array(list(zip(f1, f2)))
     def dist(a, b, ax=1):
         return np.linalg.norm(a - b, axis=ax)
     for n_cluster in range(2, 5):
         kmeans = KMeans(n_clusters=n_cluster).fit(X)
         label = kmeans.labels_
         sil_coeff = silhouette_score(X, label, metric='manhattan')
         mm.append(sil_coeff)
     k=mm.index(max(mm))+2
     print ('optimal number of clusters :',k)
     C_x = np.random.randint(0, np.max(X)-20, size=k)
     C_y = np.random.randint(0, np.max(X)-20, size=k)
     C = np.array(list(zip(C_x, C_y)), dtype=np.float32)
     C_old = np.zeros(C.shape)
     clusters = np.zeros(len(X))
     error = dist(C, C_old, None)
     kmeans = KMeans(n_clusters=k)
     kmeans = kmeans.fit(X)
     labels = kmeans.predict(X)
     labels2=kmeans.labels_
     centroids = kmeans.cluster_centers_
     jj=[]
     g=open("output_after_clustering.csv","w",newline="")
     writer=csv.writer(g)
     writer.writerow(['Date','Time','Latitude','Longitude','Message','new_lat','new_long','cluster_number'])
     for i in range(len(X)):
         writer.writerow([f3[i],f4[i],f1[i],f2[i],f5[i],centroids[labels2[i]][0],centroids[labels2[i]][1],(labels2[i]+1)])
     g.close()
     pl=['1','2','3','4','5','6','7','8','9','10']
     fig = plt.figure()
     ax = fig.add_subplot(111, projection='3d')
     ax.locator_params(nbins=1,axis='z')
     colors = ['r', 'g', 'b', 'y', 'c', 'm','Orange','ForestGreen','Brown','peachpuff','gold','rosybrown']
     ax.scatter(f1, f2, c='blue', s=9)
     sc=ax.scatter(X[:,0],X[:,1],1,c=labels2,s=100,depthshade='True',marker=".")
     ax.scatter(centroids[:,0],centroids[:,1],1,c='black',s=200,depthshade='True',marker="D")
     ax.set_zlim(1,1.01)
     plt.colorbar(sc)
     plt.show()
     print("Final centroids")
     print(centroids)
     #print("Check outputfinal.txt to get the output file containing clusters")
     with open("centroid.txt", "w") as output:
         output.write(str(centroids))

     #Srting the clustering on basis of cluster number and time

     import sys, csv ,operator,pandas as pd,numpy as np
     data=pd.read_csv('output_after_clustering.csv')
     da=data['Date'].values
     ti=data['Time'].values
     la=data['Latitude'].values
     Lo=data['Longitude'].values
     ms=data['Message'].values
     n_la=data['new_lat'].values
     n_lo=data['new_long'].values
     cl=data['cluster_number'].values
     X=np.array(list(zip(da,ti,n_la,n_lo,cl,ms)))
     sortedlist = sorted(X, key=operator.itemgetter(4,1),reverse=False)
     g1=open("sorted_clustered.csv",'w',newline="")
     writer= csv.writer(g1)
     #writer.writerow(['Date','Time','Latitude','Longitude','cluster num','Message'])
     for row in sortedlist:
                   writer.writerow(row)
     g1.close()


     import csv
     csv_file ="sorted_clustered.csv" # raw_input('Enter the name of your input file: ')
     txt_file ="c2.txt" #raw_input('Enter the name of your output file: ')
     with open(txt_file, "w") as my_output_file:
         with open(csv_file, "r") as my_input_file:
             [ my_output_file.write(" ".join(row)+'\n') for row in csv.reader(my_input_file)]
         my_output_file.close()

     import re
     with open('c2.txt') as f,open('content_input.txt','w') as f1:
         for line in f:
                  j=int((line.split()[1]).split(':')[0])
                  if j < 10 :
                           f1.write(line.replace("%s"%j , "0%s"%j , 1))
                  else:
                         f1.write(line)

def first():
     import nltk
     import sys
     import csv
     import re
     import string
     import os
     from nltk.tokenize import sent_tokenize, PunktSentenceTokenizer, word_tokenize, wordpunct_tokenize
     os.remove('c2.txt')
     sys.stdout=open("testabc.txt","w")
     with open('content_input.txt', 'r') as f:
             for rec in f.readlines():
                      print (rec[:16],rec[20:],end="")
     sys.stdout.close()
     #os.remove('c1.txt')
     with open('testabc.txt','r')as myFile:
         str1=myFile.read()
         punctuation = ['!', '"', '..', '  ', '#', '$', '%', '&', '(', ')', '*', ',', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~'] 
         for i in punctuation:
             str1 = str1.replace(i,"")
     sys.stdout=open("st2.txt","w")
     print(str1)
     sys.stdout.close()
     os.remove('testabc.txt')
     with open("st2.txt") as f:
         reader=csv.reader(f,delimiter=" ")
         d=list(reader)
     dn=[x for x in d if x]
     dlen=len(dn)
     f.close()
     for i in range(0,dlen-1):
         for k in range(i+1,dlen):
             if(dn[i][0]==dn[k][0] and dn[i][1]==dn[k][1] and dn[i][2]==dn[k][2] and dn[i][3]==dn[k][3] and dn[i][4]==dn[k][4]):
                 dn[k][0]=""
                 dn[k][1]=""
                 dn[k][2]=""
                 dn[k][3]=""
                 dn[k][4]=""
     file=open("testabc1.txt","w")
     for i in range(0,len(dn)):
         for j in d[i]:
             file.write(j)
            # print(j, end=" ")
             file.write(" ")
         if(i<len(dn)-1):
             if(d[i+1][0]!=""):
                 file.write("\n")
                 #print("\n")
     file.close()
     os.remove('st2.txt')
     with open('testabc1.txt','r') as inf, open("m.txt",'w') as m:
         for i in inf:
             i= i.replace("   "," ")
             i = i.replace("  "," ")
             m.write(i)
     inf.close()
     m.close()
     os.remove('testabc1.txt')
     sys.stdout=open("t.txt","w")
     with open('m.txt', 'r') as fileinput:
        for line in fileinput:
            line = line.rstrip().lower()
            print (line)
     sys.stdout.close()
     os.remove('m.txt')

def second():
     import nltk
     import sys
     from nltk.corpus import wordnet
     from nltk.tokenize import sent_tokenize, PunktSentenceTokenizer, word_tokenize
     lst = []
     result = []
     ndc = []
     mlist = []
     synonyms = []
     casu_syn = []
     p=1
     word_index=[]
     num=[]
     words=[]
     casu=[]
     casu_lst=['death', 'dead', 'injure', 'injured', 'casualty', 'victim', 'damaged', 'killed']
     for i in casu_lst: 
         for syn in wordnet.synsets(i):
             for l in syn.lemmas():
                 casu_syn.append(l.name())
     punctuation = ['!', '"', '..', '  ', '#', '$', '%', '&', '(', ')', '*', ',', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~', '.', ':']
     def mod1(a,b):
         if a>b:
             return a-b;
         else:
             return b-a;
             
     with open('dcw.txt') as q:
         for s in q:
             for e in nltk.word_tokenize(s):
                 lst.append(e)
                 for syn in wordnet.synsets(e):
                         for l in syn.lemmas():
                             synonyms.append(l.name())
     q.close()
     for line in set(synonyms):
         result.append(line)
     for line in set(casu_syn):
         casu.append(line)
     c = len(lst)
     sys.stdout=open("temp.txt","w")
     with open('t.txt', 'r') as inf, open ('Final_output.txt', 'w') as of:
         for i in inf:
             ndc[:]=[]
             words[:]=[]
             num[:]=[]
             word_index[:]=[]
            # mlist.clear()
             for x in range(c):
                mlist.append(0)
             v=0
             print("\n")
             print("---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n")
             print("\t\t\t\t\tMessage no:", p)
             print("\n\t\tInput message\n")
             print(i)
             print("Total Disaster Content Words present")
             words = nltk.word_tokenize(i)
             for k in nltk.word_tokenize(i):
             #for k in words:
                 for w in lst:
                    if(k==w and k in result):
                         print(k)
                         v=lst.index(w)
                         mlist[v] = mlist[v]+1
                 if( k not in result and k not in punctuation):
                     ndc.append(k)
                 if(k in casu):
                     t=words.index(k)
                     word_index.append(t)
                 if k.isdigit():
                     a=words.index(k)
                     num.append(a)
             for d in mlist:
                 if(d>0):
                     g=mlist.index(d)
                     if( lst[g] not in ndc):
                         ndc.append(lst[g])
             for q in word_index:
                    for y in num:
                         n=mod1(q,y)
                         if(n <=3):
                             a1=ndc.index(words[y])
                             ndc.pop(a1)
                             #ndc.remove(words[q])
                             ndc.append(words[q])
                             ndc.append(words[y])
             print("\nThe Counting array is as follows")
             print (mlist)
             p=p+1
             mlist[:]=[]
             print("\nOutput Message\n")
             #print(ndc)
             z=len(ndc)
             for i in range(z):
                 print(ndc[i],end=" ")
                 #of.write("".join(ndc[i]))
             for x in ndc:
                 of.write(x)
                 of.write(" ")
             of.write("\n")
     inf.close()
     sys.stdout.close()
def num_extrac():
     import nltk
     import csv
     from nltk.corpus import wordnet
     from nltk.tokenize import sent_tokenize, PunktSentenceTokenizer, word_tokenize
     from PyDictionary import PyDictionary
     synonyms = []
     word_index=[]
     num=[]
     words=[]
     lst=['death', 'dead', 'injure', 'injured', 'casualty', 'victim', 'damaged', 'killed']
     def mod1(a,b):
         if a>b:
             return a-b;
         else:
             return b-a;
     for i in lst: 
         for syn in wordnet.synsets(i):
             for l in syn.lemmas():
                 synonyms.append(l.name())
         #synonyms.extend(dictionary.synonym(i))
     #print(set(synonyms))
     g=1
     gg=open("num_extract.csv",'w',newline="")
     write=csv.writer(gg)
     #print("  Date&Time                           Information")
     #print("`````````````````````````````````````````````````")
     write.writerow(['Date','Time','Keyword','Quantity'])
     with open('t.txt') as f:
         for s in f:
             words[:]=[]
             num[:]=[]
             word_index[:]=[]
             words = nltk.word_tokenize(s)
             #print(words)
             for se in words:
                 if(se in set(synonyms)):
                     b=words.index(se)
                     word_index.append(b)
                 if se.isdigit():
                     a=words.index(se)
                     num.append(a)
             for w in word_index:
                 for k in num:
                     c=mod1(w,k)
                     if(c <=3):
                         #print(s[0:10],s[11:16],words[w], words[k])
                         write.writerow([s[0:10],s[11:16],words[w], words[k]])
                         #write.writerow("Date&time",s[:17],words[w],words[k])
             g=g+1 
     #print('See out11.csv from the folder')
     
def exp_smoothing():
     #Program to do exponential smoothing
     import numpy as np
     import pandas as pd
     import csv
     import os
     data=pd.read_csv('num_extract.csv')
     data.head()
     f1=data['Date'].values
     f2=data['Time'].values
     f3=data['Keyword'].values
     f4=data['Quantity'].values
     f=[]
     sm=[]
     cnt=0
     X=np.array(list(zip(f4)))
     '''print'length of data        ',len(X)
     for i in range(1,len(X)):
         print i,'\t\t',f1[i-1]'''
     cnt=cnt+1
     j=f4[0]
     sm.append(j)
     #print ('Enter the damping factor')
     #dmp=input('Damping Factor:')
     dmp=0.8
     for i in range(1,len(X)):
         k=int((dmp*X[i])+((1-dmp)*sm[i-1]))
         sm.append(k)
         cnt=cnt+1
     aa=X[i]
     bb=sm[i]
     gg=open('exponential_output.csv','w',newline="")
     writer=csv.writer(gg)
    # print ('Date        Time     word    number      Smoothed Values')
     #print ('----        ----     ----    ------      ---------------')
     writer.writerow(['Date','Time','Keyword','number','Smoothed Values'])
     for i in range(len(X)):
        # print (f1[i],'  ',f2[i],'  ',f3[i],'   ',f4[i],'             ',sm[i],'\n')
         writer.writerow([f1[i],f2[i],f3[i],f4[i],sm[i]])
         jj=f2[i]
     gg.close()
     da=pd.read_csv('exponential_output.csv')
     fi=open('exp_store.txt','w')
     writer2=csv.writer(fi)
     writer2.writerow(['Date','Time','word','number','Smoothed Values'])
    # print ('values to consider')
     #print (da.groupby('Keyword').tail(1))
     kt=da.groupby('Keyword').tail(1)
     kt.to_csv('final_filtered_exponential_outputp.csv')


     with open("final_filtered_exponential_outputp.csv","r",newline="") as source:
         rdr= csv.reader( source )
         with open("final_smooth_result.csv","w",newline="") as result:
             wtr= csv.writer( result )
             for r in rdr:
                 wtr.writerow( (r[1], r[2], r[3], r[4],r[5]) )
     os.remove('final_filtered_exponential_outputp.csv')
     
def master():
     import os
     import sys
     opti()
     first()
     second()
     num_extrac()
     exp_smoothing()
     os.remove('t.txt')
     
master()
