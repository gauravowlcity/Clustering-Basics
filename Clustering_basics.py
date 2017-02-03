#importing modules
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
#reading the csv file
votes=pd.read_csv("114_congress.csv")
votes.head(10)
#finding no. of candidates of each party
votes['party'].value_counts()
votes['party'].value_counts().plot(kind='bar',color=['grey','g','brown'])
plt.tick_params(bottom='off',top='off',right='off',left='off')
plt.show()
#Starting with clustering
from sklearn.metrics.pairwise import euclidean_distances
distance=euclidean_distances(votes.iloc[0,3:].reshape(1,-1),votes.iloc[1,3:].reshape(1,-1))
distance
from sklearn.cluster import KMeans
kmeans_model=KMeans(n_clusters=2,random_state=1)
senator_distances=kmeans_model.fit_transform(votes.iloc[:,3:])
labels=kmeans_model.labels_
pd.crosstab(labels,votes['party'])
pd.crosstab(labels,votes['party']).plot(kind='bar',stacked=True)
x=[0,1]
l=['cluster1','cluster2']
plt.xticks(x,l)
plt.title('Clustering')
plt.xlabel('Clusters')
plt.ylabel('No. of Senators')
plt.tick_params(bottom='off',top='off',right='off',left='off')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()
democratic_outlier=votes[(labels==1) & (votes['party']=='D')]
democratic_outlier
independents_like_democrats=votes[(labels==0) &(votes['party']=='I')]
independents_like_democrats
plt.scatter(x=senator_distances[:,0],y=senator_distances[:,1],c=labels)
plt.show()
#most extreme 
extremism = (senator_distances ** 3).sum(axis=1)
votes['extremism']=extremism
votes.sort_values('extremism',inplace=True,ascending=False)
