import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def distance_between_points(x1,x2):
    return np.sqrt(np.sum((x1-x2)**2))
class Kmeans:
    def __init__(self,k=4, max_iteration=100, plot_steps=False):
        self.k=k
        self.max_iteration=max_iteration
        self.plot_steps=False
        
        #list of data for each clusters
        self.cluster=[[] for _ in range(self.k)]
        
        # mean vector for each cluster
        self.mean=[]
        
    def future(self, X):
        self.X=X
        self.No_of_samples,self.No_of_dimensions=X.shape
        
        # initialize the means 
        random_sample_index=np.random.choice(self.No_of_samples,self.k,replace=False)
        self.mean = [self.X[idx] for idx in random_sample_index]
        self.error=[]
        self.count=0
        
        #optimisizing the data 
        for _ in range(self.max_iteration):
            #updating the clusters
            
            self.cluster = self.creatingtheclusters(self.mean)

            #updating the previous means
            mean_old = self.mean
            
            distance=[]
            for index, sample in enumerate(self.X):
                cluster_idx=self.closest_mean(sample, mean_old)
                distance.append(distance_between_points(sample, mean_old[cluster_idx]))
            self.error.append(np.sum(distance))
            self.count=self.count+1
            
            
            self.mean = self._get_mean(self.cluster)
            


            # convergence check 
            if self.converged(mean_old, self.mean):
                break
                
                
        return self._get_mean_labels(self.cluster)     
        
    def creatingtheclusters(self, mean):
        self.cluster=[[] for _ in range(self.k)]
        for index, sample in enumerate(self.X):
            mean_index = self.closest_mean(sample, mean)
            self.cluster[mean_index].append(index)
        return self.cluster
    
    
    def closest_mean(self, sample, mean):
        distance = [distance_between_points(sample, point) for point in mean]
        closest_index = np.argmin(distance)
        return closest_index
    
    def _get_mean(self, cluster):
        mean = np.zeros((self.k, self.No_of_dimensions))
        for cluster_index, cluster in enumerate(cluster):
            cluster_mean=np.mean(self.X[cluster], axis=0)
            mean[cluster_index] = cluster_mean
        return mean
    
    
    def converged(self, mean_old, mean):
        distance = [distance_between_points(mean_old[i], mean[i]) for i in range(self.k)]
        return sum(distance)==0
    
    def _get_mean_labels(self,cluster):
        labels = np.empty(self.No_of_samples)
        for cluster_index, cluster in enumerate(cluster):
            for sample_index in cluster:
                labels[sample_index] = cluster_index
        return labels
    def plot(self):
        fig, ax =plt.subplots(figsize=(5,5))
        
        
        for i,index in enumerate(self.cluster):
            point = self.X[index].T
            ax.scatter(*point)
            
        for point in self.mean:
            ax.scatter(*point, marker="x", color="black", linewidth=2)
        plt.title("final clusters with kmeans++ on the data_set")
        plt.xlabel("X")
        plt.ylabel("y")
        plt.show()
        
        array = np.empty(self.count, dtype=object) 
        for i in range(self.count):
            array[i]=i+1
#         plt.scatter(array,self.error)
#         plt.title("")
#         plt.plot(array,self.error)
#         plt.show()
        
if __name__ == "__main__":
   
    data_set=pd.read_csv("Dataset.csv", sep=",", header=None)
    data_set.columns=["x", "y"]
    X=np.array(data_set)
    for i in range(2,6):
        k=Kmeans(k=i,max_iteration=100, plot_steps=True)
        y=k.future(X)
        k.plot()