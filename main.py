'''
This is the main file for HW 1 of FE 690
Author: Will Long, MS
Date: 09/15/2019
'''


import numpy as np
import matplotlib.pyplot as plt
import data_generation_functions as dgf
from sklearn.cluster import KMeans, DBSCAN
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers



''' First, we need to make the data.'''

data_1 = dgf.create_dataset_00()
data_2 = dgf.create_dataset_01()

'''Now, we need to the actual modeling.'''

km_1 = KMeans(n_clusters=4, random_state=0).fit(data_1)
km_2 = KMeans(n_clusters=3, random_state=0).fit(data_2)

db_1 = DBSCAN(eps=0.25, min_samples= 5).fit(data_1)
db_2 = DBSCAN(eps=0.5, min_samples= 2).fit(data_2)

'''Now, we need to try and make the tensorflow model.'''

model_km = tf.keras.Sequential([
    layers.Dense(64, activation='relu'),
    layers.Dense(4, activation='softmax')
])

model_km.compile(optimizer=tf.train.RMSPropOptimizer(0.01),
                 loss=tf.keras.losses.categorical_crossentropy,
                 metrics=[tf.keras.metrics.categorical_accuracy])

'''Now, we need to make the dataset for TF.'''
labels_1 = km_1.labels_
labels_2 = km_2.labels_


t_data_1, v_data_1, t_label_1, v_label_1 = train_test_split(data_1, labels_1, test_size=0.2)
t_data_2, v_data_2, t_label_2, v_label_2 = train_test_split(data_2, labels_2, test_size=0.2)

model_km.fit(t_data_1, t_label_1, epochs=10, batch_size=32, validation_data=(v_data_1, v_label_1))
result_labels_1 = model_km.predict(data_1)

def labelmaker(x):
    '''
    This should take the 2-D labels from result_labels and turn it into a 1-D array we can use for graphing.
    :param x: 2-D array
    :return: 1-D array
    '''
    a = []
    for i in range(0,len(x)):
        a.append(np.argmax(x[i]))
    return a

graph_labels_1 = labelmaker(result_labels_1)

def point_classifier_tf(x):
    '''
    This should take a 2-D point and classify it as one of the clusters from KMeans.
    :param x: 2-tuple 
    :return: Int. label from 0 to 3
    '''
    x_a = np.array([x,[0 , 0]])    # I don't know why I have to do this, but it works.
    a = model_km.predict(x_a)
    b = labelmaker(a)
    return b[0]









if __name__=="__main__":

    print(point_classifier_tf(data_1[7]))

    plt.subplot(231)
    plt.scatter(data_1[:, 0], data_1[:, 1], c=km_1.labels_, cmap='viridis')
    plt.title('Data 1: Kmean; 4 clusters')
    plt.subplot(232)
    plt.scatter(data_2[:, 0], data_2[:, 1], c=km_2.labels_, cmap='viridis')
    plt.title('Data 2: Kmean; 3 clusters')
    plt.subplot(233)
    plt.scatter(data_1[:, 0], data_1[:, 1], c=db_1.labels_, cmap='viridis')
    plt.title('Data 2: DBSCAN; eps=0.25, min_samples= 5')
    plt.subplot(234)
    plt.scatter(data_2[:, 0], data_2[:, 1], c=db_2.labels_, cmap='viridis')
    plt.title('Data 2: DBSCAN; eps=0.5, min_samples= 2')
    plt.subplot(235)
    plt.scatter(data_1[:, 0], data_1[:, 1], c=graph_labels_1, cmap='viridis')
    plt.title('Data 1: Tf-Kmean; 4 clusters')
    plt.show()



