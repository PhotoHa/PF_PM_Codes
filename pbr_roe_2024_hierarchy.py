# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 14:38:42 2024

@author: 11149
"""

from skimage.io import imread
from skimage.feature import hog
from scipy.spatial import distance
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import numpy as np
import glob
import pandas as pd
import os
os.chdir('T:\\index\\95_곽용하\\연구\\8_pbr_roe\\img_files2') # 이미지가 저장된 디렉토리 경로
image_directory = "T:\\index\\95_곽용하\\연구\\8_pbr_roe\\img_files2\\"   

df_0 = pd.read_csv("T:\\index\\95_곽용하\\연구\\8_pbr_roe\\pbr_roe.csv")
df_0 = df_0.iloc[:,1:]


# read the images and store in a list
images = [imread(file) for file in glob.glob("*.png")]

# number of images
n = len(images)

# creating a list to store HOG feature vectors
fd_list = []

fig = plt.figure(figsize=(6, 6))
k = 0

for i in range(n):

    # execute hog function for each image that is imported from skimage.feature module
    fd, hog_image = hog(images[i], orientations=9, pixels_per_cell=(
        64, 64), cells_per_block=(2, 2), visualize=True, multichannel=True)

    # add the feature vector to the list
    fd_list.append(fd)



# create an empty nxn distance matrix
distance_matrix = np.zeros((n, n))

for i in range(n):
    fd_i = fd_list[i]
    for k in range(i):
        fd_k = fd_list[k]
        # measure Jensen–Shannon distance between each feature vector
        # and add to the distance matrix
        distance_matrix[i, k] = distance.jensenshannon(fd_i, fd_k)

# symmetrize the matrix as distance matrix is symmetric
distance_matrix = np.maximum(distance_matrix, distance_matrix.transpose())
distance_matrix



# convert square-form distance matrix to vector-form distance vector (condensed distance matrix)
cond_distance_matrix = distance.squareform(distance_matrix)
cond_distance_matrix



Z = linkage(cond_distance_matrix, method='ward')

plt.figure(figsize=(12, 6))
dendrogram(Z, truncate_mode='level', p=2, color_threshold=0.2, show_leaf_counts=True)
plt.show()



from scipy.cluster.hierarchy import fcluster
threshold = 0.2
clusters = fcluster(Z, threshold, criterion='ditance')
clustered_data = []
for i, cluster_label in enumerate(clusters):
    if cluster_label not in clustered_data:
        clustered_data[cluster_label] = []
    else:
        clustered_data[cluster_label].append(i)
    
for cluster_label, data_indices in clustered_data.items():
    print()



from scipy.cluster.hierarchy import fcluster
# 임계값 설정
threshold = 0.2
desired_height = 2

# 각 관측치에 대한 군집 레이블 얻기
clusters = fcluster(Z, desired_height, criterion='distance')

# 각 군집에 속하는 관측치 식별
clustered_data = {}
for i, cluster_label in enumerate(clusters):
    if cluster_label not in clustered_data:
        clustered_data[cluster_label] = []
    clustered_data[cluster_label].append(i)

# 결과 출력
for cluster_label, data_indices in clustered_data.items():
    print("Cluster {}: {}".format(cluster_label, data_indices))


max_length = max(len(values) for values in clustered_data.values())

for key, values in clustered_data.items():
    clustered_data[key] += [None] * (max_length - len(values))

clustered_df = pd.DataFrame.from_dict(clustered_data, orient='index')


clst_ = ['C1','C2','C3','C4','C5','C6','C7']
clustered_df.index = clst_
clustered_df = clustered_df.transpose()
clustered_df = clustered_df.stack()
clustered_df = clustered_df.reset_index()
clustered_df = clustered_df.iloc[:,1:]
clustered_df.columns = ['cluster','case']
clustered_df = clustered_df.sort_values('case')




from PIL import Image
import numpy as np
import os

path_1 = "T:\\index\\95_곽용하\\연구\\8_pbr_roe\\rst_files\\"
directory = image_directory
image_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.jpg') or f.endswith('.png')]


for cl in clst_:
    c1 = clustered_df[clustered_df['cluster'] == cl]['case'].values
    c1 = [int(x) for x in c1]
    image_paths_ = [image_paths[x] for x in c1]
    images = [np.array(Image.open(img)) for img in image_paths_] # 이미지들을 로드하여 배열에 저장
    average_image = np.mean(images, axis=0).astype(np.uint8) # 이미지 배열들의 평균 계산
    Image.fromarray(average_image).save(path_1+f"{cl}_mean.png") # 평균 이미지를 이미지로 변환하여 보기















