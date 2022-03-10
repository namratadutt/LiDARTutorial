# Import libraries
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl 
import scipy.io
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay




# Get path to your current directory
basedir = os.getcwd()


# Path tp your dataset
filename = basedir + "/muufl_gulfport_campus_1_hsi_220_label.mat"

# Open .mat file with scipy
mat = scipy.io.loadmat(filename)

hsi = ((mat['hsi'])[0])[0]

# RGB Image
rgbIm = hsi[-1]

# Ground truth
truth = ((hsi[-2])[0])[-1]
truth = truth[-1]

# LiDAR
lidar = ((((hsi[-4])[0])[0])[0])[0]

# x, y, z. z contains Height and Intensity
x, y, z, info = lidar[0], lidar[1], lidar[2], lidar[3]

# Height and Intesity of LiDAR
height = z[:,:,0]
intensity = z[:,:,1]


# Plot the intensity
fig, ax = plt.subplots(1, figsize = (10, 10))
plt.imshow(intensity, cmap= 'viridis')
plt.title("LiDAR Intensity", fontsize= 22)
plt.colorbar()
plt.savefig(basedir + "/lidar_medium/Intensity plot of Gulfport with noise.png")
plt.close()


# Plot Histogram of Intensity
plt.hist(intensity, bins= 100)
plt.savefig(basedir +"/lidar_medium/Histogram of Intensity of Gulfport.png")
plt.close('all')


# Remove noise from Intensity
intensity[intensity >= 223] = np.mean(intensity)
fig, ax = plt.subplots(1, figsize = (10, 10))
plt.imshow(intensity, cmap= 'viridis')
plt.title("LiDAR Intensity", fontsize= 22)
plt.colorbar()
plt.savefig(basedir +"/lidar_medium/Intensity plot of Gulfport without noise.png")
plt.close('all')

# Plot the Height
fig, ax = plt.subplots(1, figsize = (10, 10))
plt.imshow(height, cmap= 'viridis')
plt.title("LiDAR Height", fontsize= 22)
plt.colorbar()
plt.savefig(basedir + "/lidar_medium/Height plot of Gulfport.png")
plt.close('all')

# Plot Histogram of Height
plt.hist(height, bins= 40)
plt.savefig(basedir +"/lidar_medium/Histogram of Height of Gulfport.png")
plt.close('all')


# Create meshgrid
# Shape of Height and Intensity is same
row, col = intensity.shape
x = np.arange(0, col, 1)
y = np.arange(0, row, 1)
xx, yy = np.meshgrid(x, y)

# 3D visualization of Intensity
fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot(111, projection='3d')
I = ax.scatter(xx, yy, intensity, c=intensity, s= 3, cmap= 'coolwarm')
fig.colorbar(I, ax= ax)
plt.show()

# 3D visualization of Height
fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot(111, projection='3d')
I = ax.scatter(xx, yy, height, c=height, s= 3, cmap= 'coolwarm')
fig.colorbar(I, ax= ax)
plt.show()

# Array of class labels
a = [1, 2, 5, 7, 11]
# Dictionary of Classes and their labels
landcover_dict = {1: 'Trees', 2: 'Mostly Grass', 5: 'Road', 7: 'Building', 11: 'Cloth Panel'}
colors = ['darkgreen', 'limegreen', 'brown', 'cyan', 'red']

# Set different font sizes for plots
mpl.rcParams['xtick.labelsize'] = 22
mpl.rcParams['ytick.labelsize'] = 22 
mpl.rcParams['legend.fontsize'] = 22 
mpl.rcParams['axes.grid'] = True

# Height of different classes

# Find index of only 1, 2, 5, 7 and 11 labels
indx, indy = np.where((truth == 1) | (truth == 2) |(truth == 5) |(truth == 7) |(truth == 11))
# Find height at above mentioned labels
height_class = height[indx, indy]

# Find Global maxima and Global minima height of the classes
global_minima = round(np.min(height_class),2)
global_maxima = round(np.max(height_class),2)

count = 0
fig, ax = plt.subplots(5, figsize= (30, 20), dpi= 200)

for i in a:
    indx, indy = np.where(truth == i)
    class_height = height[indx, indy]
    ax[count].hist(class_height, bins = 120, color = colors[count], label= landcover_dict[i], alpha= 0.6)
    ax[count].set_xticks(np.arange(global_minima, global_maxima, 2))
    count +=1

fig.legend()
fig.suptitle("Histogram of height of different landcover classes", fontsize= 36)
plt.savefig(basedir + "/lidar_medium/"+"/Gulfport_StackedHeight.png")
plt.close()

# Intensity of different Classes

# Find index of only 1, 2, 5, 7 and 11 labels
indx, indy = np.where((truth == 1) | (truth == 2) |(truth == 5) |(truth == 7) |(truth == 11))
# Find intensity at above mentioned labels
intensity_class = intensity[indx, indy]

# Find Global maxima and minima of Intensity of the classes
global_minima = round(np.min(intensity_class),2)
global_maxima = round(np.max(intensity_class),2)

count = 0
fig, ax = plt.subplots(5, figsize= (30, 20), dpi= 200)

for i in a:

    indx, indy = np.where(truth == i)
    class_intensity = intensity[indx, indy]
    ax[count].hist(class_intensity, bins = 150, color = colors[count], label= landcover_dict[i], alpha= 0.6)
    ax[count].set_xticks(np.arange(global_minima, global_maxima, 20))
    count +=1

fig.legend()
fig.suptitle("Histogram of Intensity of different landcover classes", fontsize= 36)
plt.savefig(basedir + "/lidar_medium/"+"/Gulfport_StackedIntensity.png")
plt.close()

# Reshape LiDAR data (3D into 2D)
z = z.reshape(325*220,2)
# Reshape Ground Truth
truth = truth.flatten()

class_names = ["Trees", "Mostly Grass", "Road", "Building", "Cloth Panel"]
# There are other labels in ground truth data. They are for future use after processing.
landcover_dict = {1: 'Trees', 2: 'Mostly Grass', 5: 'Road', 7: 'Building', 11: 'Cloth Panel'} 

# Find index of classes
indx, = np.where((truth == 1) | (truth == 2) | (truth == 5) | (truth == 7) | (truth == 11))
# Extract ground truth for above mentioned lables
truth = truth[indx]
# Extract Height and Intensity for above mentioned lables
z = z[indx]

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(z, truth, test_size= 0.3)

# KNN Classifier

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# You can put a loop to find out what value of neighbors gives best accuracy
classifier = KNeighborsClassifier(n_neighbors = 31, metric = 'euclidean', p = 2)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

acc = accuracy_score(y_test,y_pred)
print("Accuracy : ", np.round(acc,4)*100, "%")

# Compute Confusion matrix
cm = confusion_matrix(y_test, y_pred, normalize= 'true')
cm = np.round(cm, 3)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

disp.plot()
plt.savefig(basedir + "/lidar_medium/"+"/KNN_Confusion_Matrix for Gulfport.png")
plt.close()