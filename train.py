#submission by acp22abj

import joblib
import numpy as np
import random

np.random.seed(7)

#Loading the full train joblib file
images, person_ids = joblib.load('train.full.joblib')

#print image shape for double check
print('Images shape:',images.shape)
print('Person ids shape:',person_ids.shape)


#Patch extraction for 30 x 30 pixels
from sklearn.feature_extraction import image

patch_extractor = image.PatchExtractor(patch_size=(30, 30), max_patches=2, random_state=0)

sub_images_30 = patch_extractor.transform(images)

print('Patch Size 30 Shape:' ,sub_images_30.shape)



#Rotating these patches to the left, right and upside down 
import joblib
import numpy as np

# Define orientations
orientations = [0, 1, 2, 3]  # 0, 90, 180, 270 degrees

# Initialize lists to store rotated images and labels
rotated_images_30 = []
labels_30 = []

# Iterate through each complete image
for image in sub_images_30:
    # Generate rotated sub-images and assign corresponding labels
    for orientation in orientations:
        rotated = np.rot90(image, k=orientation, axes=(0, 1))
        rotated_images_30.append(rotated)
        labels_30.append(orientation)

# Convert lists to NumPy arrays for efficient handling
rotated_images_30 = np.array(rotated_images_30)  
labels_30 = np.array(labels_30)

#Reshaping these images for the model building 
all_features_30 = rotated_images_30.reshape(len(rotated_images_30), -1)

print('Patched Resized 30 Shape:' ,all_features_30.shape)

print('Prepared Data for 30 Parsed')


#Patch extraction for 50 x 50 pixels

from sklearn.feature_extraction import image

patch_extractor = image.PatchExtractor(patch_size=(50, 50), max_patches=2, random_state=0)

sub_images_50 = patch_extractor.transform(images)

print('Patch Size 50 Shape:' ,sub_images_50.shape)


#Rotating these patches to the left, right and upside down 
import joblib
import numpy as np

# Define orientations
orientations = [0, 1, 2, 3]  # 0, 90, 180, 270 degrees

# Initialize lists to store rotated images and labels
rotated_images_50 = []
labels_50 = []

# Iterate through each complete image
for image in sub_images_50:
    # Generate rotated sub-images and assign corresponding labels
    for orientation in orientations:
        rotated = np.rot90(image, k=orientation, axes=(0, 1))
        rotated_images_50.append(rotated)
        labels_50.append(orientation)

# Convert lists to NumPy arrays for efficient handling
rotated_images_50 = np.array(rotated_images_50)  
labels_50 = np.array(labels_30)

#Reshaping these images for the model building 
all_features_50 = rotated_images_50.reshape(len(rotated_images_50), -1)
print('Patched Resized 50 Shape:' ,all_features_30.shape)

print('Prepared Data for 50 Parsed')


#Patch extraction for 90 x 90 pixels

from sklearn.feature_extraction import image

patch_extractor = image.PatchExtractor(patch_size=(90, 90), max_patches=2, random_state=0)

sub_images_90 = patch_extractor.transform(images)

print('Patch Size 90 Shape:' ,sub_images_90.shape)


#Rotating these patches to the left, right and upside down 
import joblib
import numpy as np

# Define orientations
orientations = [0, 1, 2, 3]  # 0, 90, 180, 270 degrees

# Initialize lists to store rotated images and labels
rotated_images_90 = []
labels_90 = []

# Iterate through each complete image
for image in sub_images_90:
    # Generate rotated sub-images and assign corresponding labels
    for orientation in orientations:
        rotated = np.rot90(image, k=orientation, axes=(0, 1))
        rotated_images_90.append(rotated)
        labels_90.append(orientation)

# Convert lists to NumPy arrays for efficient handling
rotated_images_90 = np.array(rotated_images_90)  # Using dtype=object for variable-sized arrays
labels_90 = np.array(labels_90)

#Reshaping these images for the model building 

all_features_90 = rotated_images_90.reshape(len(rotated_images_90), -1)
print('Patched Resized 90 Shape:' ,all_features_90.shape)

print('Prepared Data for 90 Parsed')


from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from tempfile import mkdtemp
from joblib import Memory, dump
import warnings
warnings.filterwarnings('ignore')
import joblib
import random


# Define the pipeline with the best parameters
pipeline_30 = Pipeline([
    ('scaler', StandardScaler()), 
    ('pca', PCA(n_components=80)), 
    ('mlp', MLPClassifier(hidden_layer_sizes=(300, 300), max_iter=1000, random_state=42)) 
])

# Train the model using the entire dataset
pipeline_30.fit(all_features_30, labels_30)

joblib.dump(pipeline_30, 'model.30.joblib')
print('Model Saved for 30')

# Define the pipeline with the best parameters
pipeline_50 = Pipeline([
    ('scaler', StandardScaler()),  
    ('pca', PCA(n_components=100)),  
    ('mlp', MLPClassifier(hidden_layer_sizes=(340, 340), max_iter=1000, random_state=42)) 
])

# Train the model using the entire dataset
pipeline_50.fit(all_features_50, labels_50)

joblib.dump(pipeline_50, 'model.50.joblib')
print('Model Saved for 50')


# Define the pipeline with the best parameters and a random seed
pipeline_90 = Pipeline([
    ('scaler', StandardScaler()), 
    ('pca', PCA(n_components=70)),  
    ('mlp', MLPClassifier(hidden_layer_sizes=(340, 340), max_iter=500, random_state=42))
])

# Train the model using the entire dataset
pipeline_90.fit(all_features_90, labels_90)


joblib.dump(pipeline_90, 'model.90.joblib')
print('Model Saved for 90')







