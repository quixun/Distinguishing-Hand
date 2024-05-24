import pandas as pd
import numpy as np 
import os
import cv2
import mahotas 
import h5py
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

out_path = './Data/ImageOutPut/'
train_path = './Data/ImageRotated/Train/'
test_path = './Data/ImageRotated/Test/'
df = pd.read_csv("D:\DEV\WorkSpace-for-Python\Machine-Learning\MidtermTest\Data\dataset.csv")

# get the training labels
train_labels = os.listdir(train_path)
train_labels.sort()

# num of images per class
images_per_class = 400

# fixed-sizes for image
fixed_size = (100, 100)

# empty lists to hold feature vectors and labels
global_features = []
labels = []

# Đảm bảo số bins được định nghĩa
bins = 8

def fd_histogram(image, mask=None):
    # chuyển về không gian màu HSV
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    # normalize histogram
    cv2.normalize(hist, hist)
    return hist.flatten()

def fd_hu_moments(image):
    # chuyển về ảnh gray
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

def fd_haralick(image):
    # chuyển về ảnh gray
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    return haralick

# Loop over the training data sub-folders
if os.path.isdir(train_path):
    print(f"Directory exists: {train_path}")

    # Loop through each file in the training directory
    for filename in os.listdir(train_path):
        if filename.endswith(".png"):
            file = os.path.join(train_path, filename)

            print(f"Reading file: {file}")
            # Read the image from the path
            image = cv2.imread(file)
            if image is None:
                print(f"Could not read image {file}")
                continue

            # Check the size of the image before resizing
            if image.shape[0] == 0 or image.shape[1] == 0:
                print(f"Invalid image size for {file}")
                continue

            # Resize the image
            image = cv2.resize(image, fixed_size)

            # Extract global features
            fv_hu_moments = fd_hu_moments(image)
            fv_haralick = fd_haralick(image)
            
            print("fv_hu_moments:::", fv_hu_moments)
            print("fv_haralick:::", fv_haralick)

            # Check the size of the features
            print(f"Image: {filename}, Hu Moments: {fv_hu_moments.shape}, Haralick: {fv_haralick.shape}")

            # Concatenate the global features
            global_feature = np.hstack([fv_hu_moments, fv_haralick])

            # Update the list of labels and feature vectors
            label = filename.split('_')[0]  # giả sử nhãn là phần đầu của tên file
            labels.append(label)
            global_features.append(global_feature)
else:
    print(f"Directory does not exist: {train_path}")

print("[STATUS] completed Global Feature Extraction...")

# Check if any features were extracted
if len(global_features) == 0:
    print("[ERROR] No features extracted.")
else:
    # Convert the list to numpy array
    global_features = np.array(global_features)
    labels = np.array(labels)

    # Check the size of the feature vector and labels
    print(f"[STATUS] feature vector size: {global_features.shape}")
    print(f"[STATUS] training Labels: {labels.shape}")

    # Encode the labels
    le = LabelEncoder()
    # Đổi nhãn "T" thành 0 và nhãn "P" thành 1
    target = le.fit_transform(labels)

    # Normalize the feature vector
    scaler = MinMaxScaler(feature_range=(0, 1))
    rescaled_features = scaler.fit_transform(global_features)

    # Save the feature vector and labels to HDF5 files
    h5f_data = h5py.File(os.path.join(out_path, 'data.h5'), 'w')
    h5f_data.create_dataset('dataset_1', data=rescaled_features)

    h5f_label = h5py.File(os.path.join(out_path, 'labels.h5'), 'w')
    h5f_label.create_dataset('dataset_1', data=target)

    h5f_data.close()
    h5f_label.close()

    print("[STATUS] end of training..")
    clf = DecisionTreeClassifier()
    clf.fit(global_features, target)

    print("[STATUS] Model training completed...")

    # Chuẩn bị dữ liệu thử nghiệm
    test_labels = os.listdir(test_path)
    test_labels.sort()

    test_features = []
    test_results = []

    for testing_name in test_labels:
        filename = os.path.join(test_path, testing_name)
        current_label = testing_name.split('_')[0]  # giả sử nhãn là phần đầu của tên file
        print("filename:::", filename)
        if filename.endswith(".png"):
            image = cv2.imread(filename)
            if image is None:
                continue

            image = cv2.resize(image, fixed_size)

            fv_hu_moments = fd_hu_moments(image)
            fv_haralick = fd_haralick(image)

            global_feature = np.hstack([fv_hu_moments, fv_haralick])

            test_results.append(current_label)
            test_features.append(global_feature)

    # Kiểm tra nếu test_features rỗng
    if not test_features:
        print("[LỖI] Không trích xuất được đặc trưng từ ảnh thử nghiệm.")
    else:
        test_features = np.array(test_features)
        y_result = le.transform(test_results)
        y_pred = clf.predict(test_features)

        accuracy = (y_pred == y_result).tolist().count(True) / len(y_result)
        print("Kết quả: ", accuracy)