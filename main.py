import pandas as pd
import numpy as np
import cv2
import mahotas 
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

def fd_hu_moments(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(gray)).flatten()
    return feature

def fd_haralick(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    return haralick

def predict_image(image, clf, le, scaler):
    fv_hu_moments = fd_hu_moments(image)
    fv_haralick = fd_haralick(image)
    global_feature = np.hstack([fv_hu_moments, fv_haralick])
    global_feature = scaler.transform(global_feature.reshape(1, -1))
    prediction = clf.predict(global_feature)
    predicted_label = le.inverse_transform(prediction)[0]
    return predicted_label

# Đọc dataframe từ file csv
df = pd.read_csv("./Data/dataset.csv")

# Lọc ra các dòng có Angle là số
df = df[pd.to_numeric(df['Angle'], errors='coerce').notnull()]

# Chuẩn bị dữ liệu huấn luyện
train_features = []
train_labels = []

for index, row in df.iterrows():
    # Đọc đường dẫn đến ảnh
    image_path = f"./Data/ImageRotated/{row['Folder']}/{row['ID']}.png"
    # Đọc ảnh
    image = cv2.imread(image_path)
    # Trích xuất đặc trưng
    fv_hu_moments = fd_hu_moments(image)
    fv_haralick = fd_haralick(image)
    global_feature = np.hstack([fv_hu_moments, fv_haralick])
    train_features.append(global_feature)
    # Gán nhãn
    label = row['Labels']
    train_labels.append(label)

# Chuyển đổi danh sách đặc trưng và nhãn sang numpy array
train_features = np.array(train_features)
train_labels = np.array(train_labels)

# Encode nhãn
le = LabelEncoder()
target = le.fit_transform(train_labels)

# Chuẩn hóa đặc trưng
scaler = MinMaxScaler(feature_range=(0, 1))
rescaled_features = scaler.fit_transform(train_features)

# Xây dựng mô hình
clf = DecisionTreeClassifier()
clf.fit(rescaled_features, target)

# Đọc ảnh cần dự đoán
image_path_to_predict = "./Data/Image/2388.png"
image = cv2.imread(image_path_to_predict)

# Dự đoán
predicted_label = predict_image(image, clf, le, scaler)
print("The predicted label for the image is:", predicted_label)
