import os
from PIL import Image
import pandas as pd

source_image = './Data/Image'
output_image_folder = './Data/ImageRotated/'
directory = os.listdir(source_image)

df = pd.read_csv('./Data/dataset.csv')

angle_column = df['Angle']

angles = []
for angle in angle_column:
    try:
        angles.append(float(angle))
    except ValueError:
        continue
angles = angles[:300]

num_images = len(directory)
num_angles = len(angles)

num_images_to_save = 40
for i in range(min(num_images_to_save, min(num_images, num_angles))):
    img_path = os.path.join(source_image, directory[i])
    img = Image.open(img_path)
    angle = angles[i]
    rotated_img = img.rotate(angle, expand=True)
    output_path = os.path.join(output_image_folder, f'rotated_image_{i+20}.png')
    # rotated_img.save(output_path)
    rotated_img.show()
