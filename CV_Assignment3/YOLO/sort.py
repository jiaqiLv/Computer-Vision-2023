import os
import shutil

dataset_dir = ['./photo_mypart', './xianxia']

images_dir = '../datasets/instant_noodles/images'
labels_dir = '../datasets/instant_noodles/labels'

for dir in dataset_dir:
    filenames = os.listdir(dir)
    # print('filenames:', filenames)
    for file in filenames:
        if file.split('.')[-1] == 'jpg':
            # print(file)
            shutil.copy(os.path.join(dir, file), images_dir)
        elif file.split('.')[-1] == 'txt':
            # print(file)
            shutil.copy(os.path.join(dir, file), labels_dir)
