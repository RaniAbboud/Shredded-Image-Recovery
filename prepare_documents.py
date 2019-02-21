import cv2
import os
import numpy as np
from random import shuffle
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import random

DOC_DIR = 'C:/Users/Rani/Desktop/Deep Project/project/documents/'
cwd = 'C:/Users/Rani/Desktop/Deep Project/project/'
dataset_path = cwd + 'document_dataset/'
test_path = dataset_path + 'test/'
train_path = dataset_path + 'train/'

def split_data_to_train_test(data, test_size=0.2):
    print('total number of examples:', len(data))
    shuffle(data)
    test_files = data[:int(len(data) * test_size)]
    train_files = data[int(len(data) * test_size):]
    print('test set size:', len(test_files), ' train set size:', len(train_files))
    for file in train_files:
        img = cv2.imread(DOC_DIR+file, 0)
        cv2.imwrite(train_path+file, img)
    for file in test_files:
        img = cv2.imread(DOC_DIR + file, 0)
        cv2.imwrite(test_path + file, img)
    return train_files, test_files


def resize(im, ratio=0.5):
    return cv2.resize(im, (0, 0), fx=ratio, fy=ratio)


def create_legit_combinations(files, tiles_per_dim, ratio=0.5, limit=5000):
    X = []
    # create images of legit combinations
    for f in files:
        if len(X) >= limit:
            break
        im = cv2.imread(train_path+f, 0)
        for h in range(tiles_per_dim):
            for w in range(tiles_per_dim-1):
                crop1 = img_get_crop(im, tiles_per_dim, h, w)
                crop2 = img_get_crop(im, tiles_per_dim, h, w+1)

                im2 = np.concatenate((crop1, crop2), axis=1)  # crop1 to the left of crop2
                X += [resize(im2, ratio)]
    return np.array(X)


def create_legit_combinations_up_down(files, tiles_per_dim, ratio=0.5, limit=5000):
    X = []
    # create images of legit combinations
    for f in files:
        if len(X) >= limit:
            break
        im = cv2.imread(train_path+f, 0)
        for w in range(tiles_per_dim):
            for h in range(tiles_per_dim-1):
                crop1 = img_get_crop(im, tiles_per_dim, h, w)
                crop2 = img_get_crop(im, tiles_per_dim, h+1, w)

                im2 = np.concatenate((crop1, crop2), axis=0)  # crop1 on crop2
                X += [resize(im2, ratio)]
    return np.array(X)


def img_get_crop(img, tiles_per_dim, i, j):
    if (i not in range(tiles_per_dim)) or (j not in range(tiles_per_dim)):
        print('img_get_crop Error: index out of range')
        exit()
    height = img.shape[0]
    width = img.shape[1]
    frac_h = height // tiles_per_dim
    frac_w = width // tiles_per_dim
    crop = img[i*frac_h: (i+1)*frac_h , j*frac_w: (j+1)*frac_w]
    return crop


# def img_get_row(img, tiles_per_dim, row):
#     row_im = []
#     for col in range(tiles_per_dim):
#         row_im += [img_get_crop(img, tiles_per_dim, row, col)]
#     return np.concatenate(row_im, axis=1)


# def create_synthetic_combinations(files,tiles_per_dim,ratio=0.5, limit=10000):
#     X = []
#     # create images of NON-legit combinations
#     for f in files:
#         if len(X) >= limit:
#             break
#         im = cv2.imread(train_path + f, 0)
#
#         rand_row1 = random.randint(0, tiles_per_dim)
#         rand_row2 = random.randint(0, tiles_per_dim)
#         crop1 = img_get_crop(im, tiles_per_dim, rand_row1, 0)
#         crop2 = img_get_crop(im, tiles_per_dim, rand_row2, tiles_per_dim - 1)
#         im1 = np.concatenate((crop2, crop1), axis=1)  # crop2 to the left of crop1 (white part on white part)
#         X += [resize(im1, ratio)]
#
#         for j in range(tiles_per_dim - 1):
#             crop1 = img_get_crop(im, tiles_per_dim, j, j)
#             if tiles_per_dim > 2:
#                 crop2 = img_get_crop(im, tiles_per_dim, j, (j+2) % tiles_per_dim)
#                 crop3 = img_get_crop(im, tiles_per_dim, j, (j+3) % tiles_per_dim)
#
#                 im1 = np.concatenate((crop1, crop2), axis=1)  # crop1 to the left of crop2
#                 X += [resize(im1, ratio)]
#
#                 im2 = np.concatenate((crop1, crop3), axis=1)  # crop1 to the left of crop3
#                 X += [resize(im2, ratio)]
#
#             for k in range(j + 1, tiles_per_dim):  # walk along diagonal
#                 crop2 = img_get_crop(im, tiles_per_dim, k, k)
#
#                 im1 = np.concatenate((crop1, crop2), axis=1)  # crop1 to the left of crop2
#                 X += [resize(im1, ratio)]
#
#                 im3 = np.concatenate((crop2, crop1), axis=1)  # crop2 to the left of crop1
#                 X += [resize(im3, ratio)]
#     return np.array(X)


def create_synthetic_combinations2(files,tiles_per_dim, possible_pairs, ratio=0.5, limit=10000):
    X = []
    # create images of NON-legit combinations
    for f in files:
        if len(X) >= limit:
            break
        im = cv2.imread(train_path + f, 0)
        possible = possible_pairs.copy()
        count = 0
        while count < 40:
            crop1_ind, crop2_ind = random.choice(possible)
            crop1 = img_get_crop(im, tiles_per_dim, crop1_ind[0], crop1_ind[1])
            crop2 = img_get_crop(im, tiles_per_dim, crop2_ind[0], crop2_ind[1])
            im1 = np.concatenate((crop1, crop2), axis=1)  # crop1 to the left of crop2
            X += [resize(im1, ratio)]
            possible.remove((crop1_ind, crop2_ind))
            count += 1
    return np.array(X)


def create_synthetic_combinations2_up_down(files,tiles_per_dim, possible_pairs, ratio=0.5, limit=10000):
    X = []
    # create images of NON-legit combinations
    for f in files:
        if len(X) >= limit:
            break
        im = cv2.imread(train_path + f, 0)
        possible = possible_pairs.copy()
        count = 0
        while count < 20:
            crop1_ind, crop2_ind = random.choice(possible)
            crop1 = img_get_crop(im, tiles_per_dim, crop1_ind[0], crop1_ind[1])
            crop2 = img_get_crop(im, tiles_per_dim, crop2_ind[0], crop2_ind[1])
            im1 = np.concatenate((crop1, crop2), axis=0)  # crop1 on crop2
            X += [resize(im1, ratio)]
            possible.remove((crop1_ind, crop2_ind))
            count += 1
    return np.array(X)


# def create_legit_docs(files, tiles_per_dim, ratio=0.1, limit=2000):
#     X = []
#     # create images of NON-legit combinations
#     for f in files:
#         if len(X) >= limit:
#             break
#         im = cv2.imread(train_path + f, 0)
#
#         X += [resize(im, ratio)]
#         return np.array(X)
#
#
# def create_synthetic_docs(files,tiles_per_dim, ratio=0.1, limit=10000):
#     X = []
#     # create images of NON-legit combinations
#     for f in files:
#         if len(X) >= limit:
#             break
#         im = cv2.imread(train_path + f, 0)
#
#         used = []
#         while len(X) < 10:
#             perm = np.random.permutation(range(tiles_per_dim))
#             if perm in used:
#                 continue  # generate another permutation
#             used += [perm]
#
#             perm_image = [img_get_row(im, tiles_per_dim, row) for row in perm]
#             perm_image = np.concatenate(perm_image, axis=0)
#
#             X += [resize(perm_image, ratio)]
#     return np.array(X)


def prepare():
    # files = os.listdir(DOC_DIR)

    # update this number for 4X4 crop 2X2 or 5X5 crops.
    # tiles_per_dim = 4
    # synthetic_path = cwd + 'training_synthetic/'
    # legit_path = cwd + 'training_legit/'
    # os.makedirs(synthetic_path, exist_ok=True)
    # os.makedirs(legit_path, exist_ok=True)

    cwd = 'C:/Users/Rani/Desktop/Deep Project/project/'
    dataset_path = cwd + 'document_dataset/'
    test_path = dataset_path + 'test/'
    train_path = dataset_path + 'train/'

    os.makedirs(dataset_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)
    os.makedirs(train_path, exist_ok=True)

    # train_files, test_files = split_data_to_train_test(files, 0.2)
    train_files = os.listdir(train_path)
    test_files = os.listdir(test_path)

    tiles_per_dim = 5

    # X1 = create_legit_combinations_up_down(train_files, tiles_per_dim, 0.25, 6000)
    # X1 = create_legit_docs(train_files, tiles_per_dim, 0.1, 6000)
    X1 = create_legit_combinations(train_files, tiles_per_dim, 0.3125, 10000)
    y1 = np.ones((X1.shape[0]))

    pairs = [(i,j) for i in range(tiles_per_dim) for j in range(tiles_per_dim)]
    possible_pairs = [(pair1,pair2) for pair1 in pairs for pair2 in pairs if
                    (pair1 != pair2) and not(pair1[0]==pair2[0] and pair1[1]==(pair2[1]-1))]
    # possible_pairs = [(pair1,pair2) for pair1 in pairs for pair2 in pairs if
    #                   (pair1[1] == pair2[1]) and (pair1 != pair2) and not(pair1[0] == (pair2[0]-1))]
    X2 = create_synthetic_combinations2(train_files, tiles_per_dim, possible_pairs, 0.3125, 25000)
    # X2 = create_synthetic_docs(train_files, tiles_per_dim, 0.1, 12000)
    y2 = np.zeros(X2.shape[0])

    print('X1 shape ',X1.shape)
    print('y1 shape ',y1.shape)
    print('X2 shape ',X2.shape)
    print('y2 shape ',y2.shape)

    X = np.concatenate((X1,X2))
    y = np.concatenate((y1,y2))
    y = to_categorical(y, num_classes=2)
    print('X shape ',X.shape)
    print('y shape ',y.shape)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y, shuffle=True)
####################################################
    dataset_path = dataset_path + '5x5_side/'
    os.makedirs(dataset_path, exist_ok=True)
####################################################
    np.save(dataset_path+'X_train', X_train)
    np.save(dataset_path+'X_val', X_val)
    np.save(dataset_path+'y_train', y_train)
    np.save(dataset_path+'y_val', y_val)

    for i in range(0,100,5):
        cv2.imshow(str(y_train[i]), X_train[i])
        print('label=',str(y_train[i]),'  ', str(X_train[i].shape))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    for i in range(0,100,5):
        cv2.imshow(str(y_val[i]), X_val[i])
        print('label=',str(y_val[i]),'  ', str(X_val[i].shape))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


#prepare()
