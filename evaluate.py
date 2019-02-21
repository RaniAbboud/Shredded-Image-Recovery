import os
import cv2
from evaluate_aux import solve_doc
from evaluate_aux import solve_image
# from numpy.random import permutation
# import re
import image_doc_classifier
import documents_2


def predict(images):
    labels = []
    # here comes your code to predict the labels of the images
    if image_doc_classifier.is_image(images):
        return solve_image(images)
    else:  # is document
        if len(images) == 4:
            labels = documents_2.recover_document(images)
        else:
            labels = solve_doc(images)
    return labels


def evaluate(file_dir='example/'):
    files = os.listdir(file_dir)
    files.sort()
    images = []
    for f in files:
        im = cv2.imread(file_dir + f)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        images.append(im)

    Y = predict(images)
    return Y


# def test_image(file_dir='example/'):  # for debugging
#     files = os.listdir(file_dir)
#     files.sort()
#     images = []
#     for f in files:
#         im = cv2.imread(file_dir + f, 0)
#         # im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
#         images.append(im)
#
#     actual = [int(re.search('\._(.+).jpg', filename).group(1)) for filename in files]
#     perm = list(permutation(range(len(images))))
#     # print('perm=', [actual[perm[i]] for i in range(len(images))])
#     permutated_images = []
#     for i in range(len(images)):
#         permutated_images.append(images[perm[i]])
#         # cv2.imshow('title',images[perm[i]])
#         # print((images[perm[i]]).shape)
#         # cv2.waitKey(0)
#         # cv2.destroyAllWindows()
#
#     Y = predict(permutated_images)
#     accuracy = 100*(sum([Y[i]==actual[perm[i]] for i in range(len(images))])/len(images))
#     # print('0-1 Accuracy =', accuracy, '%')
#     return accuracy
#
#
# def test_doc(file_dir='example/'):  # for debugging
#     files = os.listdir(file_dir)
#     # files.sort()
#     images = []
#     for f in files:
#         im = cv2.imread(file_dir + f, 0)
#         # im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
#         images.append(im)
#
#     actual = [int(re.search('.*_(.+)\.jpg$', filename).group(1)) for filename in files]
#     perm = list(permutation(range(len(images))))
#
#     permutated_images = []
#     for i in range(len(images)):
#         permutated_images.append(images[perm[i]])
#         # cv2.imshow('permutated',images[perm[i]])
#         # print((images[perm[i]]).shape)
#         # cv2.waitKey(0)
#         # cv2.destroyAllWindows()
#
#     Y = predict(permutated_images)
#     # print('predicted=',Y)
#     #return 0 ########################################## testing rows
#     accuracy = float(100)*(float(sum([Y[i]==actual[perm[i]] for i in range(len(images))]))/float(len(images)))
#     # print('0-1 Accuracy =', accuracy, '%')
#     return accuracy


# test_image()
