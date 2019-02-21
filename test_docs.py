from evaluate import test_doc
from random import shuffle
import os
import cv2

DOC_DIR = 'C:/Users/Rani/Desktop/Deep Project/project/document_dataset/test/'
OUTPUT_DIR = 'C:/Users/Rani/Desktop/Deep Project/project/test_docs/'
output_2x2_path = OUTPUT_DIR + '2x2/'
output_4x4_path = OUTPUT_DIR + '4x4/'
output_5x5_path = OUTPUT_DIR + '5x5/'

def make_shreds(image_file, tiles_per_dim, folder_index):
    if tiles_per_dim == 2:
        output_path = output_2x2_path
    if tiles_per_dim == 4:
        output_path = output_4x4_path
    if tiles_per_dim == 5:
        output_path = output_5x5_path
    im = cv2.imread(DOC_DIR + image_file, 0)
    # im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    height = im.shape[0]
    width = im.shape[1]
    frac_h = height // tiles_per_dim
    frac_w = width // tiles_per_dim
    i = 0
    for h in range(tiles_per_dim):
        for w in range(tiles_per_dim):
            crop = im[h * frac_h:(h + 1) * frac_h, w * frac_w:(w + 1) * frac_w]
            os.makedirs(output_path+str(folder_index), exist_ok=True)
            cv2.imwrite(output_path + str(folder_index) + '/' + image_file[:-4] + "_{}.jpg".format(i), crop)
            i = i + 1


def prepare_docs_crops(num_of_examples_per_t=10):
    os.makedirs(output_2x2_path, exist_ok=True)
    os.makedirs(output_4x4_path, exist_ok=True)
    os.makedirs(output_5x5_path, exist_ok=True)
    files = os.listdir(DOC_DIR)

    shuffle(files)
    i = 0
    for f in files[:num_of_examples_per_t]:
        i += 1
        make_shreds(f,2,i)

    shuffle(files)
    i = 0
    for f in files[:num_of_examples_per_t]:
        i += 1
        make_shreds(f, 4, i)

    shuffle(files)
    i = 0
    for f in files[:num_of_examples_per_t]:
        i += 1
        make_shreds(f, 5, i)


def test():
    total_acc = 0
    print('\n***Testing 2x2 docs***')
    image_folders = os.listdir(output_2x2_path)
    for image_folder in image_folders[:4]:
        acc = test_doc(output_2x2_path+image_folder+'/')
        total_acc += acc
        print('image#'+image_folder,'0-1 Accuracy =', acc, '%')
    print('***', '2x2 docs total accuracy =', total_acc / len(image_folders), '***')

    total_acc = 0
    print('\n***Testing 4x4 docs***')
    image_folders = os.listdir(output_4x4_path)
    for image_folder in image_folders[:5]:
        acc = test_doc(output_4x4_path+image_folder+'/')
        total_acc += acc
        print('image#'+image_folder,'0-1 Accuracy =', acc, '%')
    print('***', '4x4 docs total accuracy =', total_acc / len(image_folders), '***')

    total_acc = 0
    print('\n***Testing 5x5 docs***')
    image_folders = os.listdir(output_5x5_path)
    for image_folder in image_folders[:5]:
        acc = test_doc(output_5x5_path+image_folder+'/')
        total_acc += acc
        print('image#'+image_folder,'0-1 Accuracy =', acc, '%')
    print('***', '5x5 docs total accuracy =', total_acc/len(image_folders), '***')

# prepare_docs_crops(20)
test()





