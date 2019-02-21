from prepare_data import resize_and_make_border
from legit_img_model import Network
from legit_doc_model import Network_doc4x4
import numpy as np
from math import sqrt
import cv2


def show_row(images, row):  # for debugging
    row_image = np.concatenate([images[i] for i in row], axis=1)  # concatenate images in row to make a single image
    cv2.imshow('row image', row_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return row_image

def get_row_image(images, row):  # for debugging
    row_image = np.concatenate([images[i] for i in row], axis=1)  # concatenate images in row to make a single image
    return row_image


def apply_transform(img, t2=-1, is_image=True):
    if is_image:
        return resize_and_make_border(img, 100)
    else: # is document
        tiles_per_dim = int(sqrt(t2))
        if tiles_per_dim == 2:
            concat = cv2.resize(img, (0, 0), fx=0.125, fy=0.125)
        if tiles_per_dim == 4:
            concat = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
        if tiles_per_dim == 5:
            concat = cv2.resize(img, (0, 0), fx=0.3125, fy=0.3125)
        return np.reshape(concat, (concat.shape[0], concat.shape[1], 1))


def get_pair_probs(images, clf, is_image=True):
    # returns a matrix (2darray) containing the probabilities for concatenating crops
    # index (i,j) contains the probability that concat'ing images[i],images[j] (in this order) will yield a legit image
    t = len(images)

    concats = []
    indices = []
    pairs = [(i,j) for i in range(t) for j in range(t) if i!=j]
    for (ind1, ind2) in pairs:
        im1 = images[ind1]
        im2 = images[ind2]
        concat = np.concatenate((im1, im2), axis=1)  # im1 to the left of im2
        if is_image:
            concat = apply_transform(concat)
        else:  # is document
            tiles_per_dim = int(sqrt(t))
            if tiles_per_dim == 2:
                concat = cv2.resize(concat, (0, 0), fx=0.125, fy=0.125)
            if tiles_per_dim == 4:
                concat = cv2.resize(concat, (0, 0), fx=0.25, fy=0.25)
            if tiles_per_dim == 5:
                concat = cv2.resize(concat, (0,0), fx=0.3125, fy=0.3125)
            concat = np.reshape(concat, (concat.shape[0], concat.shape[1], 1))
        concats += [concat]
        indices += [(ind1,ind2)]
    concats = np.array(concats)

    preds = clf.predict(concats)[:,1]
    probs = np.zeros((t,t))
    for i in range(len(indices)):
        probs[indices[i]] = preds[i]
    return probs


def make_rows(images, clf, is_image=True):
    # builds row by row, using the probabilities for concat'ing each pair
    probs = get_pair_probs(images, clf, is_image)
    t = int(sqrt(len(images)))
    rows = []
    pool = list(range(t**2))
    row = [0]   # maybe better start from random
    pool.remove(0)
    # row_imgs = []
    while len(pool)>0:
        left_nominees = [(nominee, probs[nominee, row[0]]) for nominee in pool]
        right_nominees = [(nominee, probs[row[-1], nominee]) for nominee in pool]
        best_left, left_score = max(left_nominees, key=lambda x: x[1])
        best_right, right_score = max(right_nominees, key=lambda x: x[1])
        if left_score > right_score:
            probs[:, row[0]] = -1
            row.insert(0, best_left)
            pool.remove(best_left)
            probs[best_left, :] = -1  # set probability ((for best_left to be on the left of something)) to -1, so not to choose again
        else:
            probs[row[-1], :] = -1
            row.append(best_right)
            pool.remove(best_right)
            probs[:, best_right] = -1  # set probability ((for best_right to be on the right of something)) to -1, so not to choose again

        if len(row) == t:  # if we are done building this row
            probs[row[-1], :] = -1  # the rightmost crop can't be chosen again for other row
            probs[:,row[0]] = -1    # same for leftmost
            # row_imgs += [show_row(images, row)]
            # if len(rows) == t-1:
            #     row_imgs = np.concatenate([row_imgs[i] for i in range(t)],axis=0)  # concatenate images in row to make a single image
            #     cv2.imshow('rows', cv2.resize(row_imgs, (0,0), fx=0.35, fy=0.35))
            #     cv2.waitKey(0)
            #     cv2.destroyAllWindows()
            rows += [row]
            if len(pool) > 0:
                row = [pool[0]]  # take an unused crop and start building a new row
                pool.remove(pool[0])

    return rows


def two_rows_score(images, row1, row2, clf, is_image=True):
    if is_image:
        merged_images = [apply_transform(np.rot90(np.concatenate([images[row1[i]], images[row2[i]]], axis=0)))
                         for i in range(len(row1))]
    else:   # is document
        merged_images = [apply_transform(np.concatenate([images[row1[i]], images[row2[i]]], axis=0), t2=len(images), is_image=False)
                         for i in range(len(row1))]
    preds = clf.predict(np.array(merged_images))[:, 1]
    return sum(preds)


def solve_image(images):
    clf = Network()
    rows = make_rows(images, clf)
    pairs = [(i, j) for i in range(len(rows)) for j in range(len(rows)) if i != j]
    row_probs = np.zeros((len(rows), len(rows)))
    for (i,j) in pairs:
        row_probs[(i,j)] = two_rows_score(images, rows[i], rows[j], clf)

    pool = list(range(len(rows)))
    cur_rows = [0]  # maybe better start from random
    pool.remove(0)
    while len(pool) > 0:
        up_nominees = [(nominee, row_probs[nominee,cur_rows[0]]) for nominee in pool]
        down_nominees = [(nominee, row_probs[cur_rows[-1], nominee]) for nominee in pool]
        # up_score = np.max(row_probs[:, cur_rows[0]])
        # down_score = np.max(row_probs[cur_rows[-1], :])
        best_up, up_score = max(up_nominees, key=lambda x: x[1])
        best_down, down_score = max(down_nominees, key=lambda x: x[1])
        if up_score > down_score:
            # best_up = np.argmax(row_probs[:, cur_rows[0]])
            row_probs[:, cur_rows[0]] = -1
            row_probs[best_up, :] = -1
            cur_rows.insert(0, best_up)
            pool.remove(best_up)
        else:
            # best_down = np.argmax(row_probs[cur_rows[-1], :])
            row_probs[cur_rows[-1], :] = -1
            row_probs[:, best_down] = -1
            cur_rows.append(best_down)
            pool.remove(best_down)

        if len(cur_rows) == len(rows):  # if we are done building the whole image
            labels = []
            for row_index in cur_rows:
                labels.extend(rows[row_index])
            # code for showing final built image
            # img = np.concatenate([get_row_image(images,rows[row_ind]) for row_ind in cur_rows], axis=0)
            # cv2.imshow('final', img)
            # cv2.waitKey(5000)
            # cv2.destroyAllWindows()
            ###
            # img = np.concatenate([get_row_image(images, rw) for rw in [list(range(0,4)),list(range(4,8)),list(range(8,12)),list(range(12,16))]], axis=0)
            # cv2.imshow('original', img)
            # cv2.waitKey(5000)

    actual_labels = [x[0] for x in sorted(enumerate(labels), key=lambda a: a[1])]  # to match the input: for image[i] the label is labels[i]
    # print('predicted labels=', actual_labels)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return actual_labels


def solve_doc(images):
    # clf = Network_doc4x4(input_shape=[(images[0].shape[0])//2, images[0].shape[1], 1])
    clf = Network_doc4x4(input_shape=[138, 212, 1])  # for t=4 and t=5 (side) models
    t = int(sqrt(len(images)))
    if t == 4:
        clf.model.load_weights('weights_doc_4x4_side_97.h5')
    elif t == 5:
        clf.model.load_weights('weights_doc_5x5_side_97.7.h5')

    rows = make_rows(images, clf, is_image=False)
    # return []
    # ##############  Sorting the already built rows
    clf = Network_doc4x4(input_shape=[275, 106, 1])  # for t=4 and t=5 (up-down) models
    if t == 4:
        clf.model.load_weights('weights_doc_4x4_updown_88.h5')
    elif t == 5:
        clf.model.load_weights('weights_doc_5x5_updown_85.h5')

    pairs = [(i, j) for i in range(len(rows)) for j in range(len(rows)) if i != j]
    row_probs = np.zeros((len(rows), len(rows)))
    for (i,j) in pairs:
        row_probs[(i,j)] = two_rows_score(images, rows[i], rows[j], clf, is_image=False)

    pool = list(range(len(rows)))
    cur_rows = [0]  # maybe better start from random
    pool.remove(0)
    while len(pool) > 0:
        up_nominees = [(nominee, row_probs[nominee,cur_rows[0]]) for nominee in pool]
        down_nominees = [(nominee, row_probs[cur_rows[-1], nominee]) for nominee in pool]
        # up_score = np.max(row_probs[:, cur_rows[0]])
        # down_score = np.max(row_probs[cur_rows[-1], :])
        best_up, up_score = max(up_nominees, key=lambda x: x[1])
        best_down, down_score = max(down_nominees, key=lambda x: x[1])
        if up_score > down_score:
            # best_up = np.argmax(row_probs[:, cur_rows[0]])
            row_probs[:, cur_rows[0]] = -1
            row_probs[best_up, :] = -1
            cur_rows.insert(0, best_up)
            pool.remove(best_up)
        else:
            # best_down = np.argmax(row_probs[cur_rows[-1], :])
            row_probs[cur_rows[-1], :] = -1
            row_probs[:, best_down] = -1
            cur_rows.append(best_down)
            pool.remove(best_down)

        if len(cur_rows) == len(rows):  # if we are done building the whole image
            labels = []
            for row_index in cur_rows:
                labels.extend(rows[row_index])
            # code for showing the built document:
            # img = np.concatenate([get_row_image(images,rows[row_ind]) for row_ind in cur_rows], axis=0)
            # cv2.imshow('final', cv2.resize(img, (0,0), fx=0.35, fy=0.35))
            # cv2.waitKey(5000)
            # cv2.destroyAllWindows()

            ###
            # img = np.concatenate([get_row_image(images, rw) for rw in [list(range(0,4)),list(range(4,8)),list(range(8,12)),list(range(12,16))]], axis=0)
            # cv2.imshow('original', img)
            # cv2.waitKey(5000)

    actual_labels = [x[0] for x in sorted(enumerate(labels), key=lambda a: a[1])]  # to match the input: for image[i] the label is labels[i]
    # print('predicted labels=', actual_labels)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return actual_labels
