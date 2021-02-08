#!/usr/bin/env python
# Downloads the CelebA face dataset
import os
import numpy as np
import json
from subprocess import check_output
from scipy import io as sio
from tqdm import tqdm
from PIL import Image

import sys
import pickle
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

DOWNLOAD_URL_TRAIN = 'http://ufldl.stanford.edu/housenumbers/train.tar.gz'
DOWNLOAD_URL_TEST = 'http://ufldl.stanford.edu/housenumbers/test.tar.gz'

DOWNLOAD_URL_TRAIN_MAT = 'http://ufldl.stanford.edu/housenumbers/train_32x32.mat'
DOWNLOAD_URL_TEST_MAT = 'http://ufldl.stanford.edu/housenumbers/test_32x32.mat'

DATA_DIR = './data'
DATASET_NAME = 'svhn'
DATASET_PATH = os.path.join(DATA_DIR, DATASET_NAME)


def from_mat(mat, img_dir, fold):
    data = mat['X']
    labels = mat['y']
    examples = []
    for i in tqdm(range(len(labels))):
        img = data[:, :, :, i]
        label = labels[i][0]
        if label == 10:
            label = 0
        filename = os.path.join(img_dir, "{}_{:06d}.jpg".format(fold, i))
        img = Image.fromarray(img)
        img.save(filename)
        examples.append({
            'fold': fold,
            'label': label,
            'filename': filename,
            'data': img
        })
    return examples


def main():
    print("{} dataset download script initializing...".format(DATASET_NAME))
    mkdir(DATA_DIR)
    mkdir(DATASET_PATH)
    history_path = os.getcwd()
    os.chdir(DATASET_PATH)

    print("Downloading {} dataset files...".format(DATASET_NAME))

    # download('train.tar.gz', DOWNLOAD_URL_TRAIN)
    # download('test.tar.gz', DOWNLOAD_URL_TEST)
    download('train_32x32.mat', DOWNLOAD_URL_TRAIN_MAT)
    download('test_32x32.mat', DOWNLOAD_URL_TEST_MAT)

    # os.chdir(DATASET_PATH + '/train')
    train_mat = sio.loadmat('train_32x32.mat')
    test_mat = sio.loadmat('test_32x32.mat')

    test_data = test_mat['X']
    test_labels = test_mat['y']
    os.chdir(history_path)
    IMG_DIR = os.path.join(DATASET_PATH, 'images')
    if not os.path.exists(IMG_DIR):
        os.mkdir(IMG_DIR)
    examples = from_mat(train_mat, IMG_DIR, 'train') + from_mat(test_mat, IMG_DIR, 'test')

    # Generate CSV file for the full dataset
    save_examples(examples)
    print("Successfully built dataset {}".format(DATASET_PATH))


def mkdir(path):
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        print('Creating directory {}'.format(path))
        os.mkdir(path)


def listdir(path):
    filenames = os.listdir(os.path.expanduser(path))
    filenames = sorted(filenames)
    return [os.path.join(path, fn) for fn in filenames]


def download(filename, url):
    if os.path.exists(filename):
        print("File {} already exists, skipping".format(filename))
    else:
        # TODO: security lol
        os.system('wget -nc {} -O {}'.format(url, filename))
        if filename.endswith('.tgz') or filename.endswith('.tar.gz'):
            os.system('ls *gz | xargs -n 1 tar xzvf')
        elif filename.endswith('.zip'):
            os.system('unzip *.zip')


def save_examples(examples):
    print("Writing {} items to {}".format(len(examples), DATA_DIR))

    #save_image_dataset('{}/svhn.dataset'.format(DATA_DIR), examples)
    #save_image_dataset('{}/svhn-04.dataset'.format(DATA_DIR), [e for e in examples if int(e['label']) < 5])
    #save_image_dataset('{}/svhn-59.dataset'.format(DATA_DIR), [e for e in examples if int(e['label']) >= 5])

    splits = [
        [3, 6, 7, 8],
        [1, 2, 4, 6],
        [2, 3, 4, 9],
        [0, 1, 2, 6],
        [4, 5, 6, 9],
    ]

    for idx, split in enumerate(splits):
        if not os.path.exists(os.path.join(DATA_DIR, DATASET_NAME, 'split' + str(idx))):
            os.mkdir(os.path.join(DATA_DIR, DATASET_NAME, 'split' + str(idx)))
        unknown_classes = [i for i in split]
        train_examples = {}
        train_examples['labels'] = []
        train_examples['data'] = []
        train_examples['filenames'] = []

        test_examples = {}
        test_examples['labels'] = []
        test_examples['data'] = []
        test_examples['filenames'] = []
        open_test_examples = {}
        open_test_examples['labels'] = []
        open_test_examples['data'] = []
        open_test_examples['filenames'] = []
        for i, e in enumerate(examples):
            if e['fold'] == 'train' and (e['label'] not in unknown_classes):
                train_examples['filenames'] += [e['filename'].encode()]
                train_examples['data'] += [e['data']]
                train_examples['labels'] += [e['label']]
            elif e['fold'] == 'test':
                if e['label'] not in unknown_classes:
                    test_examples['filenames'] += [e['filename'].encode()]
                    test_examples['data'] += [e['data']]
                    test_examples['labels'] += [e['label']]
                else:
                    open_test_examples['filenames'] += [e['filename'].encode()]
                    open_test_examples['data'] += [e['data']]
                    open_test_examples['labels'] += [e['label']]
        open_class_to_idx = {}
        open_idx_to_class = {}

        for fake_index, _idx in enumerate(split):
            open_idx_to_class[fake_index + 6] = _idx
            open_class_to_idx[_idx] = fake_index + 6

        class_to_idx = {}
        idx_to_class = {}

        meta_dict = {'num_cases_per_batch': 10000, 'num_vis': 3072}
        meta_dict['label_names'] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        fake_index = 0
        for _idx in range(10):
            if _idx not in split:
                idx_to_class[fake_index] = _idx
                class_to_idx[_idx] = fake_index
                fake_index += 1
        train_obj_ = '{}/split{}/train_obj.pkl'.format(DATA_DIR, idx)
        test_obj_ = '{}/split{}/test_obj.pkl'.format(DATA_DIR, idx)
        open_test_obj_ = '{}/split{}/open_test_obj.pkl'.format(DATA_DIR, idx)
        idx_to_class_ = '{}/split{}/idx_to_class.pkl'.format(DATA_DIR, idx)
        class_to_idx_ = '{}/split{}/class_to_idx.pkl'.format(DATA_DIR, idx)
        meta_ = '{}/split{}/meta.pkl'.format(DATA_DIR, idx)
        open_class_to_idx_ = '{}/split{}/open_class_to_idx.pkl'.format(DATA_DIR, idx)
        open_idx_to_class_ = '{}/split{}/open_idx_to_class.pkl'.format(DATA_DIR, idx)

        save_svhn_dataset(train_examples, train_obj_)
        save_svhn_dataset(test_examples, test_obj_)
        save_svhn_dataset(open_test_examples, open_test_obj_)
        save_svhn_dataset(idx_to_class, idx_to_class_)
        save_svhn_dataset(class_to_idx, class_to_idx_)

        save_svhn_dataset(meta_dict, meta_)
        save_svhn_dataset(open_class_to_idx, open_class_to_idx_)
        save_svhn_dataset(open_idx_to_class, open_idx_to_class_)

def save_image_dataset(filename, examples):
    with open(filename, 'w') as fp:
        for ex in examples:
            fp.write(json.dumps(ex))
            fp.write('\n')

def save_svhn_dataset(examples, output_filename):

    with open(output_filename, 'wb') as fp:
        pickle.dump(examples, fp)
    print("Wrote {} items to {}".format(len(examples), output_filename))

if __name__ == '__main__':
    main()
