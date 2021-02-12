#!/usr/bin/env python
import numpy as np
import hashlib
import sys
import requests
import os
import json
from PIL import Image
from tqdm import tqdm
import pickle
DATA_DIR = './data'
DOWNLOAD_URL = 'https://s3.amazonaws.com/img-datasets/mnist.npz'
LATEST_MD5 = ''
DATASET_NAME = 'mnist'
generated_DATA_DIR = DATA_DIR + '/' + DATASET_NAME

def save_set(fold, x, y, suffix='png'):
    examples = []
    print("Writing MNIST dataset {}".format(fold))
    for i in tqdm(range(len(x))):
        label = y[i]
        img_filename = os.path.join(generated_DATA_DIR, '{}/{:05d}_{:d}.{}'.format(fold, i, label, suffix))
        img = Image.fromarray(x[i])
        if not os.path.exists(img_filename):
            img.save(os.path.expanduser(img_filename))
        entry = {
            'filename': img_filename,
            'label': label,
            'fold': fold,
            'data': img
        }
        examples.append(entry)
    return examples


def download_mnist_data(path):
    file_path = os.path.join(path, 'mnist.npz')
    if not os.path.exists(file_path):
        response = requests.get(DOWNLOAD_URL)
        open(path, 'wb').write(response.content)
    with np.load(file_path) as f:
        x_test, y_test = f['x_test'], f['y_test']
        VAL_SIZE = 5000
        # Use last 5k examples as a validation set
        x_train, y_train = f['x_train'][:-VAL_SIZE], f['y_train'][:-VAL_SIZE]
        x_val, y_val = f['x_train'][-VAL_SIZE:], f['y_train'][-VAL_SIZE:]
    return (x_train, y_train), (x_test, y_test), (x_val, y_val)


def mkdir(dirname):
    import errno
    try:
        os.makedirs(dirname)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
        if not os.path.isdir(dirname):
            raise


def is_latest_version(latest_md5):
    dataset_file = os.path.join(DATA_DIR, 'mnist.dataset')
    if not os.path.exists(dataset_file):
        return False
    data = open(dataset_file, 'rb').read()
    current_md5 = hashlib.md5(data).hexdigest()
    if current_md5 == latest_md5:
        print("Have latest version of MNIST: {}".format(current_md5))
        return True
    else:
        print("Have old version {} of MNIST, downloading version {}".format(current_md5, latest_md5))
        return False

def download_mnist(latest_md5):

    if is_latest_version(latest_md5):
        print("Already have the latest version of mnist.dataset, not downloading")
        return
    (train_x, train_y), (test_x, test_y), (val_x, val_y) = download_mnist_data(generated_DATA_DIR)

    train = save_set('train', train_x, train_y)
    test = save_set('test', test_x, test_y)
    val = save_set('val', val_x, val_y)
    for example in train:
        example['fold'] = 'train'
    for example in test:
        example['fold'] = 'test'
    for example in val:
        example['fold'] = 'validation'

    # Splits to match the CIFAR and SVHN experiments
    splits = [
        [3, 6, 7, 8],
        [1, 2, 4, 6],
        [2, 3, 4, 9],
        [0, 1, 2, 6],
        [4, 5, 6, 9],
    ]
    examples = train + test + val
    for idx, split in enumerate(splits):
        if not os.path.exists(generated_DATA_DIR + '/split' + str(idx)):
            os.mkdir(generated_DATA_DIR + '/split' + str(idx))
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

        meta_dict = {'image_size': 32, 'image_channels': 1}
        meta_dict['label_names'] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        fake_index = 0
        for _idx in range(10):
            if _idx not in split:
                idx_to_class[fake_index] = _idx
                class_to_idx[_idx] = fake_index
                fake_index += 1
        train_obj_ = '{}/split{}/train_obj.pkl'.format(generated_DATA_DIR, idx)
        test_obj_ = '{}/split{}/test_obj.pkl'.format(generated_DATA_DIR, idx)
        open_test_obj_ = '{}/split{}/open_test_obj.pkl'.format(generated_DATA_DIR, idx)
        idx_to_class_ = '{}/split{}/idx_to_class.pkl'.format(generated_DATA_DIR, idx)
        class_to_idx_ = '{}/split{}/class_to_idx.pkl'.format(generated_DATA_DIR, idx)
        meta_ = '{}/split{}/meta.pkl'.format(generated_DATA_DIR, idx)
        open_meta_ = '{}/split{}/open_meta.pkl'.format(generated_DATA_DIR, idx)
        open_class_to_idx_ = '{}/split{}/open_class_to_idx.pkl'.format(generated_DATA_DIR, idx)
        open_idx_to_class_ = '{}/split{}/open_idx_to_class.pkl'.format(generated_DATA_DIR, idx)

        save_dataset(train_examples, train_obj_)
        save_dataset(test_examples, test_obj_)
        save_dataset(open_test_examples, open_test_obj_)
        save_dataset(idx_to_class, idx_to_class_)
        save_dataset(class_to_idx, class_to_idx_)

        save_dataset(meta_dict, meta_)
        save_dataset(meta_dict, open_meta_)
        save_dataset(open_class_to_idx, open_class_to_idx_)
        save_dataset(open_idx_to_class, open_idx_to_class_)

def save_dataset(examples, output_filename):

    with open(output_filename, 'wb') as fp:
        pickle.dump(examples, fp)
    print("Wrote {} items to {}".format(len(examples), output_filename))

def save_image_dataset(filename, examples):
    with open(filename, 'w') as fp:
        for ex in examples:
            fp.write(json.dumps(ex))
            fp.write('\n')


if __name__ == '__main__':
    mkdir(generated_DATA_DIR)
    mkdir(os.path.join(generated_DATA_DIR, 'train'))
    mkdir(os.path.join(generated_DATA_DIR, 'test'))
    mkdir(os.path.join(generated_DATA_DIR, 'val'))
    download_mnist(os.path.join(generated_DATA_DIR, LATEST_MD5))
