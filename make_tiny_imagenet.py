#!/usr/bin/env python
import os
import random
import numpy as np
import json
from subprocess import check_output
from PIL import Image
import pickle
DATA_ROOT_DIR = './data'
DATASET_DIR = os.path.join(DATA_ROOT_DIR, 'tiny_imagenet')
DATASET_NAME = 'tiny_imagenet'

ANIMAL_CLASSES = [
    "dragonfly, darning needle, devil's darning needle, sewing needle, snake feeder, snake doctor, mosquito hawk, skeeter hawk",
    'African elephant, Loxodonta africana',
    'American alligator, Alligator mississipiensis',
    'American lobster, Northern lobster, Maine lobster, Homarus americanus',
    'Arabian camel, dromedary, Camelus dromedarius',
    'Chihuahua',
    'Egyptian cat',
    'European fire salamander, Salamandra salamandra',
    'German shepherd, German shepherd dog, German police dog, alsatian',
    'Labrador retriever',
    'Persian cat',
    'Yorkshire terrier',
    'albatross, mollymawk',
    'baboon',
    'bee',
    'bighorn, bighorn sheep, cimarron, Rocky Mountain bighorn, Rocky Mountain sheep, Ovis canadensis',
    'bison',
    'black stork, Ciconia nigra',
    'black widow, Latrodectus mactans',
    'boa constrictor, Constrictor constrictor',
    'brown bear, bruin, Ursus arctos',
    'bullfrog, Rana catesbeiana',
    'centipede',
    'chimpanzee, chimp, Pan troglodytes',
    'cockroach, roach',
    'cougar, puma, catamount, mountain lion, painter, panther, Felis concolor',
    'dugong, Dugong dugon',
    'feeder, snake doctor, mosquito hawk, skeeter hawk',
    'fly',
    'gazelle',
    'golden retriever',
    'goldfish, Carassius auratus',
    'goose',
    'grasshopper, hopper',
    'guinea pig, Cavia cobaya',
    'hog, pig, grunter, squealer, Sus scrofa',
    'jellyfish',
    'king penguin, Aptenodytes patagonica',
    'koala, koala bear, kangaroo bear, native bear, Phascolarctos cinereus',
    'ladybug, ladybeetle, lady beetle, ladybird, ladybird beetle',
    'lesser panda, red panda, panda, bear cat, cat bear, Ailurus fulgens',
    'lion, king of beasts, Panthera leo',
    'mantis, mantid',
    'monarch, monarch butterfly, milkweed butterfly, Danaus plexippus',
    'orangutan, orang, orangutang, Pongo pygmaeus',
    'ox',
    'scorpion',
    'sea cucumber, holothurian',
    'sea slug, nudibranch',
    'sheep, Ovis canadensis',
    'slug',
    'snail',
    'spiny lobster, langouste, rock lobster, crawfish, crayfish, sea crawfish',
    'standard poodle',
    'sulphur butterfly, sulfur butterfly',
    'tabby, tabby cat',
    'tailed frog, bell toad, ribbed toad, tailed toad, Ascaphus trui',
    'tarantula',
    'trilobite',
    'walking stick, walkingstick, stick insect',
]



def mkdir(path):
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        print('Creating directory {}'.format(path))
        os.mkdir(path)


def download(DATASET_DIR, url):
    filename = os.path.join(DATASET_DIR)

    mkdir(filename)
    os.system('wget -P {} {}'.format(filename, url))
    if url.endswith('.zip'):
        os.system('unzip -d {} {}.zip'.format(DATASET_DIR, os.path.join(filename, 'tiny-imagenet-200')))


def save_dataset(examples, output_filename):
    print("Writing {} items to {}".format(len(examples), output_filename))
    with open(output_filename, 'wb') as fp:
        pickle.dump(examples, fp)

if __name__ == '__main__':
    print("Downloading dataset {}...".format(DATASET_NAME))
    mkdir(DATA_ROOT_DIR)
    mkdir(DATASET_DIR)


    # Download and extract dataset
    data_path = os.path.join(DATASET_DIR, 'tiny-imagenet-200')
    print("Downloading dataset files...")
    download(DATASET_DIR, 'http://cs231n.stanford.edu/tiny-imagenet-200.zip')

    wnids = open(os.path.join(data_path, 'wnids.txt')).read().splitlines()

    wnid_names = {}
    for line in open(os.path.join(data_path, 'words.txt')).readlines():
        wnid, name = line.strip().split('\t')
        wnid_names[wnid] = name

    test_filenames = os.listdir(os.path.join(data_path, 'test/images'))

    examples = []

    # Collect training examples
    data_path = os.path.join(DATASET_DIR, 'tiny-imagenet-200')
    for wnid in os.listdir(os.path.join(data_path, 'train')):
        filenames = os.listdir(os.path.join(data_path, 'train', wnid, 'images'))
        for filename in filenames:
            file_path = os.path.join(data_path,  'train', wnid, 'images', filename)
            img = Image.open(file_path)
            if img.mode != 'RGB':
                img = img.convert(mode='RGB')
            examples.append({
                'filename': file_path,
                'class': wnid_names[wnid],
                'label': wnid,
                'fold': 'train',
                'data': img
            })

    # Use validation set as a test set
    for line in open(os.path.join(data_path, 'val/val_annotations.txt')).readlines():
        jpg_name, wnid, x0, y0, x1, y1 = line.split()
        file_path = os.path.join(data_path, 'val', 'images', jpg_name)
        img = Image.open(file_path)
        if img.mode != 'RGB':
            img = img.convert(mode='RGB')
        examples.append({
            'filename': file_path,
            'class': wnid_names[wnid],
            'label': wnid,
            'fold': 'test',
            'data': img
        })

    # Split animal and nonanimal (plants, vehicles, objects, etc)
    animal_examples = [e for e in examples if e['class'] in ANIMAL_CLASSES]

    not_animal_examples = [e for e in examples if e['class'] not in ANIMAL_CLASSES]

    # Put the unlabeled test set in a separate dataset
    test_examples = []
    for jpg_name in os.listdir(os.path.join(data_path, 'test/images')):
        test_examples.append({
            'filename': os.path.join(data_path, 'test', 'images', jpg_name),
            'fold': 'test',
        })

    # Select a random 10, 50, 100 classes and partition them out
    meta_classes_list = list(set(e['class'] for e in examples))
    classes = meta_classes_list[:]
    random.seed(42)
    meta_classes_to_idx = {}
    meta_idx_to_classes = {}
    for idx, iter in enumerate(meta_classes_list):
        meta_classes_to_idx[iter] = idx
        meta_idx_to_classes[idx] = iter
    meta_dict = {'image_size': 32, 'image_channels': 3}
    meta_dict['class_names'] = meta_classes_list
    for split in range(5):
        random.shuffle(classes)
        known_idx = []
        known_classes = []
        for iter in classes[:20]:
            known_idx += [meta_classes_to_idx[iter]]
        known_idx.sort()
        for idx in known_idx:
            known_classes += [meta_idx_to_classes[idx]]

        if not os.path.exists(os.path.join(DATASET_DIR, 'split' + str(split))):
            os.mkdir(os.path.join(DATASET_DIR, 'split' + str(split)))

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
            if e['fold'] == 'train' and (e['class'] in known_classes):
                train_examples['filenames'] += [e['filename']]
                train_examples['data'] += [e['data']]
                train_examples['labels'] += [meta_classes_to_idx[e['class']]]
            elif e['fold'] == 'test':
                if e['class'] in known_classes:
                    test_examples['filenames'] += [e['filename']]
                    test_examples['data'] += [e['data']]
                    test_examples['labels'] += [meta_classes_to_idx[e['class']]]
                else:
                    open_test_examples['filenames'] += [e['filename']]
                    open_test_examples['data'] += [e['data']]
                    open_test_examples['labels'] += [meta_classes_to_idx[e['class']]]
        class_to_idx = {}
        idx_to_class = {}

        for new_idx, iter in enumerate(known_classes):
            idx_to_class[new_idx] = iter
            class_to_idx[iter] = new_idx

        open_class_to_idx = {}
        open_idx_to_class = {}

        new_index = 0
        for iter in meta_classes_list:
            if iter not in known_classes:
                open_idx_to_class[new_index] = iter
                open_class_to_idx[iter] = new_index
                new_index += 1
        train_obj_ = os.path.join(DATASET_DIR, 'split{}/train_obj.pkl'.format(split))
        test_obj_ = os.path.join(DATASET_DIR, 'split{}/test_obj.pkl'.format(split))
        open_test_obj_ = os.path.join(DATASET_DIR, 'split{}/open_test_obj.pkl'.format(split))
        idx_to_class_ = os.path.join(DATASET_DIR, 'split{}/idx_to_class.pkl'.format(split))
        class_to_idx_ = os.path.join(DATASET_DIR, 'split{}/class_to_idx.pkl'.format(split))

        meta_ = os.path.join(DATASET_DIR, 'split{}/meta.pkl'.format(split))
        open_class_to_idx_ = os.path.join(DATASET_DIR, 'split{}/open_class_to_idx.pkl'.format(split))
        open_idx_to_class_ = os.path.join(DATASET_DIR, 'split{}/open_idx_to_class.pkl'.format(split))

        save_dataset(train_examples, train_obj_)
        save_dataset(test_examples, test_obj_)
        save_dataset(open_test_examples, open_test_obj_)
        save_dataset(idx_to_class, idx_to_class_)
        save_dataset(class_to_idx, class_to_idx_)

        save_dataset(meta_dict, meta_)
        save_dataset(open_class_to_idx, open_class_to_idx_)
        save_dataset(open_idx_to_class, open_idx_to_class_)

    print("Finished building dataset {}".format(DATASET_NAME))