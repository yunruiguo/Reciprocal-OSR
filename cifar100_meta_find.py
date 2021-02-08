import json
import os
files = os.listdir('./cifar100')


def sortedDictValues(adict):
    items = adict.items()
    items = sorted(items)
    return [value for key, value in items]

animals50_splits = {}
for file in files:
    if 'cifar100-animals-50' in file:
        print(file)
        animals = set()
        test_objects = open(os.path.join('./cifar100', file)).read()
        lines = test_objects.strip().splitlines()
        for l in lines:
            example = json.loads(l)
            animals.add(example['label'])
        idx = file[-9]
        animals50_splits[idx] = list(animals)
animals50_splits = sortedDictValues(animals50_splits)

not_animals10_splits = {}
for file in files:
    if 'cifar100-not-animals-10' in file:
        print(file)
        not_animals = set()
        test_objects = open(os.path.join('./cifar100', file)).read()
        lines = test_objects.strip().splitlines()
        for l in lines:
            example = json.loads(l)
            not_animals.add(example['label'])
        idx = file[-9]
        not_animals10_splits[idx] = list(not_animals)
not_animals10_splits = sortedDictValues(not_animals10_splits)

animals10_splits = {}

for file in files:
    if 'cifar100-animals-10' in file:
        print(file)
        animals = set()
        test_objects = open(os.path.join('./cifar100', file)).read()
        lines = test_objects.strip().splitlines()
        for l in lines:
            example = json.loads(l)
            animals.add(example['label'])
        idx = file[-9]
        animals10_splits[idx] = list(animals)
animals10_splits = sortedDictValues(animals10_splits)



not_animals50_splits = {}

for file in files:
    if 'cifar100-not-animals-50' in file:
        print(file)
        not_animals = set()
        test_objects = open(os.path.join('./cifar100', file)).read()
        lines = test_objects.strip().splitlines()
        for l in lines:
            example = json.loads(l)
            not_animals.add(example['label'])
        idx = file[-9]
        not_animals50_splits[idx] = list(not_animals)
not_animals50_splits = sortedDictValues(not_animals50_splits)
print('finish!')