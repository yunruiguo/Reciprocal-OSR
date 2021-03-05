import argparse
import os
import shutil
from sklearn import manifold
import torch
import pickle

import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import roc_curve, auc, roc_auc_score, f1_score
from datasets.cifar_dataset import CIFARDataset
import matplotlib.pyplot as plt
from models.backbone import encoder32
from models.backbone_wide_resnet import wide_encoder
from evaluate import collect_rpl_max

def plot_AUROC(Y, P, pos_label=1):
    fpr, tpr, _ = roc_curve(Y, P, pos_label=1)
    roc_auc = auc(fpr, tpr)
    print("AUC: ", roc_auc)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

def plot_TSNE(X, y):
    '''t-SNE'''
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    X_tsne = tsne.fit_transform(X)

    print("Org data dimension is {}. Embedded data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))

    '''嵌入空间可视化'''
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
    plt.figure(figsize=(8, 8))
    for i in range(X_norm.shape[0]):
        plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=plt.cm.Set1(y[i]),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.show()
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--gap", type=str,
                        help="TRUE iff use global average pooling layer. Otherwise, use linear layer.", default="TRUE")
    parser.add_argument("--desired_features", type=str,
                        help="None means no features desired. Other examples include last, 2_to_last.", default="None")
    parser.add_argument("--latent_size", type=int,
                        help="Dimension of embeddings.", default=256)
    parser.add_argument("--num_rp_per_cls", type=int,
                        help="Number of reciprocal points per class.", default=1)
    parser.add_argument("--gamma", type=float,
                        help="", default=0.5)
    parser.add_argument("--gpu_id", type=str,
                        help="which gpu will be used", default=0)
    parser.add_argument("--backbone", type=str,
                        help="architecture of backbone", default="wide_resnet")
    parser.add_argument("--dataset", type=str,
                        help="mnist, svhn, cifar10, cifar10plus, cifar50plus, tiny_imagenet", default="tiny_imagenet")
    parser.add_argument("--split", type=str,
                        help="Split of dataset, split0, split1...", default="split0")

    parser.add_argument("--dataset_folder", type=str,
                        help="name of folder where dataset lives.",
                        default="./data")

    parser.add_argument("--checkpoint_folder_path", type=str,
                        help="full file path to folder where the baseline is located",
                        default="./ckpt/")

    args = parser.parse_args()
    dataset_folder_path = os.path.join(args.dataset_folder, args.dataset, args.split)
    with open(dataset_folder_path + '/class_to_idx.pkl', 'rb') as f:
        class_to_idx = pickle.load(f)
    with open(dataset_folder_path + '/idx_to_class.pkl', 'rb') as f:
        idx_to_class = pickle.load(f)
    with open(dataset_folder_path + '/open_class_to_idx.pkl', 'rb') as f:
        open_class_to_idx = pickle.load(f)
    with open(dataset_folder_path + '/open_idx_to_class.pkl', 'rb') as f:
        open_idx_to_class = pickle.load(f)
    with open(dataset_folder_path + '/test_obj.pkl', 'rb') as fo:
        test_obj = pickle.load(fo)
    with open(dataset_folder_path + '/open_test_obj.pkl', 'rb') as fo:
        open_test_obj = pickle.load(fo)
    with open(dataset_folder_path + '/meta.pkl', 'rb') as fo:
        meta_dict = pickle.load(fo)
    with open(dataset_folder_path + '/open_meta.pkl', 'rb') as fo:
        open_meta_dict = pickle.load(fo)

    if args.dataset in ['mnist', 'svhn', 'cifar10']:
        known_num_classes = 6
    elif args.dataset in ['cifar10plus', 'cifar50plus']:
        known_num_classes = 4
    elif args.dataset in ['tiny_imagenet']:
        known_num_classes = 20
    else:
        raise ValueError('Wrong dataset.')

    seen_dataset = CIFARDataset(test_obj, meta_dict, class_to_idx,
                                transforms.Compose([
                                    transforms.Resize((32, 32)),
                                    transforms.ToTensor(),
                                ]))
    unseen_dataset = CIFARDataset(open_test_obj, open_meta_dict, open_class_to_idx,
                                  transforms.Compose([
                                      transforms.Resize((32, 32)),
                                      transforms.ToTensor(),
                                  ]))

    seen_loader = DataLoader(seen_dataset, batch_size=64, shuffle=False, num_workers=3)
    unseen_loader = DataLoader(unseen_dataset, batch_size=64, shuffle=False, num_workers=3)

    criterion = torch.nn.CrossEntropyLoss(reduction='none')

    if args.backbone == 'OSCRI_encoder':
        model = encoder32(meta_dict['image_size'], meta_dict['image_channels'], args.latent_size, num_classes=known_num_classes, num_rp_per_cls=args.num_rp_per_cls,
                          gap=args.gap == 'TRUE')
    elif args.backbone == 'wide_resnet':
        model = wide_encoder(meta_dict['image_size'], meta_dict['image_channels'], args.latent_size, 40, 4, 0, num_classes=known_num_classes, num_rp_per_cls=args.num_rp_per_cls)
    else:
        raise ValueError(args.backbone_type + ' is not supported.')
    experiment_ckpt_path = args.checkpoint_folder_path + args.dataset + '_' + args.split + '_' + args.backbone
    model.load_state_dict(torch.load(experiment_ckpt_path + '/best_model.pt'))
    model.cuda()
    model.eval()
    known_tsne_fea = {}
    unknown_tsne_fea = {}
    for i in range(known_num_classes):
        known_tsne_fea[i] = []
    for i in open_class_to_idx:
        unknown_tsne_fea[open_class_to_idx[i]] = []
    seen_confidence_dict = collect_rpl_max(model, seen_loader, args.gamma, len(seen_dataset), known_tsne_fea,
                                           idx_to_class=idx_to_class)
    unseen_confidence_dict = collect_rpl_max(model, unseen_loader, args.gamma, len(unseen_dataset), unknown_tsne_fea,
                                             idx_to_class=open_idx_to_class)
    tsne_fea = {**known_tsne_fea, **unknown_tsne_fea}
    # Computing AUC
    metric = 'prob'
    thres = 0.99
    preds = []
    labels = []
    pre_labels = []
    true_labels = []
    dist = []
    for _, samples in seen_confidence_dict.items():
        for s in samples:
            labels += [1]
            true_labels += [s['label']]
            preds += [s['prob']]
            dist += [s['dist']]
            if s[metric] > thres:
                pre_labels += [s['prediction']]
            else:
                pre_labels += [known_num_classes]

    for _, samples in unseen_confidence_dict.items():
        for s in samples:
            labels += [0]
            preds += [s['prob']]
            dist += [s['dist']]

            if s[metric] > thres:
                pre_labels += [s['prediction']]
            else:
                pre_labels += [known_num_classes]
            
            true_labels += [known_num_classes]

    auc_prob = roc_auc_score(labels, preds)
    auc_dist = roc_auc_score(labels, dist)
    print('AUC on prob', auc_prob)
    print('AUC on dist', auc_dist)
    X = []
    Y = []
    for key in known_tsne_fea.keys():
        X.extend(known_tsne_fea[key])
        Y.extend([key]*len(known_tsne_fea[key]))
    for key in unknown_tsne_fea.keys():
        X.extend(unknown_tsne_fea[key])
        Y.extend([known_num_classes]*len(unknown_tsne_fea[key]))
    dims = (len(X), len(X[0]))
    f1_values = f1_score(true_labels, pre_labels, average=None)
    print('F1-Scores:', f1_values)
    X, Y = np.reshape(X, dims), np.asarray(Y)
    plot_TSNE(X, Y)
    plot_AUROC(labels, preds)
    plot_AUROC(labels, dist)
