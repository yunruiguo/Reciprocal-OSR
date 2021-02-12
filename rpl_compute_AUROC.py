import argparse
import os
import shutil

import torch
import pickle

import torchvision.models as models
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from datasets.cifar_dataset import CIFARDataset
from datasets.dataset import StandardDataset
from datasets.open_dataset import OpenDataset
from models.backbone import encoder32
from models.backbone_wide_resnet import wide_encoder
from evaluate import collect_rpl_max, seenval_baseline_thresh, unseenval_baseline_thresh

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
                        help="which gpu will be used", default='1')
    parser.add_argument("--backbone", type=str,
                        help="architecture of backbone", default="wide_resnet")
    parser.add_argument("--dataset", type=str,
                        help="mnist, svhn, cifar10, cifar10plus, cifar50plus, tiny_imagenet", default="tiny_imagenet")
    parser.add_argument("--split", type=str,
                        help="Split of dataset, split0, split1...", default="split3")

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
                                    transforms.Resize((45, 45)),
                                    transforms.CenterCrop((32, 32)),
                                    transforms.ToTensor(),
                                ]))
    unseen_dataset = CIFARDataset(open_test_obj, open_meta_dict, open_class_to_idx,
                                  transforms.Compose([
                                      transforms.Resize((45, 45)),
                                      transforms.CenterCrop((32, 32)),
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

    seen_confidence_dict = collect_rpl_max(model, seen_loader, args.gamma, cifar=True,
                                           idx_to_class=idx_to_class)
    unseen_confidence_dict = collect_rpl_max(model, unseen_loader, args.gamma, cifar=True,
                                             idx_to_class=open_idx_to_class)

    # Computing AUC
    from sklearn.metrics import roc_auc_score

    preds = []
    labels = []
    dist = []
    for known_class_str, samples in seen_confidence_dict.items():
        for s in samples:
            labels += [1]
            preds += [s['prob']]
            dist += [s['dist']]

    for known_class_str, samples in unseen_confidence_dict.items():
        for s in samples:
            labels += [0]
            preds += [s['prob']]
            dist += [s['dist']]

    auc_prob = roc_auc_score(labels, preds)
    auc_dist = roc_auc_score(labels, dist)
    print('AUC on prob', auc_prob)
    print('AUC on dist', auc_dist)

"""
    print("Dist-Auroc score: " + str(dist_auroc_score))
    print("Prob-Auroc score: " + str(prob_auroc_score))

    metrics = summarize(seen_info, unseen_info, thresh, verbose=False)
    metrics['dist_auroc_lwnealstyle'] = dist_auroc_score
    metrics['prob_auroc_lwnealstyle'] = prob_auroc_score
    metrics['dist_OSR_CSR_AUC'] = dist_metrics['OSR_CSR_AUC']

    print("prob-AUC score: " + str(metrics['OSR_CSR_AUC']))
    print("dist-AUC score: " + str(metrics['dist_OSR_CSR_AUC']))

    with open(metrics_folder + 'metrics.pkl', 'wb') as f:
        pickle.dump(metrics, f)
"""