import argparse
import logging
from tqdm import tqdm
import os, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch import optim
import random
import pickle
from sklearn.preprocessing import normalize
from PIL import Image

from datasets.cifar_dataset import CIFARDataset
from datasets.dataset import StandardDataset
from evaluate import evaluate_val
from models.backbone import encoder32
from models.backbone_resnet import encoder
from models.backbone_wide_resnet import wide_encoder
from penalties import compute_rpl_loss
from utils import count_parameters, setup_logger


    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--n_epochs", type=int,
                        help="number of epochs to train", default=90)
    parser.add_argument("--gpu_id", type=int,
                        help="which gpu will be used", default=1)
    parser.add_argument("--gap", type=str,
                        help="TRUE iff use global average pooling layer. Otherwise, use linear layer.", default="TRUE")
    parser.add_argument("--lr_scheduler", type=str,
                        help="patience, step.", default="step")
    parser.add_argument("--dataset", type=str,
                        help="mnist, svhn, cifar10, cifar10plus, cifar50plus, tiny_imagenet", default="tiny_imagenet")
    parser.add_argument("--split", type=str,
                        help="Split of dataset, split0, split1...", default="split2")
    parser.add_argument("--latent_size", type=int,
                        help="Dimension of embeddings.", default=256)
    parser.add_argument("--num_rp_per_cls", type=int,
                        help="Number of reciprocal points per class.", default=1)
    parser.add_argument("--lamb", type=float,
                        help="how much to weight the open-set regularization term in objective.", default=0.1)
    parser.add_argument("--gamma", type=float,
                        help="how much to weight the probability assignment.", default=0.5)
    parser.add_argument("--divide", type=str,
                        help="TRUE or FALSE, as to whether or not to divide loss by latent_size for convergence.",
                        default="TRUE")
    parser.add_argument("--dataset_folder", type=str,
                        help="name of folder where dataset details live.", default="./data")
    parser.add_argument("--batch_size", type=int,
                        help="size of a batch during training", default=64)
    parser.add_argument("--lr", type=float,
                        help="initial learning rate during training", default=0.01)
    parser.add_argument("--patience", type=int,
                        help="patience of lr scheduler", default=30)
    parser.add_argument("--img_size", type=int,
                        help="desired square image size.", default=32)
    parser.add_argument("--num_workers", type=int,
                        help="number of workers during training", default=4)
    parser.add_argument("--backbone", type=str,
                        help="architecture of backbone", default="wide_resnet")
    parser.add_argument("--checkpoint_folder_path", type=str,
                        help="./ckpt", default="./ckpt/")
    parser.add_argument("--load_history_model", type=bool,
                        help="True or False", default=True)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    CKPT_BASE_NAME = args.backbone
    LOGFILE_NAME = CKPT_BASE_NAME + '_logfile'
    experiment_ckpt_path = args.checkpoint_folder_path + args.dataset + '_' + args.split + '_' + CKPT_BASE_NAME
    if not os.path.exists(args.checkpoint_folder_path):
        os.mkdir(args.checkpoint_folder_path)
    if not os.path.exists(experiment_ckpt_path):
        os.mkdir(experiment_ckpt_path)
        os.mkdir(experiment_ckpt_path + '/' + 'backups')

    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    logger = setup_logger('logger', formatter, LOGFILE_NAME)
    
    dataset_folder_path = os.path.join(args.dataset_folder, args.dataset, args.split)
    with open(dataset_folder_path + '/train_obj.pkl', 'rb') as f:
        train_obj = pickle.load(f)
    with open(dataset_folder_path + '/test_obj.pkl', 'rb') as f:
        test_obj = pickle.load(f)
    with open(dataset_folder_path + '/meta.pkl', 'rb') as f:
        meta_dict = pickle.load(f)
    with open(dataset_folder_path + '/class_to_idx.pkl', 'rb') as f:
        class_to_idx = pickle.load(f)
    with open(dataset_folder_path + '/idx_to_class.pkl', 'rb') as f:
        idx_to_class = pickle.load(f)

    if args.dataset in ['mnist', 'svhn', 'cifar10']:
        known_num_classes = 6
        if args.dataset == 'mnist':
            train_trans = transforms.Compose([transforms.Resize((32, 32)),
                                              transforms.ToTensor()])
            val_trans = transforms.Compose([transforms.Resize((32, 32)),
                                            transforms.ToTensor()])
        else:
            train_trans = transforms.Compose([transforms.RandomResizedCrop((32, 32)),
                                              transforms.RandomHorizontalFlip(0.5),
                                              transforms.ToTensor()])
            val_trans = transforms.Compose([transforms.ToTensor()])
    elif args.dataset in ['cifar10plus', 'cifar50plus']:
        known_num_classes = 4
        train_trans = transforms.Compose([transforms.RandomResizedCrop((32, 32)),
                                          transforms.RandomHorizontalFlip(0.5),
                                          transforms.ToTensor()])
        val_trans = transforms.Compose([transforms.ToTensor()])
    elif args.dataset == 'tiny_imagenet':
        known_num_classes = 20
        train_trans = transforms.Compose([transforms.Resize((45, 45)),
                                          transforms.RandomHorizontalFlip(0.5),
                                          transforms.RandomCrop((32, 32)),
                                          transforms.ToTensor()])
        val_trans = transforms.Compose([transforms.Resize((45, 45)),
                                        transforms.CenterCrop((32, 32)),
                                        transforms.ToTensor()])
    else:
        raise ValueError('Wrong dataset.')


    logging.info("Number of seen classes: " + str(known_num_classes))
    dataset = CIFARDataset(train_obj, meta_dict, class_to_idx, train_trans)
    val_dataset = CIFARDataset(test_obj, meta_dict, class_to_idx, val_trans)

    if args.backbone == 'OSCRI_encoder':
        model = encoder32(meta_dict['image_size'], meta_dict['image_channels'], args.latent_size, num_classes=known_num_classes, num_rp_per_cls=args.num_rp_per_cls, gap=args.gap == 'TRUE')
        
    elif args.backbone == 'wide_resnet':
        model = wide_encoder(meta_dict['image_size'], meta_dict['image_channels'], args.latent_size, 40, 4, 0, num_classes=known_num_classes, num_rp_per_cls=args.num_rp_per_cls)
    else:
        raise ValueError(args.backbone + ' is not supported.')
    if args.load_history_model:
        model.load_state_dict(torch.load(experiment_ckpt_path + '/best_model.pt'))
    model.cuda()
    
    num_params = count_parameters(model)
    logger.info("Number of model parameters: " + str(num_params))

    criterion = nn.CrossEntropyLoss(reduction='none')

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.patience, gamma=0.1)

    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    train_n = len(train_loader.dataset)
    best_used_running_loss = 100000000
    best_val_acc = 0.

    last_lr = False
    for epoch in range(args.n_epochs):
        model.train()
        logger.info("EPOCH " + str(epoch))

        actual_lr = None
        for param_group in optimizer.param_groups:
            curr_lr = param_group['lr']
            if actual_lr is None:
                actual_lr = curr_lr
            else:
                if curr_lr != actual_lr:
                    raise ValueError("some param groups have different lr")
        logger.info("Learning rate: " + str(actual_lr))
        if actual_lr < 10 ** (-7):
            last_lr = True
        tic = time.time()
        with tqdm(total=train_n) as pbar:
            for i, data in enumerate(train_loader, 0):
                # get the inputs & combine positive and negatives together
                img = data['image']
                b = img.shape[0]
                img = img.cuda()

                labels = data['label']
                labels = labels.cuda()

                # zero the parameter gradients
                optimizer.zero_grad()
                outputs = model.forward(img)

                # Compute RPL loss
                loss, open_loss, closed_loss, logits = compute_rpl_loss(model, outputs, labels, criterion, args.lamb, args.gamma, args.divide == 'TRUE')
                loss.backward()
                optimizer.step()

                # update loss for this epoch
                running_loss = loss.item()

                probs = torch.softmax(logits, dim=1)
                max_probs, max_indices = torch.max(probs, 1)
                acc = torch.sum(max_indices == labels).item() / b
                toc = time.time()
                pbar.set_description(
                    (
                        "{:.1f}s - loss: {:.3f} - acc: {:.3f}".format(
                            (toc - tic), running_loss, acc
                        )
                    )
                )
                pbar.update(b)
        model.eval()
        used_running_loss, used_val_acc = evaluate_val(model, criterion, val_loader, args.gamma, args.lamb, args.divide, logger)
        
        # Adjust learning rate
        scheduler.step()

        # case where only acc is top
        if used_val_acc > best_val_acc:
            best_val_acc = used_val_acc
            torch.save(model.state_dict(), experiment_ckpt_path + '/best_model.pt')
            
        if used_running_loss < best_used_running_loss:
            best_used_running_loss = used_running_loss
