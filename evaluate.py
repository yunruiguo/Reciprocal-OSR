import os
import pickle

import torch

from penalties import compute_rpl_loss, compute_rpl_logits


def collect_rpl_max(model, loader, gamma, total_number, tsne_fea, idx_to_class=None):
    """ Care about 1) identity of max known class 2) distance to reciprocal points of this class. """

    with torch.no_grad():
        confidence_dict = {}
        corr_number = 0
        for _, i in enumerate(idx_to_class):
            confidence_dict[idx_to_class[i]] = []
        for i, data in enumerate(loader, 0):
            # get the inputs & combine positive and negatives together
            img = data['image']
            img = img.cuda()
            label_idx = data['label']
            outputs = model.forward(img)

            logits, dist_to_rp, outputs = compute_rpl_logits(model, outputs, gamma)
            max_distances, max_indices = torch.max(logits, 1)
            corr_number += torch.sum(max_indices == label_idx.cuda()).item()
            probs = torch.softmax(logits, dim=1)
            max_probs, max_indices = torch.max(probs, 1)


            for j in range(0, img.shape[0]):

                if len(tsne_fea[data['label'][j].item()]) < 100:
                    
                    tsne_fea[data['label'][j].item()] += [outputs[j].tolist()]
                correct_leaf = idx_to_class[label_idx[j].item()]
                predicted_leaf_idx = max_indices[j].item()
                dist = max_distances[j].item()
                prob = max_probs[j].item()
                confidence_dict[correct_leaf].append({'prediction': predicted_leaf_idx, 'label': data['label'][j].item(), 'dist': dist, 'prob': prob})
        acc = corr_number / total_number
        print('Closed ACC:', acc)
        return confidence_dict


def evaluate_val(model, criterion, val_loader, gamma, lamb, divide, logger):
    with torch.no_grad():
        used_correct = 0.
        used_total = 0.0
        used_running_loss = 0.
        val_rpl_loss = 0.

        logger.info("beginning validation")

        for i, data in enumerate(val_loader, 0):
            # get the inputs & combine positive and negatives together
            img = data['image']
            img = img.cuda()

            labels = data['label']
            labels = labels.cuda()

            outputs = model.forward(img)

            # Compute RPL loss
            loss, open_loss, closed_loss, logits = compute_rpl_loss(model, outputs, labels, criterion, lamb, gamma,
                                                                    divide == 'TRUE')
            val_rpl_loss += loss.item()

            probs = torch.softmax(logits, dim=1)
            max_probs, max_indices = torch.max(probs, 1)

            used_correct += torch.sum(max_indices == labels).item()
            used_total += probs.shape[0]
            used_running_loss += loss.item()

        used_val_acc = used_correct / (used_total)
        logger.info("Used Validation Accuracy is : " + str(used_val_acc))
        logger.info("Used Average validation loss is: " + str(used_running_loss / used_total))
        logger.info("finished validation")

        return used_running_loss, used_val_acc


def seenval_baseline_thresh(confidence_dict, thresh, cifar=False, label_to_idx=None, folder_to_idx=None,
                            name_to_folder=None, save_path=None, value_key='prob'):
    """ Given a softmax classifier, evaluate performance on seen validation set using thresholds on softmax probabilities. """
    softmax_correct = [0.0 for i in range(0, len(thresh))]
    per_cls_softmax_correct = [{leaf_name: 0.0 for leaf_name in confidence_dict.keys()} for i in range(0, len(thresh))]
    mistake_log = [{leaf_name: {} for leaf_name in confidence_dict.keys()} for i in range(0, len(thresh))]
    semantic_precision = np.zeros((len(thresh),))
    semantic_tot = np.zeros((len(thresh),))
    num_correctly_predict_seen = np.zeros((len(thresh),))
    num_predict_seen = np.zeros((len(thresh),))
    total = 0.0
    # fix class
    for leaf_name in confidence_dict.keys():

        max_indices = torch.tensor([record['idx'] for record in confidence_dict[leaf_name]]).long()
        max_probs = torch.tensor([record[value_key] for record in confidence_dict[leaf_name]])

        if cifar:
            labels = torch.full(max_indices.size(), label_to_idx[leaf_name], dtype=torch.long)
        else:
            labels = torch.full(max_indices.size(), folder_to_idx[name_to_folder[leaf_name]], dtype=torch.long)

        total += max_indices.shape[0]

        for i in range(0, len(thresh)):
            proceed = max_probs.cpu() > torch.full(max_probs.size(), thresh[i])
            num_predict_seen[i] += torch.sum(proceed).item()
            correct_cls = max_indices.cpu() == labels
            correct_vec = correct_cls & proceed
            num_corr = torch.sum(correct_vec).item()
            softmax_correct[i] += num_corr
            per_cls_softmax_correct[i][leaf_name] = num_corr / len(confidence_dict[leaf_name])

    total = float(sum([len(confidence_dict[leaf_name]) for leaf_name in confidence_dict.keys()]))
    success_rate = [softmax_correct[i] / total for i in range(0, len(softmax_correct))]

    info = {'success_rate': success_rate, 'per_cls_softmax_correct': per_cls_softmax_correct,
            'mistake_log': mistake_log,
            'semantic_precision': semantic_precision, 'num_correctly_predict_seen': softmax_correct,
            'num_predict_seen': num_predict_seen, 'total': total}

    if save_path is not None:

        if os.path.exists(save_path):
            raise ValueError(save_path + ' already exists.')

        with open(save_path, 'wb') as f:
            pickle.dump(info, f)

    return info


def unseenval_baseline_thresh(confidence_dict, thresh, save_path=None, value_key='prob'):
    """ Given a softmax classifier, evaluate performance on unseen validation set using thresholds on softmax probabilities. """
    softmax_correct = [0.0 for i in range(0, len(thresh))]
    per_cls_softmax_correct = [{leaf_name: 0.0 for leaf_name in confidence_dict.keys()} for i in range(0, len(thresh))]
    mistake_log = [{leaf_name: {} for leaf_name in confidence_dict.keys()} for i in range(0, len(thresh))]
    semantic_precision = np.zeros((len(thresh),))
    num_predict_seen = np.zeros((len(thresh),))
    # fix class
    for leaf_name in confidence_dict.keys():

        max_indices = torch.tensor([record['idx'] for record in confidence_dict[leaf_name]]).long()
        max_probs = torch.tensor([record[value_key] for record in confidence_dict[leaf_name]])

        for i in range(0, len(thresh)):
            proceed = max_probs.cpu() > torch.full(max_probs.size(), thresh[i])
            num_predict_seen[i] += torch.sum(proceed).item()
            correct_vec = ~proceed
            num_corr = torch.sum(correct_vec).item()
            softmax_correct[i] += num_corr
            per_cls_softmax_correct[i][leaf_name] = num_corr / len(confidence_dict[leaf_name])

    total = float(sum([len(confidence_dict[leaf_name]) for leaf_name in confidence_dict.keys()]))
    success_rate = [softmax_correct[i] / total for i in range(0, len(softmax_correct))]

    info = {'success_rate': success_rate, 'per_cls_softmax_correct': per_cls_softmax_correct,
            'mistake_log': mistake_log,
            'semantic_precision': semantic_precision, 'num_predict_seen': num_predict_seen, 'total': total}

    if save_path is not None:

        if os.path.exists(save_path):
            raise ValueError(save_path + ' already exists.')

        with open(save_path, 'wb') as f:
            pickle.dump(info, f)

    return info