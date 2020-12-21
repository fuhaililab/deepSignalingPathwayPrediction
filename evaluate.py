"""Evaluate SigGraInferNet.

"""

import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import util
from args import get_test_args
from model import SigGraInferNet_GCN,SigGraInferNet_GAT,FocalLoss
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
from json import dumps

def get_model(log,args):
    if args.model_name == "GAT":
        model = SigGraInferNet_GAT(feature_input_size=args.feature_input_size,
                                   feature_output_size=args.feature_output_size,
                                   PPI_input_size=args.PPI_input_size,
                                   PPI_output_size=args.PPI_output_size,
                                   num_GAT=args.num_GNN,
                                   num_head=args.num_head,
                                   drop_prob=args.drop_prob)
    elif args.model_name == "GCN":
        model = SigGraInferNet_GCN(feature_input_size=args.feature_input_size,
                              feature_output_size=args.feature_output_size,
                              PPI_input_size=args.PPI_input_size,
                              PPI_output_size=args.PPI_output_size,
                              num_GCN=args.num_GNN,
                              drop_prob=args.drop_prob)

    else:
        raise ValueError("Model name doesn't exist.")
    model = nn.DataParallel(model, args.gpu_ids)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    if args.load_path:
        log.info(f'Loading checkpoint from {args.load_path}...')
        model, step = util.load_model(model, args.load_path, args.gpu_ids)
    else:
        step = 0

    return model,step




def main(args):
    # Set up logging and devices
    args.save_dir = util.get_save_dir(args.save_dir, args.name, type="evaluation")
    log = util.get_logger(args.save_dir, args.name)
    device, args.gpu_ids = util.get_available_devices()
    log.info(f'Args: {dumps(vars(args), indent=4, sort_keys=True)}')
    args.batch_size *= max(1, len(args.gpu_ids))


    # Get your model
    log.info('Building model...')
    model, step=get_model(log,args)
    model = model.to(device)
    model.eval()

    # Get data loader
    log.info('Building dataset...')

    dev_dataset = util.load_dataset(args.test_file,args.PPI_dir,args.PPI_gene_feature_dir,
                                    args.PPI_gene_query_dict_dir,args.max_nodes,train=False)

    dev_loader = data.DataLoader(dev_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_workers=args.num_workers,
                                 collate_fn=util.collate_fn)

    # Train
    log.info('Evaluating...')

    #get loss computer
    cri=FocalLoss(alpha=torch.tensor([args.alpha,1]).to(device),gamma=args.gamma)

    loss_meter = util.AverageMeter()
    ground_true = dev_loader.dataset.y_list
    ground_true = ground_true.to(device)
    predict_list=torch.zeros([dev_loader.dataset.__len__(),2],dtype=torch.float)
    predict_list = predict_list.to(device)
    sample_index=0
    with torch.no_grad(), \
         tqdm(total=len(dev_loader.dataset)) as progress_bar:
        for batch_a, batch_bio_a, batch_A, batch_b, batch_bio_b, batch_B, batch_y in dev_loader:
            # Setup for forward
            batch_a = batch_a.to(device)
            batch_bio_a = batch_bio_a.to(device)
            batch_A = batch_A.to(device)
            batch_bio_b = batch_bio_b.to(device)
            batch_b = batch_b.to(device)
            batch_B = batch_B.to(device)
            batch_y = batch_y.to(device)
            batch_y = batch_y.long()
            batch_size = batch_bio_a.size(0)
            # Forward
            output= model(batch_a, batch_bio_a, batch_A, batch_b, batch_bio_b, batch_B)
            loss = cri(output, batch_y)
            loss_val = loss.item()
            loss_meter.update(loss_val, batch_size)
            predict_list[sample_index:sample_index+batch_size]=output
            sample_index=sample_index+batch_size

            # Log info
            progress_bar.update(batch_size)
            progress_bar.set_postfix(NLL=loss_meter.avg)

    results = util.metrics_compute(predict_list, ground_true)

    log.info("Evaluation result of model:")
    log.info(f"Loss in test dataset is {loss_meter.avg}")
    log.info(f"Accuracy:{results['Accuracy']}, AUC:{results['AUC']}, Recall:{results['Recall']},Precision:{results['Precision']},Specificity:{results['Specificity']}")
    log.info(f"TP:{results['TP']},FN:{results['FN']}")
    log.info(f"FP:{results['FP']},TN:{results['TN']}")
    log.info("plot prediction curve...")
    ROC_AUC(results["fpr"],results["tpr"],results["AUC"],os.path.join(args.save_dir,"ROC_curve.pdf"))
    log.info("Save evaluation result...")
    np.savez(os.path.join(args.save_dir,"results.npz"),predict=np.array(predict_list.cpu().tolist()),result=results)


def ROC_AUC(fpr,tpr,auc_score,path,figsize=(16,9)):

    pdf = PdfPages(path)
    fig = plt.figure(figsize=figsize)
    plt.plot(fpr,tpr,color='darkorange',
         lw=2, label='ROC curve (area = %0.3f)' % auc_score)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC and AUC Result of Model')
    plt.legend(loc="lower right")
    pdf.savefig(orientation="landscape")
    plt.close()
    pdf.close()
    return auc_score

if __name__ == '__main__':
    main(get_test_args())
