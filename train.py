"""Train SigGraInferNet

"""

import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torch.utils.data as data
import util
from args import get_train_args
from collections import OrderedDict
from json import dumps
from model import SigGraInferNet_GAT,SigGraInferNet_GCN,FocalLoss

from tensorboardX import SummaryWriter
from tqdm import tqdm


def get_model(log,args):
    if args.model_name=="GAT":
        model=SigGraInferNet_GAT(     feature_input_size=args.feature_input_size,
                               feature_output_size=args.feature_output_size,
                               PPI_input_size=args.PPI_input_size,
                               PPI_output_size=args.PPI_output_size,
                               num_GAT=args.num_GNN,
                               num_head=args.num_head,
                               drop_prob=args.drop_prob)
    elif args.model_name=="GCN":
        model=SigGraInferNet_GCN(     feature_input_size=args.feature_input_size,
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
    args.save_dir = util.get_save_dir(args.save_dir, args.name, type="train")
    log = util.get_logger(args.save_dir, args.name)
    tbx = SummaryWriter(args.save_dir)
    device, args.gpu_ids = util.get_available_devices()
    log.info(f'Args: {dumps(vars(args), indent=4, sort_keys=True)}')
    args.batch_size *= max(1, len(args.gpu_ids))

    # Set random seed
    log.info(f'Using random seed {args.seed}...')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Get your model
    log.info('Building model...')
    model, step=get_model(log,args)
    model = model.to(device)
    model.train()

    #Exponential moving average
    ema = util.EMA(model, args.ema_decay)

    # Get saver
    saver = util.CheckpointSaver(args.save_dir,
                                 max_checkpoints=args.max_checkpoints,
                                 metric_name=args.metric_name,
                                 maximize_metric=args.maximize_metric,
                                 log=log)

    # Get optimizer and scheduler
    optimizer = optim.Adam(model.parameters(),lr=args.lr,betas=[0.8,0.999],eps=1e-7,weight_decay=args.l2_wd)

    scheduler=sched.LambdaLR(optimizer,lambda step: 1)

    #get loss computer
    cri=FocalLoss(alpha=torch.tensor([args.alpha,1]).to(device),gamma=args.gamma)

    # Get data loader
    log.info('Building dataset...')

    dev_dataset = util.load_dataset(args.dev_file,args.PPI_dir,args.PPI_gene_feature_dir,
                                    args.PPI_gene_query_dict_dir,args.max_nodes,train=False)


    dev_loader = data.DataLoader(dev_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_workers=args.num_workers,
                                 collate_fn=util.collate_fn)

    train_dataset = util.load_dataset(args.train_file, args.PPI_dir, args.PPI_gene_feature_dir,
                                      args.PPI_gene_query_dict_dir, args.max_nodes)
    train_loader = data.DataLoader(train_dataset,
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   num_workers=args.num_workers,
                                   collate_fn=util.collate_fn)
    # Train
    log.info('Training...')
    steps_till_eval = args.eval_steps
    epoch = 0
    while epoch != args.num_epochs:
        epoch += 1


        log.info(f'Starting epoch {epoch}...')
        with torch.enable_grad(), \
                tqdm(total=len(train_loader.dataset)) as progress_bar:
                for batch_a,batch_bio_a,batch_A,batch_b,batch_bio_b,batch_B,batch_y in train_loader:
                    # Setup for forward
                    batch_a=batch_a.to(device)
                    batch_bio_a=batch_bio_a.to(device)
                    batch_A=batch_A.to(device)
                    batch_bio_b=batch_bio_b.to(device)
                    batch_b=batch_b.to(device)
                    batch_B=batch_B.to(device)
                    batch_y=batch_y.to(device)
                    batch_y=batch_y.long()
                    batch_size=batch_bio_a.size(0)
                    optimizer.zero_grad()
                    # Forward
                    output=model(batch_a,batch_bio_a,batch_A,batch_b,batch_bio_b,batch_B)
                    loss = cri(output, batch_y)
                    #loss = F.nll_loss(output, batch_y)
                    loss_val = loss.item()

                    # Backward
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    ema(model, step // batch_size)

                    # Log info
                    step += batch_size
                    progress_bar.update(batch_size)
                    progress_bar.set_postfix(epoch=epoch,
                                             NLL=loss_val)
                    tbx.add_scalar('train/Loss', loss_val, step)
                    tbx.add_scalar('train/LR',
                                   optimizer.param_groups[0]['lr'],
                                   step)

                    steps_till_eval -= batch_size
                    if steps_till_eval <= 0:
                        steps_till_eval = args.eval_steps

                        # Evaluate and save checkpoint
                        log.info(f'Evaluating at step {step}...')
                        ema.assign(model)
                        results = evaluate(model, dev_loader,cri, device)
                        saver.save(step, model, results[args.metric_name], device)
                        ema.resume(model)

                        # Log to console
                        results_str = ', '.join(f'{k}: {v:05.5f}' for k, v in results.items())
                        log.info(f'Dev {results_str}')

                        log.info('Visualizing in TensorBoard...')
                        for k, v in results.items():
                            tbx.add_scalar(f'dev/{k}', v, step)


def evaluate(model, data_loader,cri, device):
    loss_meter = util.AverageMeter()
    model.eval()
    ground_true=data_loader.dataset.y_list
    ground_true=ground_true.to(device)
    predict_list=torch.zeros([data_loader.dataset.__len__(),2],dtype=torch.float)
    predict_list=predict_list.to(device)
    sample_index=0
    with torch.no_grad(), \
            tqdm(total=len(data_loader.dataset)) as progress_bar:
        for batch_a, batch_bio_a, batch_A,batch_b,batch_bio_b, batch_B, batch_y in data_loader:
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
            #loss=F.nll_loss(output,batch_y)
            loss_val = loss.item()
            loss_meter.update(loss_val, batch_size)
            predict_list[sample_index:sample_index+batch_size]=output
            sample_index=sample_index+batch_size
            # Log info
            progress_bar.update(batch_size)
            progress_bar.set_postfix(NLL=loss_meter.avg)
    model.train()

    results = util.metrics_compute(predict_list, ground_true)
    results_list = [('Loss', loss_meter.avg),
                    ('Accuracy', results['Accuracy']),
                    ('Recall',results["Recall"]),
                    ('AUC',results["AUC"])]
    results = OrderedDict(results_list)
    return results





if __name__ == '__main__':
    main(get_train_args())



