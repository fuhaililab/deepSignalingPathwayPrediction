import ujson as json
import torch.utils.data as data
import numpy as np
import torch
import os
import queue
import shutil
import logging
import tqdm
from sklearn.metrics import roc_curve,auc

def save_json(filename, obj, message=None):
    """Save data in JSON format
    Args:
        filename(str):name of save directory(including file name)
        obj(object):data you want to save
        message(str):anything you want to print
    """
    if message is not None:
        print(f"Saving {message}...")
    with open(filename, "w") as fh:
        json.dump(obj, fh)


class load_dataset(data.Dataset):
    """Create torch dataset used for training and testing
    Args:
        data_path:path of TCGA data
        PPI_path:path of PPI data
        node_path:path of PPI node feature data
        dict_path:path of PPI gene index dict data
        max_nodes:maximum number of nodes in subgraph
    """
    def __init__(self, data_path,PPI_path,node_path,dict_path,max_nodes,train=True):
        super(load_dataset, self).__init__()
        self.train=train
        self.max_nodes=max_nodes
        #load TCGA dataset.
        dataset = np.load(data_path,allow_pickle=True)
        y=torch.from_numpy(dataset["y_list"])
        self.y_list=torch.argmax(y,dim=-1)
        self.source_sample=torch.from_numpy(dataset["source_features"])
        self.target_sample=torch.from_numpy(dataset["target_features"])
        self.pair_list=dataset["pair_list"]
        #load PPI data

        with open(PPI_path,"r") as fh:
            self.PPI=json.load(fh)
        with open(dict_path,"r") as fh:
            self.PPI_gene_dict=json.load(fh)
        self.PPI_node_feature=torch.from_numpy(np.load(node_path)["feature_matrix"])
        self.PPI_feature_size=self.PPI_node_feature.shape[-1]




    def __getitem__(self, idx):
        gene_pair=self.pair_list[idx]
        source_gene=gene_pair[0]
        try:
            source_gene_neighbor=self.PPI[source_gene]
        except:
            source_gene="UNKNOWN"
            source_gene_neighbor=self.PPI[source_gene]
        source_subgraph=[source_gene]
        source_subgraph.extend(source_gene_neighbor)
        source_subgraph_size=len(source_subgraph)
        if(source_subgraph_size>self.max_nodes):
            if(self.train):
                source_subgraph=np.random.choice(source_subgraph,self.max_nodes,replace=False)
            else:
                source_subgraph=source_subgraph[:self.max_nodes]
            source_subgraph_size=self.max_nodes
        a=torch.zeros([self.max_nodes,self.PPI_feature_size],dtype=torch.float)
        A=torch.eye(self.max_nodes,dtype=torch.float)
        for i in range(source_subgraph_size):
            gene=source_subgraph[i]
            index=self.PPI_gene_dict[gene]
            a[i]=self.PPI_node_feature[index]
            edge_list=self.PPI[gene]
            target_in_graph = list(set(edge_list).intersection(set(source_subgraph)))
            if (len(target_in_graph) > 0):
                k = [np.where(source_subgraph == target_gene)[0] for target_gene in target_in_graph]
                A[i, k] = 1

        d = torch.sum(A, dim=-1)
        D = torch.diag_embed(d.pow(-0.5))
        A= torch.matmul(torch.matmul(D, A), D)

        target_gene=gene_pair[1]
        try:
            target_gene_neighbor = self.PPI[target_gene]
        except:
            target_gene="UNKNOWN"
            target_gene_neighbor=self.PPI[target_gene]
        target_subgraph=[target_gene]
        target_subgraph.extend(target_gene_neighbor)
        target_subgraph_size=len(target_subgraph)
        if(target_subgraph_size>self.max_nodes):
            if(self.train):
                target_subgraph=np.random.choice(target_subgraph,self.max_nodes,replace=False)
            else:
                target_subgraph=target_subgraph[:self.max_nodes]
            target_subgraph_size=self.max_nodes
        b=torch.zeros([self.max_nodes,self.PPI_feature_size],dtype=torch.float)
        B=torch.eye(self.max_nodes,dtype=torch.float)
        for i in range(target_subgraph_size):
            gene=target_subgraph[i]
            index=self.PPI_gene_dict[gene]
            b[i]=self.PPI_node_feature[index]
            edge_list=self.PPI[gene]
            target_in_graph = list(set(edge_list).intersection(set(target_subgraph)))
            if (len(target_in_graph) > 0):
                k = [np.where(target_subgraph == target_gene)[0] for target_gene in target_in_graph]
                B[i, k] = 1

        d = torch.sum(B, dim=-1)
        D = torch.diag_embed(d.pow(-0.5))
        B= torch.matmul(torch.matmul(D, B), D)

        example = (
            # return data
            self.source_sample[idx],
            a,
            A,
            self.target_sample[idx],
            b,
            B,
            self.y_list[idx],
            source_subgraph_size,
            target_subgraph_size
        )

        return example

    def __len__(self):
        return self.source_sample.size(0)



def collate_fn(examples):
    """Create batch data if we need further processing of data.
    Args:
        examples (list): List of tuples of the form (x,y). See the  __getitem__ in load_dataset
    """
    def compact_graph_feature(features,max_size):
        feature_matrix_size=features[0].size(-1)
        compact=torch.zeros([len(features),max_size,feature_matrix_size],dtype=torch.float)
        for i, feature in enumerate(features):
            compact[i]=features[i][:max_size,:]
        return compact

    def compact_adj(adj_matrix,max_size):
        compact=torch.zeros([len(adj_matrix),max_size,max_size],dtype=torch.float)
        for i ,matrix in enumerate(adj_matrix):
            compact[i]=adj_matrix[i][:max_size,:max_size]
        return compact

    source_sample,a, A,target_sample,b,B,y_list,source_subgraph_size,target_subgraph_size = zip(*examples)
    max_source_subgraph_size=max(source_subgraph_size)
    max_target_subgraph_size=max(target_subgraph_size)
    a=compact_graph_feature(a,max_source_subgraph_size)
    A=compact_adj(A,max_source_subgraph_size)
    b=compact_graph_feature(b,max_target_subgraph_size)
    B=compact_adj(B,max_target_subgraph_size)

    source_feature_size=source_sample[0].size(-1)
    source_sample_re=torch.zeros([len(source_sample),source_feature_size],dtype=torch.float)
    for i, sample in enumerate(source_sample):
        source_sample_re[i]=sample

    target_feature_size = target_sample[0].size(-1)
    target_sample_re=torch.zeros([len(target_sample),target_feature_size],dtype=torch.float)
    for i, sample in enumerate(target_sample):
        target_sample_re[i]=sample

    y_list_re=torch.zeros([len(y_list)],dtype=torch.float)
    for i,sample in enumerate(y_list):
        y_list_re[i]=sample

    return (source_sample_re,a, A,target_sample_re,b,B,y_list_re)



class AverageMeter:
    """Keep track of average values over time.

    Adapted from:
        > https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        """Reset meter."""
        self.__init__()

    def update(self, val, num_samples=1):
        """Update meter with new value `val`, the average of `num` samples.

        Args:
            val (float): Average value to update the meter with.
            num_samples (int): Number of samples that were averaged to
                produce `val`.
        """
        self.count += num_samples
        self.sum += val * num_samples
        self.avg = self.sum / self.count


class EMA:
    """Exponential moving average of model parameters.
    Args:
        model (torch.nn.Module): Model with parameters whose EMA will be kept.
        decay (float): Decay rate for exponential moving average.
    """
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {}
        self.original = {}

        # Register model parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def __call__(self, model, num_updates):
        decay = min(self.decay, (1.0 + num_updates) / (10.0 + num_updates))
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = \
                    (1.0 - decay) * param.data + decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def assign(self, model):
        """Assign exponential moving average of parameter values to the
        respective parameters.
        Args:
            model (torch.nn.Module): Model to assign parameter values.
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.original[name] = param.data.clone()
                param.data = self.shadow[name]

    def resume(self, model):
        """Restore original parameters to a model. That is, put back
        the values that were in each parameter at the last call to `assign`.
        Args:
            model (torch.nn.Module): Model to assign parameter values.
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                param.data = self.original[name]


class CheckpointSaver:
    """Class to save and load model checkpoints.

    Save the best checkpoints as measured by a metric value passed into the
    `save` method. Overwrite checkpoints with better checkpoints once
    `max_checkpoints` have been saved.

    Args:
        save_dir (str): Directory to save checkpoints.
        max_checkpoints (int): Maximum number of checkpoints to keep before
            overwriting old ones.
        metric_name (str): Name of metric used to determine best model.
        maximize_metric (bool): If true, best checkpoint is that which maximizes
            the metric value passed in via `save`. Otherwise, best checkpoint
            minimizes the metric.
        log (logging.Logger): Optional logger for printing information.
    """
    def __init__(self, save_dir, max_checkpoints, metric_name,
                 maximize_metric=False, log=None):
        super(CheckpointSaver, self).__init__()

        self.save_dir = save_dir
        self.max_checkpoints = max_checkpoints
        self.metric_name = metric_name
        self.maximize_metric = maximize_metric
        self.best_val = None
        self.ckpt_paths = queue.PriorityQueue()
        self.log = log
        self._print(f"Saver will {'max' if maximize_metric else 'min'}imize {metric_name}...")

    def is_best(self, metric_val):
        """Check whether `metric_val` is the best seen so far.

        Args:
            metric_val (float): Metric value to compare to prior checkpoints.
        """
        if metric_val is None:
            # No metric reported
            return False

        if self.best_val is None:
            # No checkpoint saved yet
            return True

        return ((self.maximize_metric and self.best_val < metric_val)
                or (not self.maximize_metric and self.best_val > metric_val))

    def _print(self, message):
        """Print a message if logging is enabled."""
        if self.log is not None:
            self.log.info(message)

    def save(self, step, model, metric_val, device):
        """Save model parameters to disk.

        Args:
            step (int): Total number of examples seen during training so far.
            model (torch.nn.DataParallel): Model to save.
            metric_val (float): Determines whether checkpoint is best so far.
            device (torch.device): Device where model resides.
        """
        ckpt_dict = {
            'model_name': model.__class__.__name__,
            'model_state': model.cpu().state_dict(),
            'step': step
        }
        model.to(device)

        checkpoint_path = os.path.join(self.save_dir,
                                       f'step_{step}.pth.tar')
        torch.save(ckpt_dict, checkpoint_path)
        self._print(f'Saved checkpoint: {checkpoint_path}')

        if self.is_best(metric_val):
            # Save the best model
            self.best_val = metric_val
            best_path = os.path.join(self.save_dir, 'best.pth.tar')
            shutil.copy(checkpoint_path, best_path)
            self._print(f'New best checkpoint at step {step}...')

        # Add checkpoint path to priority queue (lowest priority removed first)
        if self.maximize_metric:
            priority_order = metric_val
        else:
            priority_order = -metric_val

        self.ckpt_paths.put((priority_order, checkpoint_path))

        # Remove a checkpoint if more than max_checkpoints have been saved
        if self.ckpt_paths.qsize() > self.max_checkpoints:
            _, worst_ckpt = self.ckpt_paths.get()
            try:
                os.remove(worst_ckpt)
                self._print(f'Removed checkpoint: {worst_ckpt}')
            except OSError:
                # Avoid crashing if checkpoint has been removed or protected
                pass


def load_model(model, checkpoint_path, gpu_ids, return_step=True):
    """Load model parameters from disk.

    Args:
        model (torch.nn.DataParallel): Load parameters into this model.
        checkpoint_path (str): Path to checkpoint to load.
        gpu_ids (list): GPU IDs for DataParallel.
        return_step (bool): Also return the step at which checkpoint was saved.

    Returns:
        model (torch.nn.DataParallel): Model loaded from checkpoint.
        step (int): Step at which checkpoint was saved. Only if `return_step`.
    """
    device = f"cuda:{gpu_ids[0]}" if gpu_ids else 'cpu'
    ckpt_dict = torch.load(checkpoint_path, map_location=device)

    # Build model, load parameters
    model.load_state_dict(ckpt_dict['model_state'])

    if return_step:
        step = ckpt_dict['step']
        return model, step

    return model


def get_available_devices() -> object:
    """Get IDs of all available GPUs.

    Returns:
        device (torch.device): Main device (GPU 0 or CPU).
        gpu_ids (list): List of IDs of all GPUs that are available.
    """
    gpu_ids = []
    if torch.cuda.is_available():
        gpu_ids += [gpu_id for gpu_id in range(torch.cuda.device_count())]
        device = torch.device(f'cuda:{gpu_ids[0]}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')

    return device, gpu_ids


def get_save_dir(base_dir, name, type, id_max=100):
    """Get a unique save directory by appending the smallest positive integer
    `id < id_max` that is not already taken (i.e., no dir exists with that id).

    Args:
        base_dir (str): Base directory in which to make save directories.
        name (str): Name to identify this training run. Need not be unique.
        training (bool): Save dir. is for training (determines subdirectory).
        id_max (int): Maximum ID number before raising an exception.

    Returns:
        save_dir (str): Path to a new directory with a unique name.
    """
    for uid in range(1, id_max):
        subdir = type
        save_dir = os.path.join(base_dir, subdir, f'{name}-{uid:02d}')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            return save_dir

    raise RuntimeError('Too many save directories created with the same name. \
                       Delete old save directories or use another name.')


def get_logger(log_dir, name):
    """Get a `logging.Logger` instance that prints to the console
    and an auxiliary file.

    Args:
        log_dir (str): Directory in which to create the log file.
        name (str): Name to identify the logs.

    Returns:
        logger (logging.Logger): Logger instance for logging events.
    """
    class StreamHandlerWithTQDM(logging.Handler):
        """Let `logging` print without breaking `tqdm` progress bars.

        See Also:
            > https://stackoverflow.com/questions/38543506
        """
        def emit(self, record):
            try:
                msg = self.format(record)
                tqdm.tqdm.write(msg)
                self.flush()
            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                self.handleError(record)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Log everything (i.e., DEBUG level and above) to a file
    log_path = os.path.join(log_dir, 'log.txt')
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)

    # Log everything except DEBUG level (i.e., INFO level and above) to console
    console_handler = StreamHandlerWithTQDM()
    console_handler.setLevel(logging.INFO)

    # Create format for the logs
    file_formatter = logging.Formatter('[%(asctime)s] %(message)s',
                                       datefmt='%m.%d.%y %H:%M:%S')
    file_handler.setFormatter(file_formatter)
    console_formatter = logging.Formatter('[%(asctime)s] %(message)s',
                                          datefmt='%m.%d.%y %H:%M:%S')
    console_handler.setFormatter(console_formatter)

    # add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def torch_from_json(path, dtype=torch.float32):
    """Load a PyTorch Tensor from a JSON file.

    Args:
        path (str): Path to the JSON file to load.
        dtype (torch.dtype): Data type of loaded array.

    Returns:
        tensor (torch.Tensor): Tensor loaded from JSON file.
    """
    with open(path, 'r') as fh:
        array = np.array(json.load(fh))

    tensor = torch.from_numpy(array).type(dtype)

    return tensor


def metrics_compute(predict,target):
    """compute any metrics you want in validation
    Args:
        predict: model prediction
        target:real value
    """
    target=target.cpu()
    predict=predict.cpu()
    pre=torch.argmax(predict,dim=-1)

    TP=torch.sum((pre==0) & (target ==0)).item()
    FP=torch.sum((pre==0) & (target ==1)).item()
    FN=torch.sum((pre==1) & (target ==0)).item()
    TN=torch.sum((pre==1) & (target ==1)).item()

    accuracy=(TP+TN)/(TP+FP+FN+TN)
    recall=TP/(TP+FN)
    precision=TP/(TP+FP)
    specificity=TN/(TN+FP)
    fpr, tpr, thresholds = roc_curve(target.tolist(), predict[:,1].tolist())
    auc_score = auc(fpr, tpr)
    metrics_result={'Accuracy':accuracy,
                    "Recall":recall,
                    "Precision":precision,
                    "Specificity":specificity,
                    "TN":TN,
                    "FP":FP,
                    "FN":FN,
                    "TP":TP,
                    "fpr":fpr,
                    "tpr":tpr,
                    'AUC':auc_score}

    return metrics_result
