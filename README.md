# deepSignalingPathwayPrediction

The repository contains the pytorch implementation of SigGraInferNet.

## Requirement
pytorch 1.0.0 or above
tensorboard 
tqdm
we also provide conda virtual environment for you. You can create a new terminal and cd to the root directory of repository and run following code:<br>
  ```conda env create -f environment.yml```<br>
A new virtual environment called SigGraInferNet will be created. You can activate the environment by:<br>
  ```conda activate SigGraInferNet``` <br>
 
## datasets
Unfortunately we cannot provide the dataset in the repository, you can download corresponding dataset here:<br>
STRING: [https://string-db.org/](https://string-db.org/)<br>
Hallmark gene sets:[https://www.gsea-msigdb.org/gsea/msigdb/collections.jsp](https://www.gsea-msigdb.org/gsea/msigdb/collections.jsp)<br>
TCGA dataset:[https://xenabrowser.net/datapages/](https://xenabrowser.net/datapages/)<br>
The KEGG pathway is get and processed by R with package `graphite`<br>
After you have all the data, we dicuss the detail of how to process the data in the paper.

## training
First, modify the data path in `args.py` and corresponding pytorch data processing function in `util.py` based on your data format. Then run the following code to start training:
```
python train.py -n=SigGraInferNet
```
This will run the SigGraInferNet with default parameter setting. You can also change the parameters in the `Args.py`. You can also visualize the training process in tensorboard:

```
tensorboard --logdir=save --port=5678
```
## Citation

 

