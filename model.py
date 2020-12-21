import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
def clones( module, N):
    return nn.ModuleList(copy.deepcopy(module) for _ in range(N))

class GCN(nn.Module):
    """Graph Convolutional Network
    Args:
        input_size:the size of input feature
        output_size:the size of output feature
        drop_prob: drop out probability
    """
    def __init__(self,input_size,output_size,drop_prob):
        super(GCN, self).__init__()
        self.proj=nn.Linear(input_size,output_size,bias=False)
        self.dropout=nn.Dropout(drop_prob)
        self.batchNorm=nn.BatchNorm1d(output_size)

    def forward(self,x,A):
        x=torch.matmul(A,self.proj(x)) #B * N * H
        x=x.transpose(-1,-2).contiguous()
        x=F.relu(self.batchNorm(x))
        return self.dropout(x.transpose(-1,-2).contiguous())

class GraphPooling(nn.Module):
    """Graph Pooling Layer
    Args:
        hidden_size:the size of input and output feature
    """
    def __init__(self,hidden_size):
        super(GraphPooling, self).__init__()
        self.proj=nn.Linear(hidden_size,hidden_size)
    def forward(self,x):
        x=self.proj(x) # B * N * H
        x=torch.max(x,dim=-2)[0]
        return x.squeeze()


def masking_softmax(att,A):
    """masking softmax in GAT layer"""
    masking=A>0 #B * N * N
    masking=masking.int()
    masking=masking.unsqueeze(1) #B * 1 * N * N
    att=att.masked_fill_(masking==0,-1e30)
    return F.softmax(att,dim=-1) #B * h * N * N

class GraphAttentionLayer(nn.Module):
    """GAT layer
    Args:
        input_size:the size of input feature
        output_size:the size of output feature
        head: number of head in multi-head attention
        drop_prob: drop out probability
    """
    def __init__(self,input_size,output_size,head,drop_prob):
        super(GraphAttentionLayer, self).__init__()
        self.k=output_size//head
        self.head=head
        self.proj=nn.Linear(input_size,output_size,bias=False)
        self.att_proj_list=clones(nn.Linear(2*self.k,1),self.head)
        self.dropout=nn.Dropout(drop_prob)
    def forward(self,x,A):
        B=x.size(0)
        x=self.proj(x) # B * N * H
        x=x.view(B,-1,self.head,self.k).transpose(1,2).contiguous() # B * h * N * k
        att_input=self.attention_input(x) #h * B * N * N * 2k
        att=torch.cat([F.leaky_relu(self.att_proj_list[i](att_input[i]),negative_slope=0.2)for i in range(att_input.size(0))],dim=-1) # B * N * N * h
        att=masking_softmax(att.permute(0,3,1,2),A) # B * h * N * N
        x=F.relu(torch.matmul(att,x)) # B * h * N * k
        x=x.transpose(1,2).contiguous().view(B,-1,self.k*self.head)
        return self.dropout(x) # B * N * hk(H)

    def attention_input(self,x):
        B,h,N,k=x.size()
        Wi=x.repeat_interleave(N,dim=2) # B * h * (N*N) * k
        Wj=x.repeat(1,1,N,1) # B * h * (N*N) * k
        cat=torch.cat([Wi,Wj],dim=-1) #B * h * (N*N) * 2k
        return cat.view(B,h,N,N,2*k).transpose(0,1) # h * B * N * N * 2k


class PPIGE_GCN(nn.Module):
    """PPIGE GCN version
        Args:
        input_size:the size of input feature
        output_size:the size of output feature
        N:number of GCN layer
        drop_prob: drop out probability
    """
    def __init__(self,input_size,output_size,N,drop_prob):
        super(PPIGE_GCN, self).__init__()
        self.init_GCN=GCN(input_size,output_size,drop_prob)
        self.GCN_list=clones(GCN(output_size,output_size,drop_prob),N-1)
    def forward(self,x,A):
        x=self.init_GCN(x,A)
        for l in self.GCN_list:
            x=l(x,A)+x
        return x


class PPIGE_GAT(nn.Module):
    """PPIGE GAT version
        Args:
        input_size:the size of input feature
        output_size:the size of output feature
        N:number of GCN layer
        drop_prob: drop out probability
    """
    def __init__(self,input_size,output_size,N,head,drop_prob):
        super(PPIGE_GAT, self).__init__()
        self.init_proj=nn.Linear(input_size,output_size,drop_prob)
        self.GAT_list=clones(GraphAttentionLayer(output_size,output_size,head,drop_prob),N-1)
    def forward(self,x,A):
        x=self.init_proj(x)
        for l in self.GAT_list:
            x=l(x,A)+x
        return x



class GGE(nn.Module):
    """GGE module
    Args:
        input_size:size of input feature
        drop_prob:dropout probability
    """
    def __init__(self,input_size,output_size,drop_prob):
        super(GGE, self).__init__()
        self.proj1=nn.Linear(input_size,128)
        self.proj2=nn.Linear(128,output_size)
        self.dropout=nn.Dropout(drop_prob)
    def forward(self,x):
        x=self.dropout(F.relu(self.proj1(x))) # B * 128
        x=self.dropout(F.relu(self.proj2(x))) # B * H
        return x

def GAGA(G,h):
    # G: B * N * H
    # h: B * H
    h=h.unsqueeze(1).transpose(-1,-2)
    att=torch.matmul(G,h).squeeze() # B * N
    att=F.softmax(att,dim=-1).unsqueeze(-1) # B * N * 1
    e=torch.mul(G,att).sum(dim=1).squeeze() # B * H
    return e





class LED(nn.Module):
    """LED module for link probability prediction
    Args:
        hidden_size: size of the encoded feature
    """
    def __init__(self,hidden_size):
        super(LED, self).__init__()
        self.proj1=nn.Linear(4*hidden_size,hidden_size)
        self.proj2=nn.Linear(hidden_size,2)
        self.conv=nn.Sequential(nn.Conv1d(2*hidden_size,2*hidden_size,kernel_size=1),nn.MaxPool1d(kernel_size=2))
    def forward(self,a,b):
        #cnn interaction

        cnn_a=a.unsqueeze(1)
        cnn_b=b.unsqueeze(1)
        cnn_x=torch.cat([cnn_a,cnn_b],axis=1)
        cnn_x=self.conv(cnn_x.transpose(-1,-2).contiguous()).squeeze()  #B * 2*H

        x=F.relu(self.proj1(torch.cat([cnn_x,a-b],dim=1))) # B * 4H
        return self.proj2(x)


class SigGraInferNet_GCN(nn.Module):
    """SigGraInferNet_GCN with PPIGE-GCN
    Args:
        feature_input_size: the input dimension of genomic feature
        feature_output_size: the output dimension of genomic feature
        PPI_input_size: the input dimension of protein-protein database feature
        PPI_output_size:the output dimension of protein-protein database feature
        num_GCN: number of GCN layer in model
        drop_prob: dropout probability
    """
    def __init__(self,feature_input_size,feature_output_size,PPI_input_size,PPI_output_size,num_GCN,drop_prob):
        super(SigGraInferNet_GCN, self).__init__()
        self.PPIGE_GCN=PPIGE_GCN(PPI_input_size,PPI_output_size,num_GCN,drop_prob)
        self.GGE=GGE(feature_input_size,feature_output_size,drop_prob)
        self.LED=LED(feature_output_size)

    def forward(self,a,bio_a,A,b,bio_b,B):
        #Graph representation from BioGRID
        G_a=self.PPIGE_GCN(bio_a,A)
        G_b=self.PPIGE_GCN(bio_b,B)

        #gene encoder
        h_a=self.GGE(a)
        h_b=self.GGE(b)

        e_a=GAGA(G_a,h_a)
        e_b=GAGA(G_b,h_b)

        e_a=torch.cat([h_a,e_a],dim=-1)
        e_b=torch.cat([h_b,e_b],dim=-1)

        #link prediction
        predict=self.LED(e_a,e_b)

        return F.log_softmax(predict,dim=-1)

class SigGraInferNet_GAT(nn.Module):
    """SigGraInferNet_GCN with PPIGE-GAT
    Args:
        feature_input_size: the input dimension of genomic feature
        feature_output_size: the output dimension of genomic feature
        PPI_input_size: the input dimension of protein-protein database feature
        PPI_output_size:the output dimension of protein-protein database feature
        num_GAT: number of GCN layer in model
        num_head: number of head in GAT layer
        drop_prob: dropout probability
    """
    def __init__(self,feature_input_size,feature_output_size,PPI_input_size,PPI_output_size,num_GAT,num_head,drop_prob):
        super(SigGraInferNet_GAT, self).__init__()
        self.PPIGE_GAT=PPIGE_GAT(PPI_input_size,PPI_output_size,num_GAT,num_head,drop_prob)
        self.GGE=GGE(feature_input_size,feature_output_size,drop_prob)
        self.LED=LED(feature_output_size)

    def forward(self,a,bio_a,A,b,bio_b,B):
        #Graph representation from BioGRID
        G_a=self.PPIGE_GAT(bio_a,A)
        G_b=self.PPIGE_GAT(bio_b,B)

        #gene encoder
        h_a=self.GGE(a)
        h_b=self.GGE(b)

        e_a=GAGA(G_a,h_a)
        e_b=GAGA(G_b,h_b)

        e_a=torch.cat([h_a,e_a],dim=-1)
        e_b=torch.cat([h_b,e_b],dim=-1)

        #link prediction
        predict=self.LED(e_a,e_b)

        return F.log_softmax(predict,dim=-1)


#focal loss
class FocalLoss(nn.Module):

    def __init__(self, alpha=torch.tensor([1.,1.]),
                 gamma=5.):
        nn.Module.__init__(self)
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.nll_loss(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # prevents nans when probability 0
        alpha=self.alpha[targets]
        F_loss = alpha *(1-pt) ** self.gamma * BCE_loss
        return F_loss.mean()
