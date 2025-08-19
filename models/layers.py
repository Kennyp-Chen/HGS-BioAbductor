# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import utils.attention as attention
# from torch.nn.parameter import Parameter
# from torch.autograd import Variable

# import numpy as np, scipy.sparse as sp
# class AttentionPoolingLayer(nn.Module):
#     def __init__(self, input_dim, num_latent_queries, num_heads=3):
#         super(AttentionPoolingLayer, self).__init__()
#         self.num_latent_queries = num_latent_queries
#         self.latent_queries = nn.Parameter(torch.randn(num_latent_queries, input_dim))
#         self.multihead_attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads)
 
#     def forward(self, x):
#         # x shape: (B, N, C)
#         x=x.reshape([x.shape[1],x.shape[0],x.shape[2]])
#         x.reshape
#         B, N, C = x.shape
#         latent_queries = self.latent_queries.unsqueeze(1).expand(-1, B, -1)  # shape: (L, B, C)
#         x = x.permute(1, 0, 2)  # shape: (N, B, C)
        
#         # Multihead Attention
#         attn_output, _ = self.multihead_attn(latent_queries, x, x)  # shape: (L, B, C)
#         attn_output = attn_output.permute(1, 0, 2)  # shape: (B, L, C)
        
#         return attn_output
 

# ###########HGAT################
# class HGNN_ATT(nn.Module):
#     def __init__(self, input_size, n_hid, output_size, dropout=0.3):
#         super(HGNN_ATT, self).__init__()
#         self.dropout = dropout
#         self.gat1 = HyperGraphAttentionLayerSparse(input_size, n_hid, dropout=self.dropout, alpha=0.2, transfer = False, concat=True)
#         self.gat2 = HyperGraphAttentionLayerSparse(n_hid, output_size, dropout=self.dropout, alpha=0.2, transfer = True, concat=False)
        
#     def forward(self, x, H):   
#         x = self.gat1(x, H)
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = self.gat2(x, H)

#         return x

# class HyperGraphAttentionLayerSparse(nn.Module):

#     def __init__(self, in_features, out_features, dropout, alpha, transfer, concat=True, bias=False):
#         super(HyperGraphAttentionLayerSparse, self).__init__()
#         self.dropout = dropout
#         self.in_features = in_features
#         self.out_features = out_features
#         self.alpha = alpha
#         self.concat = concat

#         self.transfer = transfer

#         if self.transfer:
#             self.weight = Parameter(torch.Tensor(self.in_features, self.out_features))
#         else:
#             self.register_parameter('weight', None)

#         self.weight2 = Parameter(torch.Tensor(self.in_features, self.out_features))
#         self.weight3 = Parameter(torch.Tensor(self.out_features, self.out_features))

#         if bias:
#             self.bias = Parameter(torch.Tensor(self.out_features))
#         else:
#             self.register_parameter('bias', None)

#         self.word_context = nn.Embedding(1, self.out_features)
      
#         self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))   
#         self.a2 = nn.Parameter(torch.zeros(size=(2*out_features, 1)))        
#         self.leakyrelu = nn.LeakyReLU(self.alpha)
        

#         self.reset_parameters()

#     def reset_parameters(self):
#         stdv = 1. / math.sqrt(self.out_features)
#         if self.weight is not None:
#             self.weight.data.uniform_(-stdv, stdv)
#         self.weight2.data.uniform_(-stdv, stdv)
#         self.weight3.data.uniform_(-stdv, stdv)
#         if self.bias is not None:
#             self.bias.data.uniform_(-stdv, stdv)
        
#         nn.init.uniform_(self.a.data, -stdv, stdv)
#         nn.init.uniform_(self.a2.data, -stdv, stdv)
#         nn.init.uniform_(self.word_context.weight.data, -stdv, stdv)


#     def forward(self, x, adj):
#         x_4att = x.matmul(self.weight2)

#         if self.transfer:
#             x = x.matmul(self.weight)
#             if self.bias is not None:
#                 x = x + self.bias        

#         N1 = adj.shape[1] #number of edge
#         N2 = adj.shape[2] #number of node

#         pair = adj.nonzero().t()        

#         get = lambda i: x_4att[i][adj[i].nonzero().t()[1]]
#         x1 = torch.cat([get(i) for i in torch.arange(x.shape[0]).int()])


#         q1 = self.word_context.weight[0:].view(1, -1).repeat(x1.shape[0],1).view(x1.shape[0], self.out_features)
        
#         pair_h = torch.cat((q1, x1), dim=-1)
#         pair_e = self.leakyrelu(torch.matmul(pair_h, self.a).squeeze()).t()
#         assert not torch.isnan(pair_e).any()
#         pair_e = F.dropout(pair_e, self.dropout, training=self.training)

#         e = torch.sparse_coo_tensor(pair, pair_e, torch.Size([x.shape[0], N1, N2])).to_dense()

#         zero_vec = -1e9*(torch.ones_like(e)).float()
#         attention = torch.where(adj > 0, e, zero_vec)


#         attention_edge = F.softmax(attention, dim=2)

#         edge = torch.matmul(attention_edge, x)
        
#         edge = F.dropout(edge, self.dropout, training=self.training)

#         edge_4att = edge.matmul(self.weight3)

#         get = lambda i: edge_4att[i][adj[i].nonzero().t()[0]]
#         y1 = torch.cat([get(i) for i in torch.arange(x.shape[0]).int()])

#         get = lambda i: x_4att[i][adj[i].nonzero().t()[1]]
#         q1 = torch.cat([get(i) for i in torch.arange(x.shape[0]).int()])

#         pair_h = torch.cat((q1, y1), dim=-1)
#         pair_e = self.leakyrelu(torch.matmul(pair_h, self.a2).squeeze()).t()
#         assert not torch.isnan(pair_e).any()
#         pair_e = F.dropout(pair_e, self.dropout, training=self.training)

#         e = torch.sparse_coo_tensor(pair, pair_e, torch.Size([x.shape[0], N1, N2])).to_dense()

#         zero_vec = -1e9*(torch.ones_like(e)).float()
#         attention = torch.where(adj > 0, e, zero_vec)

#         attention_node = F.softmax(attention.transpose(1,2), dim=2)

#         node = torch.matmul(attention_node, edge)


#         if self.concat:
#             node = F.elu(node)

#         return node

#     def __repr__(self):
#         return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


# ##########HGAT##################

# class TransformerPooling(nn.Module):
#     def __init__(self, cls_dim, dim_forward, nhead, nlayers, activation, dropout) -> None:
#         '''
#         cls_dim : n_hid of hyperedges
#         dim_forward : feed_forward:cls_dim->dim_forward->cls_dim ; 200->256
#         nhead : cls_dim % nhead = 0 ; 4...
#         n_layer : layer number of transformer block ( MHA + Add&Norm ) 4
#         activation : F.relu;F.gelu 
#         dropout : 0.3-0.5
#         '''
#         super(TransformerPooling, self).__init__()
#         self.cls_token = nn.Parameter(torch.randn(1, 1, cls_dim))
#         encoder_layer = nn.TransformerEncoderLayer(cls_dim, nhead, dim_forward, dropout, activation)
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, nlayers, enable_nested_tensor=False)

#     def forward(self, x):
#         cls_token = self.cls_token.repeat(x.size(0), 1, 1)# bs,1,1; [1,1,nhid]->[bs,1,nhid] 每个病人的初始嵌入是一样的
#         x = torch.cat([ cls_token, x], dim=1) # x:[bs,he,nhid];->[bs,1+he,nhid]

#         x = x.permute(1, 0, 2) #->[1+he,bs,nhid] seq在前
#         x = self.transformer_encoder(x) # learn ->[1+he,bs,nhid]相当于he来决定了病人（1）
#         cls_token = x[0] #[1,bs,nhid]
#         return cls_token# ([Pats,N_hids]) class token 可以理解为汇聚 小token信息后分class的token


# class Linear_pooling(nn.Module):
#     def __init__(self, in_ch_n:int, pooling_layer_nodes:list,) -> None:
#         super(Linear_pooling, self).__init__()
#         self.drop_out = nn.Dropout()
#         self.weight = nn.ParameterList()
#         self.bias = nn.ParameterList()
#         self.layers_number = len(pooling_layer_nodes)
#         if self.layers_number==0:
#             self.weight.append(nn.Parameter(torch.Tensor(in_ch_n, 1)))
#             self.bias.append(nn.Parameter(torch.Tensor(1)))
#         else:
#             for i in range(self.layers_number):
#                 if i == 0:
#                     self.weight.append(nn.Parameter(torch.Tensor(in_ch_n, pooling_layer_nodes[i])))
#                 else:
#                     self.weight.append(nn.Parameter(torch.Tensor(pooling_layer_nodes[i-1], pooling_layer_nodes[i])))
#                 self.bias.append(nn.Parameter(torch.Tensor(pooling_layer_nodes[i])))
#             self.weight.append(nn.Parameter(torch.Tensor(pooling_layer_nodes[-1], 1)))
#             self.bias.append(nn.Parameter(torch.Tensor(1)))

#         self.reset_parameters()
#     def reset_parameters(self): 
        
#         for i, (w, b) in enumerate(zip(self.weight, self.bias)):
#             nn.init.xavier_normal_(w)
#             nn.init.zeros_(b)


#     def forward(self, x):
#         x = x.permute(0, 2, 1)
#         for i, (w, b) in enumerate(zip(self.weight, self.bias)):
#             if i < self.layers_number:
#                 x = F.tanh(F.linear(x, w.T, b))
#                 x = self.drop_out(x)
#             else:
#                 x = F.linear(x, w.T, b)
#         x = x.squeeze(-1)
#         return x
    
# class Pathway_pooling(nn.Module):
#     def __init__(self, mask) -> None:
#         super(Pathway_pooling, self).__init__()
#         self.mask = mask
#         self.weight = nn.ParameterList()
#         self.bias = nn.ParameterList()
#         for h in mask:
#             # in_pathway, out_pathway = h.size()
#             in_pathway, out_pathway = h.shape

#             self.weight.append(nn.Parameter(torch.Tensor(in_pathway, out_pathway)))
#             self.bias.append(nn.Parameter(torch.Tensor(out_pathway)))
#         self.weight.append(nn.Parameter(torch.Tensor(out_pathway, 1)))
#         self.bias.append(nn.Parameter(torch.Tensor(1)))

#         self.reset_parameters()

#     def reset_parameters(self): 
#         for i, (w, b) in enumerate(zip(self.weight, self.bias)):
#             nn.init.xavier_uniform_(w)
#             # nn.init.kaiming_normal_(w)
#             nn.init.zeros_(b) 

#     def forward(self, x):
#         x = x.permute(0, 2, 1)
#         for i, (w, b) in enumerate(zip(self.weight, self.bias)):
#             if i < len(self.mask):
#                 m = self.mask[i].float()
#                 masked_weight = w * m
#                 x = F.tanh(F.linear(x, masked_weight.T, b))# relu
#             else:
#                 # x = F.tanh(F.linear(x, w.T, b))
#                 x = F.linear(x, w.T, b)
#         x = x.squeeze(-1)
#         return x        
# class HGAT_sparse_qy(nn.Module):

#     def __init__(self, in_ch_n, out_ch, dropout, alpha, transfer, concat=True, bias=False, coarsen=False):
#         super(HGAT_sparse_qy, self).__init__()
#         self.e_dropout = nn.Dropout(dropout)
#         self.in_ch_n = in_ch_n
#         self.out_ch = out_ch
#         self.alpha = alpha
#         self.concat = concat
#         self.transfer = transfer
#         # self.batch_norm = nn.BatchNorm1d(out_ch)
#         if self.transfer:
#             self.wt = Parameter(torch.Tensor(self.in_ch_n, self.out_ch))
#         else:
#             self.register_parameter('wt', None)

#         if bias:
#             self.bias = Parameter(torch.Tensor(1, self.out_ch))
#         else:
#             self.register_parameter('bias', None)
        
#         self.a = Parameter(torch.Tensor(self.out_ch, 1))
        
#         self.coarsen = coarsen
#         self.reset_parameters()

        
#     def reset_parameters(self): 
#         stdv = 1. / math.sqrt(self.out_ch)
#         if self.wt is not None:
#             self.wt.data.uniform_(-stdv, stdv)
        
#         if self.bias is not None:
#             self.bias.data.uniform_(-stdv, stdv)
        
#         self.a.data.uniform_(-stdv, stdv)

#     def reset_parameters_xavier(self): 
#         if self.wt is not None:
#             nn.init.xavier_uniform_(self.wt)

#         if self.bias is not None:
#             nn.init.xavier_uniform_(self.bias) 
        
#         nn.init.xavier_uniform_(self.a)

#     def std_scale(self, x):
#         xstd = x.std(1, unbiased=False, keepdim=True)
#         xstd = torch.where(xstd>0, xstd, torch.tensor(1., device=x.device)) 
#         x = (x - x.mean(1, keepdim=True)) / xstd
#         return x

#     def forward(self, x, xe, pair, val=None, e_degs=None, n_degs=None): 
#         if self.transfer:
#             x = torch.einsum('bni,io->bno', x, self.wt)
#             xe = torch.einsum('bni,io->bno', xe, self.wt)
            
#             if self.bias is not None:
#                 x += self.bias.unsqueeze(0).unsqueeze(1)
#                 xe += self.bias.unsqueeze(0).unsqueeze(1)
        
#         n_edge = xe.shape[1] 
#         n_node = x.shape[1]
#         b_sz = x.shape[0]

#         # x = self.batch_norm(x.transpose(1, 2)).transpose(1, 2)
#         # xe = self.batch_norm(xe.transpose(1, 2)).transpose(1, 2)
#         # xe.float()
#         pair_h = xe[:, pair[0], :] * x[:, pair[1], :]
#         # broadcast
#         if val is not None:
#             pair_h *= val.unsqueeze(-1)
            
#         if e_degs is not None and n_degs is not None:
#             pair_h /= (e_degs[:, pair[0]].sqrt().unsqueeze(-1) * n_degs[:, pair[1]].sqrt().unsqueeze(-1))
#         pair_e = torch.einsum('bni,ij->bn', pair_h, self.a).squeeze().half()# 64,1100,1 · 1,150 -> 64,1100,150

#         pair_e = self.e_dropout(pair_e) # direct dropout is stable (no nan)
#         base0 = -1e4
#         pair_e = pair_e.clone()
#         pair_e[pair_e==0] = base0
        
#         e = base0*torch.ones(b_sz, n_edge, n_node, device   =pair.device, dtype=torch.float16)
#         e[:, pair[0], pair[1]] = pair_e

#         attention_edge = F.softmax(e, dim=2) 
#         self.mean_attention_edge = torch.mean(attention_edge, dim=0).detach().cpu().numpy()
#         xe_out = torch.einsum('ben,bno->beo', attention_edge, x) 
        
#         attention_node = F.softmax(e.transpose(1,2), dim=2)

#         x = torch.einsum('bne, beo->bno', attention_node, xe) 
        
#         x = F.elu(x)
#         xe_out = F.elu(xe_out)

#         if self.coarsen:
#             return x, xe_out, torch.exp(e.T) 
#         else:
#             return x, xe_out

#     def __repr__(self):
#         return self.__class__.__name__ + ' (' + str(self.in_ch_n) + ' -> ' + str(self.out_ch) + ')'

# class HGAT_embedding(nn.Module):
#     def __init__(self, incidence_matrixs, in_ch_n, n_hid, dropout, alpha, transfer, concat=True, bias=True, coarsen=False, double=False, jk=False, device='cuda'):
#         """
#         incidence_matrix: [(gene number, pathway number, (pathway number, parent pathway number) ]etc..
#         in_ch_n = n_hid
#         """
#         super(HGAT_embedding, self).__init__()
#         self.e_dropout = dropout
#         self.alpha = alpha
#         self.concat = concat
#         self.transfer = transfer
#         self.hgat_layers = nn.ModuleList()

#         self.e_degs = incidence_matrixs.sum(0)
#         self.n_degs = incidence_matrixs.sum(1)
#         self.pair = incidence_matrixs.nonzero(as_tuple=False).t().to(device)
#         self.HTa = incidence_matrixs
#         self.hgat_layers.append(HGAT_sparse_qy(in_ch_n, 
#                                             n_hid, 
#                                             self.e_dropout, 
#                                             self.alpha, 
#                                             self.transfer, 
#                                             bias, 
#                                             self.concat))
#         if double:
#             self.hgat_layers.append(HGAT_sparse_qy(n_hid, 
#                                                 n_hid, 
#                                                 self.e_dropout, 
#                                                 self.alpha, 
#                                                 self.transfer, 
#                                                 bias, 
#                                                 self.concat))
#         self.coarsen = coarsen

#     def forward(self, x, xe, val=None, e_degs=False, n_degs=False):
#         if e_degs:
#             e_degs = self.e_degs
#         if n_degs:
#             n_degs = self.n_degs

#         for _, layers in enumerate(self.hgat_layers):
#                 x, xe = layers(x, xe, self.pair, val = None)

#         return xe


# #############################qy#############################################
# def weighted_sum(matrix: torch.Tensor, attention: torch.Tensor) -> torch.Tensor:
#     """
#     Takes a matrix of vectors and a set of weights over the rows in the matrix (which we call an
#     "attention" vector), and returns a weighted sum of the rows in the matrix.  This is the typical
#     computation performed after an attention mechanism.
#     Note that while we call this a "matrix" of vectors and an attention "vector", we also handle
#     higher-order tensors.  We always sum over the second-to-last dimension of the "matrix", and we
#     assume that all dimensions in the "matrix" prior to the last dimension are matched in the
#     "vector".  Non-matched dimensions in the "vector" must be `directly after the batch dimension`.
#     For example, say I have a "matrix" with dimensions `(batch_size, num_queries, num_words,
#     embedding_dim)`.  The attention "vector" then must have at least those dimensions, and could
#     have more. Both:
#         - `(batch_size, num_queries, num_words)` (distribution over words for each query)
#         - `(batch_size, num_documents, num_queries, num_words)` (distribution over words in a
#           query for each document)
#     are valid input "vectors", producing tensors of shape:
#     `(batch_size, num_queries, embedding_dim)` and
#     `(batch_size, num_documents, num_queries, embedding_dim)` respectively.
#     """
    
#     if attention.dim() == 2 and matrix.dim() == 3:
#         return attention.unsqueeze(1).bmm(matrix).squeeze(1)
#     if attention.dim() == 3 and matrix.dim() == 3:
#         return attention.bmm(matrix)
#     if matrix.dim() - 1 < attention.dim():
#         expanded_size = list(matrix.size())
#         for i in range(attention.dim() - matrix.dim() + 1):
#             matrix = matrix.unsqueeze(1)
#             expanded_size.insert(i + 1, attention.size(i + 1))
#         matrix = matrix.expand(*expanded_size)
#     intermediate = attention.unsqueeze(-1).expand_as(matrix) * matrix
#     return intermediate.sum(dim=-2)

# def masked_sum(
#     vector: torch.Tensor, mask: torch.BoolTensor, dim: int, keepdim: bool = False) -> torch.Tensor:
#     """
#     **
#     Adapted from AllenNLP's masked mean: 
#     https://github.com/allenai/allennlp/blob/90e98e56c46bc466d4ad7712bab93566afe5d1d0/allennlp/nn/util.py
#     ** 
#     To calculate mean along certain dimensions on masked values
    
#     vector : `torch.Tensor`
#         The vector to calculate mean.
#     mask : `torch.BoolTensor`
#         The mask of the vector. It must be broadcastable with vector.
#     dim : `int`
#         The dimension to calculate mean
#     keepdim : `bool`
#         Whether to keep dimension
    
#     `torch.Tensor`
#         A `torch.Tensor` of including the mean values.
#     """
    
#     replaced_vector = vector.masked_fill(~mask, 0.0)
#     value_sum = torch.sum(replaced_vector, dim=dim, keepdim=keepdim)
#     return value_sum 

# class HGNN_conv(nn.Module):
#     def __init__(self, in_ft, out_ft, bias=True):
#         super(HGNN_conv, self).__init__()

#         self.weight = Parameter(torch.Tensor(in_ft, out_ft)) 
#         if bias:
#             self.bias = Parameter(torch.Tensor(out_ft)) 
#         else:
#             self.register_parameter('bias', None)
#         self.reset_parameters()

#     def reset_parameters(self): 
#         stdv = 1. / math.sqrt(self.weight.size(1))
#         self.weight.data.uniform_(-stdv, stdv)
#         if self.bias is not None:
#             self.bias.data.uniform_(-stdv, stdv)
        
#     def reset_parameters_xavier(self): 
#         nn.init.xavier_uniform_(self.weight)
#         if self.bias is not None:
#             nn.init.xavier_uniform_(self.bias)
    
#     def forward(self, x: torch.Tensor, G: torch.Tensor):
#         x = x.matmul(self.weight)
#         if self.bias is not None:
#             x = x + self.bias
#         # if 3d then  
#         G = G.unsqueeze(0).expand(x.shape[0],G.shape[0],G.shape[1])

#         x = G.matmul(x)

#         return x
    
# class HGS_fc(nn.Module):
#     '''
#     HyperGraphSurv full connection layer
#     '''
#     def __init__(self, in_ch, out_ch):
#         super(HGS_fc, self).__init__()
#         self.fc = nn.Linear(in_ch, out_ch)
#         nn.init.xavier_uniform_(self.fc.weight)## change for Hpsurv
#         nn.init.zeros_(self.fc.bias)## change for Hpsurv
#     def forward(self, x):
#         return self.fc(x)

# class HGNN_fc(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super(HGNN_fc, self).__init__()
#         self.fc = nn.Linear(in_ch, out_ch)
#     def forward(self, x):
#         return self.fc(x)

# class HGNN_sg_attn_simple(nn.Module):
#     def __init__(self, edim, ddim, atype='additive'):
#         super(HGNN_sg_attn, self).__init__()
#         if atype == 'additive':
#             self.attention = attention.AdditiveAttention(edim, ddim)
#         elif atype == 'multiplicative':
#             self.attention = attention.MultiplicativeAttention(edim, ddim)
#         else:
#             exit(f'unrecognized attention type {atype}')


#     def forward(self, x, sgs):
#         xsize = list(x.size())
#         bsize = sgs.shape[0]
#         b_attn = torch.matmul(sgs, x)
#         y = self.attention(b_attn, x)
#         return y

# class HGNN_sg_attn_Allen(nn.Module):
#     def __init__(self, vdim, mdim, atype='additive'):
#         super(HGNN_sg_attn_Allen, self).__init__()
#         if atype == 'additive':
#             self.attention = attention.AdditiveAttention(vdim, mdim)
#         elif atype == 'dotprod':
#             self.attention = attention.DotProductAttention()
#         else:
#             exit(f'unrecognized attention type {atype}')


#     def forward(self, x, sgs):
#         xsize = list(x.size())
#         bsize = sgs.shape[0]
#         b_attn = torch.matmul(sgs, x)
#         attn_wts = self.attention(b_attn, x.unsqueeze(0).expand(bsize, *xsize))
#         x = torch.matmul(attn_wts, x)
#         return x
    
# class HGNN_sg_attn(nn.Module):
#     def __init__(self, vdim, mdim, atype='additive'):# jk=F,vdim = n_hid
#         super(HGNN_sg_attn, self).__init__()
#         self.attn_vector = torch.nn.Parameter(torch.zeros((vdim,1), dtype=torch.float), requires_grad=True)   
        
#         stdv = 1. / math.sqrt(vdim) 
#         self.attn_vector.data.uniform_(-stdv, stdv)


#     def forward(self, x, sgs):
#         xsize = list(x.size())# gene dim , n
#         bsize = sgs.shape[0]# num_samples,gene
#         attn_wts = torch.matmul(x, self.attn_vector) #(n,200)·(200,1)=(n,1)[1100*200]@[200*1] = [1100*1]
#         attn_wts = attn_wts.squeeze().unsqueeze(0).expand(bsize, xsize[0]) # 618,1100扩展1100维 最终有一些元素为0
#         # n,1-> n -> 1,n -> s,n
#         # print(sgs.shape,attn_wts.shape,x.shape)
#         x = torch.matmul(sgs*attn_wts, x) #(618,1100)·(, 200) = (s,200)
#         return x

# class HHGNN_sg_attn(nn.Module):
#     def __init__(self, vdim, mdim, atype='additive'):
#         super(HHGNN_sg_attn, self).__init__()
#         self.attn_vector = torch.nn.Parameter(torch.zeros((vdim,mdim), dtype=torch.float), requires_grad=True)   
        
#         stdv = 1. / math.sqrt(vdim) 
#         self.attn_vector.data.uniform_(-stdv, stdv)


#     def forward(self, x, sgs):
        
#         attn_wts = torch.matmul(sgs, self.attn_vector) # sgs[patience_dim, gene_dim] attn_vector [ gene_dim, pathway_dim ]
#         output = torch.matmul(attn_wts, x) #[patience_dim, gene_dim] mm [gene_dim, pathway_dim]
#         return output

# class HGNN_sg_attn_multiplicative(nn.Module):
#     def __init__(self, vdim, mdim, atype='additive'):
#         super(HGNN_sg_attn, self).__init__()
#         self.W = nn.Parameter(torch.FloatTensor(mdim, vdim), requires_grad=True)
#         nn.init.xavier_uniform_(self.W)


#     def forward(self, x, sgs):
#         x = sgs.matmul(x).matmul(self.W)
#         return x
    
# class HGNN_embedding(nn.Module):
#     def __init__(self, in_ch, n_hid, dropout=0.5):
#         super(HGNN_embedding, self).__init__()
#         self.dropout = dropout
#         self.hgc1 = HGNN_conv(in_ch, n_hid)
#         self.hgc2 = HGNN_conv(n_hid, n_hid)

#     def forward(self, x, G):
#         x = F.relu(self.hgc1(x, G))
#         x = F.dropout(x, self.dropout)
#         x = F.relu(self.hgc2(x, G))
#         return x


# class HGNN_classifier(nn.Module):
#     def __init__(self, n_hid, n_class):
#         super(HGNN_classifier, self).__init__()
#         self.fc1 = nn.Linear(n_hid, n_class)

#     def forward(self, x):
#         x = self.fc1(x)
#         return x

# class H3HGAT_sparse3(nn.Module):
#     '''
#     尝试从3传递到节点，
#     '''
#     def __init__(self, in_ch_n1,in_ch_n2,in_ch_n3, out_ch, dropout, alpha, transfer, concat=True, bias=False, coarsen=False):
#         super(H3HGAT_sparse3, self).__init__()
#         self.e_dropout = nn.Dropout(dropout)
#         self.in_ch_n1 = in_ch_n1
#         self.in_ch_n2 = in_ch_n2
#         self.in_ch_n3 = in_ch_n3
#         self.out_ch = out_ch
#         self.alpha = alpha
#         self.concat = concat
#         self.transfer = transfer

#         if self.transfer:
#             self.wt1 = Parameter(torch.Tensor(self.in_ch_n1, self.out_ch))# n,200
#             self.wt2 = Parameter(torch.Tensor(self.in_ch_n2, self.out_ch))# e1,200
#             self.wt3 = Parameter(torch.Tensor(self.in_ch_n3, self.out_ch))# e1,200

#         else:
#             self.register_parameter('wt1', None)
#             self.register_parameter('wt2', None)
#             self.register_parameter('wt3', None)


#         if bias:
#             self.bias = Parameter(torch.Tensor(1, self.out_ch))
#         else:
#             self.register_parameter('bias', None)
        
#         self.coarsen = coarsen
#         self.reset_parameters()

        
#     def reset_parameters(self): 
#         stdv = 1. / math.sqrt(self.out_ch)
#         if self.wt1 is not None:
#             self.wt1.data.uniform_(-stdv, stdv)
#         if self.wt2 is not None:
#             self.wt2.data.uniform_(-stdv, stdv)
#         if self.wt3 is not None:
#             self.wt3.data.uniform_(-stdv, stdv)
        
        
#         if self.bias is not None:
#             self.bias.data.uniform_(-stdv, stdv)
        

#     def reset_parameters_xavier(self): #?
#         if self.wt1 is not None:
#             nn.init.xavier_uniform_(self.wt1)
#         if self.wt2 is not None:
#             nn.init.xavier_uniform_(self.wt2)
#         if self.wt3 is not None:
#             nn.init.xavier_uniform_(self.wt3)
#         if self.bias is not None:
#             nn.init.xavier_uniform_(self.bias) 
        

#     def std_scale(self, x):
#         xstd = x.std(1, unbiased=False, keepdim=True)
#         xstd = torch.where(xstd>0, xstd, torch.tensor(1., device=x.device)) 
#         x = (x - x.mean(1, keepdim=True)) / xstd
#         return x
#     def forward(self, x, xe1,xe2,xe3, pairEN,pairE21,pairE32, a, val=None, e_degs=None, n_degs=None): 
#             if self.transfer:
#                 x = x.mm(self.wt1) 
#                 # n*n · n*200 = n*200 , 1st：I * wt = wt
#                 xe1 = xe1.mm(self.wt1) 
#                 # e*n · n*200 = e*200 , 1st: H.T_stand * wt, include node connect inf
#                 xe2 = xe2.mm(self.wt2)
#                 # e2,e1
#                 xe3 = xe3.mm(self.wt3)
                
#                 #bias 是同时更新 
#                 if self.bias is not None:
#                     x = x + self.bias
#                     # n*200 + 1*200 ,node+bias
#                     xe1 = xe1 + self.bias
#                     xe2 = xe2 + self.bias
#                     # e*200 + 1*200 ,wt*edge+bias        
                                
#             # x = n*200 =wt+b
#             # xe = e*200 =wt·HTs+b
#             n_edge3 = xe3.shape[0]
#             n_edge2 = xe2.shape[0]  # e2_num
#             n_edge1 = xe1.shape[0]  # e1_num
#             n_node  = x.shape[0]    # n_num

#             # pair:# 2 * relation_di(37499), incidence matrix zip
#             # 1st row: pathway indexs ;2nd row: gene indexs - both can duplicated , presents the connection relation of them.
#             if val is None:
#                 pairEN_h = xe1[ pairEN[0] ] * x[ pairEN[1] ] # 指定行的每一元素对应相乘
#                 pairE21_h = xe2[ pairE21[0] ] * xe1[ pairE21[1] ] # 指定行的每一元素对应相乘
#                 pairE32_h = xe3[ pairE32[0] ] * xe2[ pairE32[1] ] # 指定行的每一元素对应相乘
                
#                 # pair_h = hypE_emb[pathway_indxs]*node_emb[gene_idx], add the connection inf
#                 # pair_h = r*200 X r*200 = r*200【r=pair.shape[1],即节点超边对数】
#             else:
#                 pairEN_h = xe1[ pairEN[0] ] * x[ pairEN[1] ] * val# val用于pair_h的缩放
#                 pairE21_h = xe2[ pairE21[0] ] * xe1[ pairE21[1] ] * val# val用于pair_h的缩放
#                 pairE32_h = xe3[ pairE32[0] ] * xe2[ pairE32[1] ] * val# 指定行的每一元素对应相乘

#             # None ? 
#             if e_degs is not None:
#                 pairEN_h /= e_degs[ pairEN[0] ].sqrt().unsqueeze(-1) 


#             if n_degs is not None:
#                 pairEN_h /= n_degs[ pairEN[1] ].sqrt().unsqueeze(-1) 

#             pair_e32 = torch.mm(pairE32_h, a).squeeze() 
#             pair_e32 = self.e_dropout(pair_e32) # direct dropout is stable (no nan)
#             base0 = -1e10 # large minus number
#             pair_e32[pair_e32==0] = base0 # softmax?
#             e32 = base0*torch.ones(n_edge3, n_edge2, device=pairE32.device)
#             e32[pairE32[0], pairE32[1]] = pair_e32
            


#             pair_e21 = torch.mm(pairE21_h, a).squeeze() 
#             pair_e21 = self.e_dropout(pair_e21) # direct dropout is stable (no nan)
#             base0 = -1e10 # large minus number
#             pair_e21[pair_e21==0] = base0 # softmax?
#             e2 = base0*torch.ones(n_edge2, n_edge1, device=pairE21.device)
#             e2[pairE21[0], pairE21[1]] = pair_e21
            
         
#             pair_en = torch.mm(pairEN_h, a).squeeze() 
#             # pair_e = r*200 · 200*1 .squeeze = torch.Size([r]) CT context vector？
#             pair_en = self.e_dropout(pair_en) # direct dropout is stable (no nan)
#             base0 = -1e10 # large minus number
#             pair_en[pair_en==0] = base0 # softmax?
#             e1 = base0*torch.ones(n_edge1, n_node, device=pairEN.device)
#             # e(e,n)
#             e1[pairEN[0], pairEN[1]] = pair_en
#             # 将pair_e放回到 e*n matrix,无连接的为负极大值，连接的为embedding,带入嵌入的邻接矩阵
#             attention_edge32 = F.softmax(e32, dim=1) 
#             attention_edge21 = F.softmax(e2, dim=1) 
#             attention_edge1node = F.softmax(e1, dim=1)
#             attention_edge23 = F.softmax(e32.transpose(0,1), dim=1) 
#             attention_edge12 = F.softmax(e2.transpose(0,1), dim=1)
#             attention_nodee1 = F.softmax(e1.transpose(0,1), dim=1)
#             # (e,n) dim=1在节点的维度上进行softmax，大的更大小的更小，和为一
#             # 节点对边注意力
            
#             # 节点计算从上层3边开始传入
#             xe2_pass = torch.mm(attention_edge23, xe3) 
#             xe1_pass = torch.mm(attention_edge12, xe2_pass) 
#             x_out = torch.mm(attention_nodee1, xe1_pass)  
#             # 边嵌入从上一层计算图传递   
#             xe3_out = torch.mm(attention_edge32, xe2)  
#             xe2_out = torch.mm(attention_edge21, xe1)  
#             xe1_out = torch.mm(attention_edge1node, x)
#             ## attention calculation done 
#             # -----------------------------------------------------------

#             # elu x xe, node
#             if self.concat:# False 
#                 x_out = F.elu(x_out)
#                 xe1_out = F.elu(xe1_out)
#                 xe2_out = F.elu(xe2_out)
#                 xe3_out = F.elu(xe3_out)

#             else:
#                 x_out = F.elu(x_out)
#                 xe1_out = F.elu(xe1_out)
#                 xe2_out = F.elu(xe2_out)
#                 xe3_out = F.elu(xe3_out)
#             # 对嵌入非线性激活
#             # embedding and edge embeding done

#             if self.coarsen:# 粗糙
#                 return x_out, xe1_out,xe2_out,xe3_out, torch.exp(e1.T) # for what?
#             else:
#                 return x_out, xe1_out,xe2_out,xe3_out# x(n,200);xe1(e1,200);xe2(e2,200)

# class _H3HGAT_sparse2(nn.Module):
#     '''
#     1. 尝试在两层超边的版本一中加入第三层超边；测试上效果差不多
#     2. 尝试共用wt 运行结果与wt123相同
#     '''
#     def __init__(self, in_ch_n1,in_ch_n2,in_ch_n3, out_ch, dropout, alpha, transfer, concat=True, bias=False, coarsen=False):
#         super(_H3HGAT_sparse2, self).__init__()
#         self.e_dropout = nn.Dropout(dropout)
#         self.in_ch_n1 = in_ch_n1
#         self.in_ch_n2 = in_ch_n2
#         self.in_ch_n3 = in_ch_n3
#         self.out_ch = out_ch
#         self.alpha = alpha
#         self.concat = concat
#         self.transfer = transfer

#         if self.transfer:
#             self.wt1 = Parameter(torch.Tensor(self.in_ch_n1, self.out_ch))# n,200
#             self.wt2 = Parameter(torch.Tensor(self.in_ch_n2, self.out_ch))# e1,200
#             self.wt3 = Parameter(torch.Tensor(self.in_ch_n3, self.out_ch))# e1,200

#         else:
#             self.register_parameter('wt1', None)
#             self.register_parameter('wt2', None)
#             self.register_parameter('wt3', None)


#         if bias:
#             self.bias = Parameter(torch.Tensor(1, self.out_ch))
#         else:
#             self.register_parameter('bias', None)
        
#         self.coarsen = coarsen
#         self.reset_parameters()

        
#     def reset_parameters(self): 
#         stdv = 1. / math.sqrt(self.out_ch)
#         if self.wt1 is not None:
#             self.wt1.data.uniform_(-stdv, stdv)
#         if self.wt2 is not None:
#             self.wt2.data.uniform_(-stdv, stdv)
#         if self.wt3 is not None:
#             self.wt3.data.uniform_(-stdv, stdv)
        
        
#         if self.bias is not None:
#             self.bias.data.uniform_(-stdv, stdv)
        

#     def reset_parameters_xavier(self): #?
#         if self.wt1 is not None:
#             nn.init.xavier_uniform_(self.wt1)
#         if self.wt2 is not None:
#             nn.init.xavier_uniform_(self.wt2)
#         if self.wt3 is not None:
#             nn.init.xavier_uniform_(self.wt3)
#         if self.bias is not None:
#             nn.init.xavier_uniform_(self.bias) 
        

#     def std_scale(self, x):
#         xstd = x.std(1, unbiased=False, keepdim=True)
#         xstd = torch.where(xstd>0, xstd, torch.tensor(1., device=x.device)) 
#         x = (x - x.mean(1, keepdim=True)) / xstd
#         return x
#     def forward(self, x, xe1,xe2,xe3, pairEN,pairE21,pairE32, a, val=None, e_degs=None, n_degs=None): 
#             if self.transfer:
#                 x = x.mm(self.wt1) 
#                 xe1 = xe1.mm(self.wt1) 
#                 xe2 = xe2.mm(self.wt2)
#                 xe3 = xe3.mm(self.wt3)
#                 if self.bias is not None:
#                     x = x + self.bias
#                     xe1 = xe1 + self.bias
#                     xe2 = xe2 + self.bias      
                                
#             n_edge3 = xe3.shape[0]
#             n_edge2 = xe2.shape[0]  
#             n_edge1 = xe1.shape[0]  
#             n_node  = x.shape[0]    

#             if val is None:
#                 pairEN_h = xe1[ pairEN[0] ] * x[ pairEN[1] ] # 指定行的每一元素对应相乘
#                 pairE21_h = xe2[ pairE21[0] ] * xe1[ pairE21[1] ] # 指定行的每一元素对应相乘
#                 pairE32_h = xe3[ pairE32[0] ] * xe2[ pairE32[1] ] # 指定行的每一元素对应相乘
                
#                 # pair_h = hypE_emb[pathway_indxs]*node_emb[gene_idx], add the connection inf
#                 # pair_h = r*200 X r*200 = r*200【r=pair.shape[1],即节点超边对数】
#             else:
#                 pairEN_h = xe1[ pairEN[0] ] * x[ pairEN[1] ] * val# val用于pair_h的缩放
#                 pairE21_h = xe2[ pairE21[0] ] * xe1[ pairE21[1] ] * val# val用于pair_h的缩放
#                 pairE32_h = xe3[ pairE32[0] ] * xe2[ pairE32[1] ] * val# 指定行的每一元素对应相乘

#             if e_degs is not None:
#                 pairEN_h /= e_degs[ pairEN[0] ].sqrt().unsqueeze(-1) 
#             if n_degs is not None:
#                 pairEN_h /= n_degs[ pairEN[1] ].sqrt().unsqueeze(-1) 

#             pair_e32 = torch.mm(pairE32_h, a).squeeze() 
#             pair_e32 = self.e_dropout(pair_e32) # direct dropout is stable (no nan)
#             base0 = -1e10 # large minus number
#             pair_e32[pair_e32==0] = base0 # softmax?
#             e32 = base0*torch.ones(n_edge3, n_edge2, device=pairE32.device)
#             e32[pairE32[0], pairE32[1]] = pair_e32
#             attention_edge3 = F.softmax(e32, dim=1) 


#             pair_e21 = torch.mm(pairE21_h, a).squeeze() 
#             pair_e21 = self.e_dropout(pair_e21) # direct dropout is stable (no nan)
#             base0 = -1e10 # large minus number
#             pair_e21[pair_e21==0] = base0 # softmax?
#             e2 = base0*torch.ones(n_edge2, n_edge1, device=pairE21.device)
#             e2[pairE21[0], pairE21[1]] = pair_e21
#             attention_edge2 = F.softmax(e2, dim=1) 
         
#             pair_en = torch.mm(pairEN_h, a).squeeze() 
#             # pair_e = r*200 · 200*1 .squeeze = torch.Size([r]) CT context vector？
#             pair_en = self.e_dropout(pair_en) # direct dropout is stable (no nan)
#             base0 = -1e10 # large minus number
#             pair_en[pair_en==0] = base0 # softmax?
#             e1 = base0*torch.ones(n_edge1, n_node, device=pairEN.device)
#             # e(e,n)
#             e1[pairEN[0], pairEN[1]] = pair_en
#             # 将pair_e放回到 e*n matrix,无连接的为负极大值，连接的为embedding,带入嵌入的邻接矩阵
#             attention_edge1 = F.softmax(e1, dim=1) 
#             # (e,n) dim=1在节点的维度上进行softmax，大的更大小的更小，和为一
#             # 节点对边注意力

#             xe1_out = torch.mm(attention_edge1, x) 
#             # xe2_out = torch.mm(attention_edge2, xe1)
#             xe2_out = torch.mm(attention_edge2, xe1_out) 
#             # (e2,e1)(e1,200)
#             xe3_out = torch.mm(attention_edge3, xe2_out) 
            
#             # 超边嵌入(e,200)xe_out= (e,n）·(n,200) = 对节点的注意力·节点嵌入成为超边嵌入，(e,200)
#             attention_node = F.softmax(e1.transpose(0,1), dim=1)
#             # 同上 边对节点注意力 (n,e)
#             x = torch.mm(attention_node, xe1) 
#             # 节点嵌入(n,200)

#             ## attention calculation done 
#             # -----------------------------------------------------------

#             # elu x xe, node
#             if self.concat:# False 
#                 x = F.elu(x)
#                 xe1_out = F.elu(xe1_out)
#                 xe2_out = F.elu(xe2_out)
#                 xe3_out = F.elu(xe3_out)

#             else:
#                 x = F.elu(x)
#                 xe1_out = F.elu(xe1_out)
#                 xe2_out = F.elu(xe2_out)
#                 xe3_out = F.elu(xe3_out)
#             # 对嵌入非线性激活
#             # embedding and edge embeding done

#             if self.coarsen:# 粗糙
#                 return x, xe1_out,xe2_out,xe3_out, torch.exp(e1.T) # for what?
#             else:
#                 return x, xe1_out,xe2_out,xe3_out# x(n,200);xe1(e1,200);xe2(e2,200)
            

# class _H2HGAT_sparse12(nn.Module):
#     '''
#     尝试复现H2成功案例
#     总共只有两层超边但是却有所提升
#     实现了上一计算层一级超边传递到此层二级超边传递版本1以及先完成本层一级超边传递以后传入二级超边版本2。
#     其中不加入G时，根据test 版本2会比版本1优秀0.008左右，并且比不加入二级超边优秀0.01左右
#     '''
#     def __init__(self, in_ch_n1,in_ch_n2, out_ch, dropout, alpha, transfer, concat=True, bias=False, coarsen=False):
#         super(_H2HGAT_sparse12, self).__init__()
#         self.e_dropout = nn.Dropout(dropout)
#         self.in_ch_n1 = in_ch_n1
#         self.in_ch_n2 = in_ch_n2
#         self.out_ch = out_ch
#         self.alpha = alpha
#         self.concat = concat
#         self.transfer = transfer

#         if self.transfer:
#             self.wt1 = Parameter(torch.Tensor(self.in_ch_n1, self.out_ch))# n,200
#             self.wt2 = Parameter(torch.Tensor(self.in_ch_n2, self.out_ch))# e1,200

#         else:
#             self.register_parameter('wt1', None)
#             self.register_parameter('wt2', None)


#         if bias:
#             self.bias = Parameter(torch.Tensor(1, self.out_ch))
#         else:
#             self.register_parameter('bias', None)
        
#         self.coarsen = coarsen
#         self.reset_parameters()

        
#     def reset_parameters(self): 
#         stdv = 1. / math.sqrt(self.out_ch)
#         if self.wt1 is not None:
#             self.wt1.data.uniform_(-stdv, stdv)
#         if self.wt2 is not None:
#             self.wt2.data.uniform_(-stdv, stdv)
        
        
#         if self.bias is not None:
#             self.bias.data.uniform_(-stdv, stdv)
        

#     def reset_parameters_xavier(self): #?
#         if self.wt1 is not None:
#             nn.init.xavier_uniform_(self.wt1)
#         if self.wt2 is not None:
#             nn.init.xavier_uniform_(self.wt2)
#         if self.bias is not None:
#             nn.init.xavier_uniform_(self.bias) 
        

#     def std_scale(self, x):
#         xstd = x.std(1, unbiased=False, keepdim=True)
#         xstd = torch.where(xstd>0, xstd, torch.tensor(1., device=x.device)) 
#         x = (x - x.mean(1, keepdim=True)) / xstd
#         return x
#     def forward(self, x, xe1,xe2, pair,pairE, a, val=None, e_degs=None, n_degs=None): 
#         if self.transfer:
#             x = x.mm(self.wt1) 
#             # n*n · n*200 = n*200 , 1st：I * wt = wt
#             xe1 = xe1.mm(self.wt1) 
#             # e*n · n*200 = e*200 , 1st: H.T_stand * wt, include node connect inf
#             xe2 = xe2.mm(self.wt2)
#             # e2,e1
            
#             #bias 是同时更新 
#             if self.bias is not None:
#                 x = x + self.bias
#                 # n*200 + 1*200 ,node+bias
#                 xe1 = xe1 + self.bias
#                 xe2 = xe2 + self.bias
#                 # e*200 + 1*200 ,wt*edge+bias        
                            
#         # x = n*200 =wt+b
#         # xe = e*200 =wt·HTs+b
#         n_edge2 = xe2.shape[0] # e2_num
#         n_edge1 = xe1.shape[0] # e1_num
#         n_node = x.shape[0] # n_num

#         # pair:# 2 * relation_di(37499), incidence matrix zip
#         # 1st row: pathway indexs ;2nd row: gene indexs - both can duplicated , presents the connection relation of them.
#         if val is None:
#             pairEN_h = xe1[ pair[0] ] * x[ pair[1] ] # 指定行的每一元素对应相乘
#             pairE21_h = xe2[ pairE[0] ] * xe1[ pairE[1] ] # 指定行的每一元素对应相乘
            
#             # pair_h = hypE_emb[pathway_indxs]*node_emb[gene_idx], add the connection inf
#             # pair_h = r*200 X r*200 = r*200【r=pair.shape[1],即节点超边对数】
#         else:
#             pairEN_h = xe1[ pair[0] ] * x[ pair[1] ] * val# val用于pair_h的缩放
#             pairE21_h = xe2[ pairE[0] ] * xe1[ pairE[1] ] * val# val用于pair_h的缩放

#         # None ? 
#         if e_degs is not None:
#             pairEN_h /= e_degs[ pair[0] ].sqrt().unsqueeze(-1) 
#             pairE21_h /= e_degs[ pairE[0] ].sqrt().unsqueeze(-1) 

#         if n_degs is not None:
#             pairEN_h /= n_degs[ pair[1] ].sqrt().unsqueeze(-1) 
#             pairE21_h /= n_degs[ pairE[1] ].sqrt().unsqueeze(-1) 
        
#         pair_e21 = torch.mm(pairE21_h, a).squeeze() 
#         pair_e21 = self.e_dropout(pair_e21) # direct dropout is stable (no nan)
#         base0 = -1e10 # large minus number
#         pair_e21[pair_e21==0] = base0 # softmax?
#         e2 = base0*torch.ones(n_edge2, n_edge1, device=pairE.device)
#         e2[pairE[0], pairE[1]] = pair_e21
#         attention_edge2 = F.softmax(e2, dim=1) 
        
#         pair_en = torch.mm(pairEN_h, a).squeeze() 
#         # pair_e = r*200 · 200*1 .squeeze = torch.Size([r]) CT context vector？
#         pair_en = self.e_dropout(pair_en) # direct dropout is stable (no nan)
#         base0 = -1e10 # large minus number
#         pair_en[pair_en==0] = base0 # softmax?
#         e1 = base0*torch.ones(n_edge1, n_node, device=pair.device)
#         # e(e,n)
#         e1[pair[0], pair[1]] = pair_en
#         # 将pair_e放回到 e*n matrix,无连接的为负极大值，连接的为embedding,带入嵌入的邻接矩阵
#         attention_edge1 = F.softmax(e1, dim=1) 
#         # (e,n) dim=1在节点的维度上进行softmax，大的更大小的更小，和为一
#         # 节点对边注意力

#         xe1_out = torch.mm(attention_edge1, x) 
#         # xe2_out = torch.mm(attention_edge2, xe1)
#         xe2_out = torch.mm(attention_edge2, xe1_out) 
#         # (e2,e1)(e1,200)
        
#         # 超边嵌入(e,200)xe_out= (e,n）·(n,200) = 对节点的注意力·节点嵌入成为超边嵌入，(e,200)
#         attention_node = F.softmax(e1.transpose(0,1), dim=1)
#         # 同上 边对节点注意力 (n,e)
#         x = torch.mm(attention_node, xe1) 
#         # 节点嵌入(n,200)

#         ## attention calculation done 
#         # -----------------------------------------------------------

#         # elu x xe, node
#         if self.concat:# False 
#             x = F.elu(x)
#             xe1_out = F.elu(xe1_out)
#             xe2_out = F.elu(xe2_out)

#         else:
#             x = F.elu(x)
#             xe1_out = F.elu(xe1_out)
#             xe2_out = F.elu(xe2_out)

#         # 对嵌入非线性激活
#         # embedding and edge embeding done

#         if self.coarsen:# 粗糙
#             return x, xe1_out,xe2_out, torch.exp(e1.T) # for what?
#         else:
#             return x, xe1_out,xe2_out# x(n,200);xe1(e1,200);xe2(e2,200)

# class HGAT_sparse(nn.Module):

#     def __init__(self, in_ch_n, out_ch, dropout, alpha, transfer, concat=True, bias=False, coarsen=False):
#         super(HGAT_sparse, self).__init__()
#         self.e_dropout = nn.Dropout(dropout)
#         self.in_ch_n = in_ch_n
#         self.out_ch = out_ch
#         self.alpha = alpha
#         self.concat = concat
#         self.transfer = transfer

#         if self.transfer:
#             self.wt = Parameter(torch.Tensor(self.in_ch_n, self.out_ch))# n,200
#         else:
#             self.register_parameter('wt', None)

#         if bias:
#             self.bias = Parameter(torch.Tensor(1, self.out_ch))
#         else:
#             self.register_parameter('bias', None)
        
#         self.coarsen = coarsen
#         self.reset_parameters()

        
#     def reset_parameters(self): 
#         stdv = 1. / math.sqrt(self.out_ch)
#         if self.wt is not None:
#             self.wt.data.uniform_(-stdv, stdv)
        
#         if self.bias is not None:
#             self.bias.data.uniform_(-stdv, stdv)
        

#     def reset_parameters_xavier(self): #?
#         if self.wt is not None:
#             nn.init.xavier_uniform_(self.wt)

#         if self.bias is not None:
#             nn.init.xavier_uniform_(self.bias) 
        

#     def std_scale(self, x):
#         xstd = x.std(1, unbiased=False, keepdim=True)
#         xstd = torch.where(xstd>0, xstd, torch.tensor(1., device=x.device)) 
#         x = (x - x.mean(1, keepdim=True)) / xstd
#         return x

#     def forward(self, x, xe, pair, a, val=None, e_degs=None, n_degs=None): 
#         if self.transfer:
#             x = x.mm(self.wt ) 
#             # n*n · n*200 = n*200 , 1st：I * wt = wt
#             xe = xe.mm(self.wt) 
#             # e*n · n*200 = e*200 , 1st: H.T_stand * wt, include node connect inf
            
#             if self.bias is not None:
#                 x = x + self.bias
#                 # n*200 + 1*200 ,node+bias
#                 xe = xe + self.bias
#                 # e*200 + 1*200 ,wt*edge+bias
                
# ### ----------------- pair_h -> s(p,g)准备状态---------
                
#         # x = n*200 =wt+b
#         # xe = e*200 =wt·HTs+b
#         n_edge = xe.shape[0] # e_num
#         n_node = x.shape[0] # n_num

#         # pair:# 2 * relation_di(37499), incidence matrix zip
#         # 1st row: pathway indexs ;2nd row: gene indexs - both can duplicated , presents the connection relation of them.
#         if val is None:
#             pair_h = xe[ pair[0] ] * x[ pair[1] ] # 指定行的每一元素对应相乘
#             # pair_h = hypE_emb[pathway_indxs]*node_emb[gene_idx], add the connection inf
#             # pair_h = r*200 X r*200 = r*200【r=pair.shape[1],即节点超边对数】
#         else:
#             pair_h = xe[ pair[0] ] * x[ pair[1] ] * val# val用于pair_h的缩放
        
#         # None ? 
#         if e_degs is not None:
#             pair_h /= e_degs[ pair[0] ].sqrt().unsqueeze(-1) 
#         if n_degs is not None:
#             pair_h /= n_degs[ pair[1] ].sqrt().unsqueeze(-1) 

#         # a,a2: 200*1 
#         pair_e = torch.mm(pair_h, a).squeeze() 
#         # pair_e = r*200 · 200*1 .squeeze = torch.Size([r]) CT context vector？
#         pair_e = self.e_dropout(pair_e) # direct dropout is stable (no nan)
#         # 为什么？
#         base0 = -1e10 # large minus number
#         pair_e[pair_e==0] = base0 # softmax?
        
#         e = base0*torch.ones(n_edge, n_node, device=pair.device)
#         # e(e,n)
        
#         e[pair[0], pair[1]] = pair_e
#         # 将pair_e放回到 e*n matrix,无连接的为负极大值，连接的为embedding,带入嵌入的邻接矩阵

#         attention_edge = F.softmax(e, dim=1) 
#         # (e,n) dim=1在节点的维度上进行softmax，大的更大小的更小，和为一
#         # 节点对边注意力

#         xe_out = torch.mm(attention_edge, x) 
#         # 超边嵌入(e,200)xe_out= (e,n）·(n,200) = 对节点的注意力·节点嵌入成为超边嵌入，(e,200)
#         attention_node = F.softmax(e.transpose(0,1), dim=1)
#         # 同上 边对节点注意力 (n,e)
#         x = torch.mm(attention_node, xe) 
#         # 节点嵌入(n,200)

#         ## attention calculation done 
#         # -----------------------------------------------------------

#         # elu x xe, node
#         if self.concat:# False 
#             x = F.elu(x)
#             xe_out = F.elu(xe_out)
#         else:
#             x = F.elu(x)
#             xe_out = F.elu(xe_out)
#         # 对嵌入非线性激活
#         # embedding and edge embeding done

#         if self.coarsen:# 粗糙
#             return x, xe_out, torch.exp(e.T) # for what?
#         else:
#             return x, xe_out# x(n,200);xe(e,200)

#     def __repr__(self):
#         return self.__class__.__name__ + ' (' + str(self.in_ch_n) + ' -> ' + str(self.out_ch) + ')'
    


# class _MultiLayerHGAT_sparse2(nn.Module):
#     '''
#     ver2 
#     尝试让每一个输出的嵌入包含所有的信息且只有一次传入以此保持对成性和所有嵌入的有效
#     最复杂的结构，但是理论较为对称
#     '''
#     def __init__(self, in_ch_n1,in_ch_n2,in_ch_n3, out_ch, dropout, alpha, transfer, concat=True, bias=False, coarsen=False):
#         super(_MultiLayerHGAT_sparse2, self).__init__()
#         self.e_dropout = nn.Dropout(dropout)
#         self.in_ch_n1 = in_ch_n1
#         self.in_ch_n2 = in_ch_n2
#         self.in_ch_n3 = in_ch_n3
#         self.out_ch = out_ch
#         self.alpha = alpha
#         self.concat = concat
#         self.transfer = transfer

#         if self.transfer:
#             self.wt1 = Parameter(torch.Tensor(self.in_ch_n1, self.out_ch))# n,200
#             self.wt2 = Parameter(torch.Tensor(self.in_ch_n2, self.out_ch))# e1,200
#             self.wt3 = Parameter(torch.Tensor(self.in_ch_n3, self.out_ch))# e2,200
#         else:
#             self.register_parameter('wt1', None)
#             self.register_parameter('wt2', None)
#             self.register_parameter('wt3', None)
#         if bias:
#             self.bias = Parameter(torch.Tensor(1, self.out_ch))
#         else:
#             self.register_parameter('bias', None)
        
#         self.coarsen = coarsen
#         self.reset_parameters()

        
#     def reset_parameters(self): 
#         stdv = 1. / math.sqrt(self.out_ch)
#         if self.wt1 is not None:
#             self.wt1.data.uniform_(-stdv, stdv)
        
#         if self.wt2 is not None:
#             self.wt2.data.uniform_(-stdv, stdv)

#         if self.wt3 is not None:
#             self.wt3.data.uniform_(-stdv, stdv)
        
#         if self.bias is not None:
#             self.bias.data.uniform_(-stdv, stdv)
        

#     def reset_parameters_xavier(self): #?
#         if self.wt1 is not None:
#             nn.init.xavier_uniform_(self.wt1)
#         if self.wt2 is not None:
#             nn.init.xavier_uniform_(self.wt2)
#         if self.wt3 is not None:
#             nn.init.xavier_uniform_(self.wt3)
#         if self.bias is not None:
#             nn.init.xavier_uniform_(self.bias) 
        

#     def std_scale(self, x):
#         xstd = x.std(1, unbiased=False, keepdim=True)
#         xstd = torch.where(xstd>0, xstd, torch.tensor(1., device=x.device)) 
#         x = (x - x.mean(1, keepdim=True)) / xstd
#         return x

#     def forward(self, x, xe1, xe2, xe3, pair01, pair12,pair23, a0,a1,a2,a3,a4,a5,a6 ,val=None,e3_degs=None,e2_degs=None,e1_degs=None, n_degs=None): 
        
       
       
#         if self.transfer:
#             x = x.mm(self.wt1) 
#             xe1 = xe1.mm(self.wt1) 
#             xe2 = xe2.mm(self.wt2) 
#             xe3 = xe3.mm(self.wt3)
#             # xe2 = xe2.mm(self.wt1) 
#             # xe3 = xe3.mm(self.wt1)
#             if self.bias is not None:
#                 x = x + self.bias
#                 xe1 = xe1 + self.bias
#                 xe2 = xe2 + self.bias
#                 xe3 = xe3 + self.bias
#         # x(n,200); xe1(e1,200); xe2(e2,200); xe3(e3,200)
#         n_edge3 = xe3.shape[0]
#         n_edge2 = xe2.shape[0]
#         n_edge1 = xe1.shape[0] 
#         n_node  = x.shape[0] 
#         ## 1 attention10/01 = xnxe1; attention32/23 = xe3xe2
#         # 暂时去掉val以及度的计算
#         base0 = -1e10

#         pair23_h = xe3[pair23[0]]*xe2[pair23[1]]
#         pair_e32 = torch.mm(pair23_h,a0).squeeze()
#         pair_e32 = self.e_dropout(pair_e32)
#         pair_e32[pair_e32==0] = base0
#         e32 = base0*torch.ones(n_edge3, n_edge2, device=pair23.device)
#         e32[pair23[0],pair23[1]] = pair_e32
#         attention3_2 = F.softmax(e32,dim=1)# 3,2
#         attention2_3 = F.softmax(e32.transpose(0,1), dim=1)# 2,3
        
#         pair01_h = xe1[pair01[0]]*x[pair01[1]]
#         pair_e10 = torch.mm(pair01_h,a1).squeeze()
#         pair_e10 = self.e_dropout(pair_e10)
#         pair_e10[pair_e10==0] = base0
#         e10 = base0*torch.ones(n_edge1, n_node, device=pair01.device)
#         e10[pair01[0],pair01[1]] = pair_e10
#         attention1_0 = F.softmax(e10,dim=1)
#         attention0_1 = F.softmax(e10.transpose(0,1), dim=1)

#         xe23 = torch.mm(attention2_3, xe3)
#         xe1n = torch.mm(attention1_0, x)

#         ## attention23_1 = xe23 xe1 ; att21_10 = xe23 xe1n ; atten10_2 = xe1n.xe2
#         pair12_h = xe23[pair12[0]]*xe1[pair12[1]]
#         pair_23_1 = torch.mm(pair12_h,a2).squeeze()
#         pair_23_1 = self.e_dropout(pair_23_1)
#         pair_23_1[pair_23_1==0] = base0
#         e23_1 = base0*torch.ones(n_edge2, n_edge1, device=pair12.device)
#         e23_1[pair12[0],pair12[1]] = pair_23_1
#         attention23_1 = F.softmax(e23_1,dim=1)
#         attention1_23 = F.softmax(e23_1.transpose(0,1), dim=1)

#         pair12_h = xe23[pair12[0]]*xe1n[pair12[1]]
#         pair_32_10 = torch.mm(pair12_h,a3).squeeze()
#         pair_32_10 = self.e_dropout(pair_32_10)
#         pair_32_10[pair_32_10==0] = base0
#         e23_10 = base0*torch.ones(n_edge2, n_edge1, device=pair12.device)
#         e23_10[pair12[0],pair12[1]] = pair_32_10
#         attention23_10 = F.softmax(e23_10,dim=1)
#         attention10_23 = F.softmax(e23_10.transpose(0,1), dim=1)

#         pair01_h = xe2[pair12[0]]*xe1n[pair12[1]]
#         pair_2_10 = torch.mm(pair01_h,a4).squeeze()
#         pair_2_10 = self.e_dropout(pair_2_10)
#         pair_2_10[pair_2_10==0] = base0
#         e2_10 = base0*torch.ones(n_edge2, n_edge1, device=pair01.device)
#         e2_10[pair12[0],pair12[1]] = pair_2_10
#         attention2_10 = F.softmax(e2_10,dim=1)
#         attention10_2 = F.softmax(e2_10.transpose(0,1), dim=1)

#         xe123  = torch.mm(attention1_23, xe23)
#         xe21n  = torch.mm(attention2_10, xe1n)
#         xe231n = torch.mm(attention23_10, xe1n)

#         ## a123_0 a3_210
#         pair01_h = xe123[pair01[0]]*x[pair01[1]]
#         pair_123_0 = torch.mm(pair01_h,a5).squeeze()
#         pair_123_0 = self.e_dropout(pair_123_0)
#         pair_123_0[pair_123_0==0] = base0
#         e123_0 = base0*torch.ones(n_edge1, n_node, device=pair01.device)
#         e123_0[pair01[0],pair01[1]] = pair_123_0
#         attention123_0 = F.softmax(e123_0,dim=1)# 3,2
#         attention0_123 = F.softmax(e123_0.transpose(0,1), dim=1)

#         pair23_h = xe3[pair23[0]]*xe21n[pair23[1]]
#         pair_3_210 = torch.mm(pair23_h,a6).squeeze()
#         pair_3_210 = self.e_dropout(pair_3_210)
#         pair_3_210[pair_3_210==0] = base0
#         e3_210 = base0*torch.ones(n_edge3, n_edge2, device=pair23.device)
#         e3_210[pair23[0],pair23[1]] = pair_3_210
#         attention3_210 = F.softmax(e3_210,dim=1)
#         attention210_3 = F.softmax(e3_210.transpose(0,1), dim=1)

#         xne123 = torch.mm(attention0_123, xe123)
#         xe123n = torch.mm(attention123_0, x)
#         xe321n = torch.mm(attention3_210, xe21n)
        
#         x = xne123
#         xe1_out = xe123n
#         xe2_out = xe231n
#         xe3_out = xe321n

#         if self.concat:
#             x = F.elu(x)
#             xe1_out = F.elu(xe1_out)
#             xe2_out = F.elu(xe2_out)
#             xe3_out = F.elu(xe3_out)
#         else:
#             x = F.elu(x)
#             xe1_out = F.elu(xe1_out)
#             xe2_out = F.elu(xe2_out)
#             xe3_out = F.elu(xe3_out)


#         return x, xe1_out, xe2_out, xe3_out# x(n,200);xe(e,200)

#     def __repr__(self):
#         return self.__class__.__name__ + ' (' + str(self.in_ch_n) + ' -> ' + str(self.out_ch) + ')'

# class _MultiLayerHGAT_sparse1(nn.Module):
#     '''
#     MultiLayerHGAT_sparse ver1
    
#     xe3(k-1)->xe2->xe1->x_out;其中的attention计算为新生成的嵌入和k-1层嵌入一同计算；超边们的嵌入输出为节点流所对称的输出。
    
#     在初始化版本0下：
#     此版本的日志文件: logs/HyperParamExperiments/240305__MultiLayerHGAT_sparse1_computelayers1-4_iniV0.log
#     结果文件路径：Results/HSHINEprogExperiments/_MultiLayerHGAT_sparse1-iniv0-compute_layers
#     test_ci:0.6445599437496733±0.04687963368178867
#     效果差，ci降低0.1。

#     在初始化版本1下：logs/HyperParamExperiments/240305__MultiLayerHGAT_sparse1_computelayers1-4_iniV1.log
#     结果文件路径：Results/HSHINEprogExperiments/_MultiLayerHGAT_sparse1-iniv1-compute_layers
#     通过从第三层超边传递到节点，同时对称的生成输出对应的嵌入,理论上可以进行多次计算层传递
#     比0版本好，但是仍比原版差
#     '''
#     def __init__(self, in_ch_n1,in_ch_n2,in_ch_n3, out_ch, dropout, alpha, transfer, concat=True, bias=False, coarsen=False):
#         super(_MultiLayerHGAT_sparse1, self).__init__()
#         self.e_dropout = nn.Dropout(dropout)
#         self.in_ch_n1 = in_ch_n1
#         self.in_ch_n2 = in_ch_n2
#         self.in_ch_n3 = in_ch_n3
#         self.out_ch = out_ch
#         self.alpha = alpha
#         self.concat = concat
#         self.transfer = transfer

#         if self.transfer:
#             self.wt1 = Parameter(torch.Tensor(self.in_ch_n1, self.out_ch))# n,200
#             self.wt2 = Parameter(torch.Tensor(self.in_ch_n2, self.out_ch))# e1,200
#             self.wt3 = Parameter(torch.Tensor(self.in_ch_n3, self.out_ch))# e2,200
#         else:
#             self.register_parameter('wt1', None)
#             self.register_parameter('wt2', None)
#             self.register_parameter('wt3', None)
#         if bias:
#             self.bias = Parameter(torch.Tensor(1, self.out_ch))
#         else:
#             self.register_parameter('bias', None)
        
#         self.coarsen = coarsen
#         self.reset_parameters()

        
#     def reset_parameters(self): 
#         stdv = 1. / math.sqrt(self.out_ch)
#         if self.wt1 is not None:
#             self.wt1.data.uniform_(-stdv, stdv)
        
#         if self.wt2 is not None:
#             self.wt2.data.uniform_(-stdv, stdv)

#         if self.wt3 is not None:
#             self.wt3.data.uniform_(-stdv, stdv)
        
#         if self.bias is not None:
#             self.bias.data.uniform_(-stdv, stdv)
        

#     def reset_parameters_xavier(self): #?
#         if self.wt1 is not None:
#             nn.init.xavier_uniform_(self.wt1)
#         if self.wt2 is not None:
#             nn.init.xavier_uniform_(self.wt2)
#         if self.wt3 is not None:
#             nn.init.xavier_uniform_(self.wt3)
#         if self.bias is not None:
#             nn.init.xavier_uniform_(self.bias) 
        

#     def std_scale(self, x):
#         xstd = x.std(1, unbiased=False, keepdim=True)
#         xstd = torch.where(xstd>0, xstd, torch.tensor(1., device=x.device)) 
#         x = (x - x.mean(1, keepdim=True)) / xstd
#         return x

#     def forward(self, x, xe1, xe2, xe3, pair, pairE1,pairE2, a3,a2,a1, val=None,e3_degs=None,e2_degs=None,e1_degs=None, n_degs=None): 
#         if self.transfer:
#             x = x.mm(self.wt1) 
#             xe1 = xe1.mm(self.wt1) 
#             xe2 = xe2.mm(self.wt2) 
#             xe3 = xe3.mm(self.wt3)
#             # xe2 = xe2.mm(self.wt1) 
#             # xe3 = xe3.mm(self.wt1)
#             if self.bias is not None:
#                 x = x + self.bias
#                 xe1 = xe1 + self.bias
#                 xe2 = xe2 + self.bias
#                 xe3 = xe3 + self.bias
#         # x(n,200); xe1(e1,200); xe2(e2,200); xe3(e3,200)
#         n_edge3 = xe3.shape[0]
#         n_edge2 = xe2.shape[0]
#         n_edge1 = xe1.shape[0] 
#         n_node  = x.shape[0] 
# #-----------------由上一计算层的超边3和超边2计算新的此层超边3和超边2-------------------------
#         #-------计算e3e2 pair embedding->e3(e3,e2)-------
#         if val is None:
#             pairE2_h = xe3[ pairE2[0] ] * xe2[ pairE2[1] ]
#         else:
#             pairE2_h = xe3[ pairE2[0] ] * xe2[ pairE2[1] ] * val
#         if e3_degs is not None:
#             pairE2_h /= e3_degs[ pairE2[0] ].sqrt().unsqueeze(-1) 
#         if e2_degs is not None:
#             pairE2_h /= e2_degs[ pairE2[1] ].sqrt().unsqueeze(-1) 
#         pair_e3 = torch.mm(pairE2_h, a3).squeeze() 
#         pair_e3 = self.e_dropout(pair_e3)
#         base0 = -1e10
#         pair_e3[pair_e3==0] = base0 
#         e3 = base0*torch.ones(n_edge3, n_edge2, device=pairE2.device)
#         e3[pairE2[0], pairE2[1]] = pair_e3
#         #------计算32之间的相互注意力，并计算嵌入-------
#         attention_e3toe2 = F.softmax(e3, dim=1) 
#         xe3_out = torch.mm(attention_e3toe2, xe2) 
#         # (e3,e2)·(e2,200) = xe3(e3,200)
#         attention_e2toe3 = F.softmax(e3.transpose(0,1), dim=1) 
#         xe2 = torch.mm(attention_e2toe3, xe3) 
#         # (e2,e3)·(e3,200) = xe2(e2,200)
# #-----------------由上一计算层的超边1嵌入和刚得到的xe2嵌入计算新的此层超边2和超边1嵌入-------------------------
#         #-------计算e2e1 pair embedding->e2(e2,e1)-------
#         if val is None:
#             pairE1_h = xe2[ pairE1[0] ] * xe1[ pairE1[1] ]
#         else:
#             pairE1_h = xe2[ pairE1[0] ] * xe1[ pairE1[1] ] * val
#         if e2_degs is not None:
#             pairE1_h /= e2_degs[ pairE1[0] ].sqrt().unsqueeze(-1) 
#         if e1_degs is not None:
#             pairE1_h /= e1_degs[ pairE1[1] ].sqrt().unsqueeze(-1) 
#         pair_e2 = torch.mm(pairE1_h, a2).squeeze() 
#         pair_e2 = self.e_dropout(pair_e2)
#         base0 = -1e10
#         pair_e2[pair_e2==0] = base0 
#         e2 = base0*torch.ones(n_edge2, n_edge1, device=pairE1.device)
#         e2[pairE1[0], pairE1[1]] = pair_e2
#         #------计算21之间的相互注意力，并计算嵌入-------
#         attention_e2toe1 = F.softmax(e2, dim=1) 
#         xe2_out = torch.mm(attention_e2toe1, xe1) 
#         # (e2,e1)·(e1,200) = xe2(e2,200)
#         attention_e1toe2 = F.softmax(e2.transpose(0,1), dim=1) 
#         xe1 = torch.mm(attention_e1toe2, xe2) 
#         # (e1,e2)·(e2,200) = xe1(e1,200)
# #-----------------由上一计算层节点嵌入和刚得到的xe1嵌入计算新的此层超边1和节点嵌入-------------------------
#         #-------计算e1node pair embedding->e1(e1,n)-------
#         if val is None:
#             pair_h = xe1[ pair[0] ] * x[ pair[1] ]
#         else:
#             pair_h = xe1[ pair[0] ] * x[ pair[1] ] * val
#         if e1_degs is not None:
#             pair_h /= e1_degs[ pair[0] ].sqrt().unsqueeze(-1) 
#         if n_degs is not None:
#             pair_h /= n_degs[ pair[1] ].sqrt().unsqueeze(-1) 
#         pair_e1 = torch.mm(pair_h, a1).squeeze() 
#         pair_e1 = self.e_dropout(pair_e1)
#         base0 = -1e10
#         pair_e1[pair_e1==0] = base0 
#         e1 = base0*torch.ones(n_edge1, n_node, device=pair.device)
#         e1[pair[0], pair[1]] = pair_e1
#         #------计算21之间的相互注意力，并计算嵌入-------
#         attention_e1tono = F.softmax(e1, dim=1) 
#         xe1_out = torch.mm(attention_e1tono, x) 
#         # (e1,n)·(n,200) = xe1(e1,200)
#         attention_notoe1 = F.softmax(e1.transpose(0,1), dim=1) 
#         x = torch.mm(attention_notoe1, xe1) 
#         # (n,e1)·(e1,200) = x(n,200)
        
# #-----------------non-linear activation for embedding--------------------------------------------------
#         # elu x xe
#         if self.concat:# False 
#             x = F.elu(x)
#             xe1_out = F.elu(xe1_out)
#             xe2_out = F.elu(xe2_out)
#             xe3_out = F.elu(xe3_out)
#         else:
#             x = F.elu(x)
#             xe1_out = F.elu(xe1_out)
#             xe2_out = F.elu(xe2_out)
#             xe3_out = F.elu(xe3_out)

#         # embedding done
#         if self.coarsen:# 粗糙  
#             return x, xe1_out, xe2_out, xe3_out, torch.exp(e1.T) # for what?
#         else:
#             return x, xe1_out, xe2_out, xe3_out# x(n,200);xe(e,200)

#     def __repr__(self):
#         return self.__class__.__name__ + ' (' + str(self.in_ch_n) + ' -> ' + str(self.out_ch) + ')'

# class _MultiLayerHGAT_sparse0(nn.Module):
#     '''
#     MultiLayerHGAT_sparse ver0
    
#     全传递版本，首先通过上一计算层的x xe1 xe2 xe3生成他们之间的注意力: attention_e1tono、attention_e2toe1、attention_e3toe2、attention_e2toe3、attention_e1toe2、attention_node
#     接着通过对应的注意力，使用上一计算层的x计算出xe1，在用刚计算出的xe1计算出xe2，以此类推，即传递流为x(k-1)->xe1->xe2->xe3_out->xe2_out->xe1_out->x_out
    
#     在初始化版本0下：
#     日志文件: logs/HyperParamExperiments/240305__MultiLayerHGAT_sparse0_computelayers1-4_iniV0.log
#     结果文件路径：Results/HSHINEprogExperiments/_MultiLayerHGAT_sparse0-iniv0-compute_layers
#     test_ci:0.6445599437496733±0.04687963368178867
#     效果差，ci降低0.1。

#     在初始化版本1下：
#     日志文件：logs/HyperParamExperiments/240305__MultiLayerHGAT_sparse0_computelayers1-4_iniV1.log
#     结果文件路径：Results/HSHINEprogExperiments/_MultiLayerHGAT_sparse0-iniv1-compute_layers

#     尝试分别上下文变量a；分为a0,a1,a2

#     '''
#     def __init__(self, in_ch_n1,in_ch_n2,in_ch_n3, out_ch, dropout, alpha, transfer, concat=True, bias=False, coarsen=False):
#         super(_MultiLayerHGAT_sparse0, self).__init__()
#         self.e_dropout = nn.Dropout(dropout)
#         self.in_ch_n1 = in_ch_n1
#         self.in_ch_n2 = in_ch_n2
#         self.in_ch_n3 = in_ch_n3
#         self.out_ch = out_ch
#         self.alpha = alpha
#         self.concat = concat
#         self.transfer = transfer

#         if self.transfer:
#             self.wt1 = Parameter(torch.Tensor(self.in_ch_n1, self.out_ch))# n,200
#             self.wt2 = Parameter(torch.Tensor(self.in_ch_n2, self.out_ch))# e1,200
#             self.wt3 = Parameter(torch.Tensor(self.in_ch_n3, self.out_ch))# e2,200
#         else:
#             self.register_parameter('wt1', None)
#             self.register_parameter('wt2', None)
#             self.register_parameter('wt3', None)
#         if bias:
#             self.bias = Parameter(torch.Tensor(1, self.out_ch))
#         else:
#             self.register_parameter('bias', None)
        
#         self.coarsen = coarsen
#         self.reset_parameters()

        
#     def reset_parameters(self): 
#         stdv = 1. / math.sqrt(self.out_ch)
#         if self.wt1 is not None:
#             self.wt1.data.uniform_(-stdv, stdv)
        
#         if self.wt2 is not None:
#             self.wt2.data.uniform_(-stdv, stdv)

#         if self.wt3 is not None:
#             self.wt3.data.uniform_(-stdv, stdv)
        
#         if self.bias is not None:
#             self.bias.data.uniform_(-stdv, stdv)
        

#     def reset_parameters_xavier(self): #?
#         if self.wt1 is not None:
#             nn.init.xavier_uniform_(self.wt1)
#         if self.wt2 is not None:
#             nn.init.xavier_uniform_(self.wt2)
#         if self.wt3 is not None:
#             nn.init.xavier_uniform_(self.wt3)
#         if self.bias is not None:
#             nn.init.xavier_uniform_(self.bias) 
        

#     def std_scale(self, x):
#         xstd = x.std(1, unbiased=False, keepdim=True)
#         xstd = torch.where(xstd>0, xstd, torch.tensor(1., device=x.device)) 
#         x = (x - x.mean(1, keepdim=True)) / xstd
#         return x

#     def forward(self, x, xe1, xe2, xe3, pair, pairE1,pairE2, a0,a1,a2, val=None, e_degs=None, n_degs=None): 
#         if self.transfer:
#             x = x.mm(self.wt1) 
#             # n*n · n*200 = n*200 , 1st：I * wt = wt
#             xe1 = xe1.mm(self.wt1) 
#             # xe2 = xe2.mm(self.wt1) 
#             # xe3 = xe3.mm(self.wt1) 

#             # e*n · n*200 = e*200 , 1st: H.T_stand * wt, include node connect inf
#             xe2 = xe2.mm(self.wt2) 
#             # # xe2 = H2.T_norm(e2,e1)@efts(e1,e1)=(e2,e1)
#             # # (e2,e1)·(e1,200) = (e2,200)
#             xe3 = xe3.mm(self.wt3)


#             if self.bias is not None:
#                 x = x + self.bias
#                 # n*200 + 1*200 ,node+bias
#                 xe1 = xe1 + self.bias
#                 # e*200 + 1*200 ,wt*edge+bias
#                 xe2 = xe2 + self.bias
#                 # e2,200 + 1,200
#                 xe3 = xe3 + self.bias

#         n_edge3 = xe3.shape[0]
#         n_edge2 = xe2.shape[0]
#         n_edge1 = xe1.shape[0] # e_num
#         n_node  = x.shape[0] # n_num

# ### --------------------计算准备状态s(p,g)pair_h,pair_E1,pair_E2----------------------------
#         # pair:# 2 * relation_di(37499), incidence matrix zip
#         # 1st row: pathway indexs ;2nd row: gene indexs - both can duplicated , presents the connection relation of them.
#         if val is None:
#             pair_h = xe1[ pair[0] ] * x[ pair[1] ] # 指定行的每一元素对应相乘
#             # pair_h = hypE_emb[pathway_indxs]*node_emb[gene_idx], add the connection inf
#             # pair_h = r*200 X r*200 = r*200【r=pair.shape[1],即节点超边对数】
#             pairE1_h = xe2[ pairE1[0] ] * xe1[ pairE1[1] ]
#             # r2,200 两层超边对数
#             pairE2_h = xe3[ pairE2[0] ] * xe2[ pairE2[1] ]
#         else:
#             pair_h = xe1[ pair[0] ] * x[ pair[1] ] * val# val用于pair_h的缩放
#             pairE1_h = xe2[ pairE1[0] ] * xe1[ pairE1[1] ] * val
#             pairE2_h = xe3[ pairE2[0] ] * xe2[ pairE2[1] ] * val

#         # None ???
#         if e_degs is not None:
#             pair_h /= e_degs[ pair[0] ].sqrt().unsqueeze(-1) 
#         if n_degs is not None:
#             pair_h /= n_degs[ pair[1] ].sqrt().unsqueeze(-1) 

#         # 尝试相同的上下文变量或者不同上下文变量开关
#         use_same_a = False
#         if use_same_a:
#             a1 = a0
#             a2 = a0
        
#         # a: (200*1) 
#         pair_e1 = torch.mm(pair_h, a0).squeeze() # 超边1节点对数r1;
#         # pair_e1 = r1*200 · 200*1 .squeeze = torch.Size([r1]) CT context vector？
#         pair_e1 = self.e_dropout(pair_e1) # direct dropout is stable (no nan)
#         # 为什么？
#         base0 = -1e10 # large minus number
#         pair_e1[pair_e1==0] = base0 # 不关注赋予负的大值？防止梯度消失？
#         e1 = base0*torch.ones(n_edge1, n_node, device=pair.device)
#         # e(e,n)
#         e1[pair[0], pair[1]] = pair_e1
#         # 将pair_e放回到 e*n matrix,无连接的为负极大值，连接的为embedding,带入嵌入的邻接矩阵

#         pair_e2 = torch.mm(pairE1_h, a1).squeeze() #超边2 超边1 关联对数
#         # pair_e = r2*200 · 200*1 .squeeze = torch.Size([r2]) 
#         pair_e2 = self.e_dropout(pair_e2)
#         base0 = -1e10
#         pair_e2[pair_e2==0] = base0 
#         e2 = base0*torch.ones(n_edge2, n_edge1, device=pairE1.device)
#         # e2(e2,e1)
#         e2[pairE1[0], pairE1[1]] = pair_e2
#         # (e2,e1)

#         pair_e3 = torch.mm(pairE2_h, a2).squeeze() # 超边3 超边2 关联对数
#         # 是否应该使用不同的a？
#         pair_e3 = self.e_dropout(pair_e3)
#         base0 = -1e10
#         pair_e3[pair_e3==0] = base0 
#         e3 = base0*torch.ones(n_edge3, n_edge2, device=pairE2.device)# e3,e2
#         e3[pairE2[0], pairE2[1]] = pair_e3

#         attention_e1tono = F.softmax(e1, dim=1) 
#         attention_e2toe1 = F.softmax(e2, dim=1) 
#         attention_e3toe2 = F.softmax(e3, dim=1)  
#         attention_e2toe3 = F.softmax(e3.transpose(0,1), dim=1) 
#         attention_e1toe2 = F.softmax(e2.transpose(0,1), dim=1) 
#         attention_node   = F.softmax(e1.transpose(0,1), dim=1)

#         xe1 = torch.mm(attention_e1tono, x) 
#         xe2 = torch.mm(attention_e2toe1, xe1) 
#         xe3_out = torch.mm(attention_e3toe2, xe2) 
#         xe2_out = torch.mm(attention_e2toe3, xe3_out) 
#         xe1_out = torch.mm(attention_e1toe2, xe2_out) 
#         x = torch.mm(attention_node, xe1_out) 
        
#         if self.concat:# False 
#             x = F.elu(x)
#             xe1_out = F.elu(xe1_out)
#             xe2_out = F.elu(xe2_out)
#             xe3_out = F.elu(xe3_out)
#         else:
#             x = F.elu(x)
#             xe1_out = F.elu(xe1_out)
#             xe2_out = F.elu(xe2_out)
#             xe3_out = F.elu(xe3_out)

#         if self.coarsen:# 粗糙  
#             return x, xe1_out, xe2_out, xe3_out, torch.exp(e1.T) # for what?
#         else:
#             return x, xe1_out, xe2_out, xe3_out# x(n,200);xe(e,200)

#     def __repr__(self):
#         return self.__class__.__name__ + ' (' + str(self.in_ch_n) + ' -> ' + str(self.out_ch) + ')'

# class HHGAT_sparse(nn.Module):
#     def __init__(self, in_ch_n, out_ch, dropout, alpha, transfer, concat=True, bias=False, coarsen=False):
#         super(HHGAT_sparse, self).__init__()
#         self.e_dropout = nn.Dropout(dropout)
#         self.in_ch_n = in_ch_n
#         self.out_ch = out_ch
#         self.alpha = alpha
#         self.concat = concat
#         self.transfer = transfer

#         if self.transfer:
#             self.wt = Parameter(torch.Tensor(self.in_ch_n, self.out_ch))
#         else:
#             self.register_parameter('wt', None)

#         if bias:
#             self.bias = Parameter(torch.Tensor(1, self.out_ch))
#         else:
#             self.register_parameter('bias', None)
        
#         self.coarsen = coarsen
#         self.reset_parameters()

        
#     def reset_parameters(self): 
#         stdv = 1. / math.sqrt(self.out_ch)
#         if self.wt is not None:
#             self.wt.data.uniform_(-stdv, stdv)
        
#         if self.bias is not None:
#             self.bias.data.uniform_(-stdv, stdv)
        

#     def reset_parameters_xavier(self): 
#         if self.wt is not None:
#             nn.init.xavier_uniform_(self.wt)

#         if self.bias is not None:
#             nn.init.xavier_uniform_(self.bias) 
        

#     def std_scale(self, x):
#         xstd = x.std(1, unbiased=False, keepdim=True)
#         xstd = torch.where(xstd>0, xstd, torch.tensor(1., device=x.device)) 
#         x = (x - x.mean(1, keepdim=True)) / xstd
#         return x

#     def forward(self, x, xe, sgs, pair, a, val=None, e_degs=None, n_degs=None): 
#         if self.transfer:
#             x = x.mm(self.wt) 
#             xe = xe.mm(self.wt) 
            
#             if self.bias is not None:
#                 x = x + self.bias
#                 xe = xe + self.bias

#         n_edge = xe.shape[0] 
#         n_node = x.shape[0] 

#         if val is None:
#             pair_h = xe[ pair[0] ] * x[ pair[1] ] 
#         else:
#             pair_h = xe[ pair[0] ] * x[ pair[1] ] * val
            
#         if e_degs is not None:
#             pair_h /= e_degs[ pair[0] ].sqrt().unsqueeze(-1) 
#         if n_degs is not None:
#             pair_h /= n_degs[ pair[1] ].sqrt().unsqueeze(-1) 
#         pair_e = torch.mm(pair_h, a).squeeze() 
        
#         pair_e = self.e_dropout(pair_e) # direct dropout is stable (no nan)
#         base0 = -1e10
#         pair_e[pair_e==0] = base0
        
#         e = base0*torch.ones(n_edge, n_node, device=pair.device)
#         e[pair[0], pair[1]] = pair_e


#         attention_edge = F.softmax(e, dim=1) 

#         xe_out = torch.mm(attention_edge, x) 
        
#         attention_node = F.softmax(e.transpose(0,1), dim=1)
#         xsg = torch.mm(sgs, attention_node)
#         x = torch.mm(attention_node, xe) 


#         if self.concat:
#             x = F.elu(x)
#             xe_out = F.elu(xe_out)
#         else:
#             x = F.elu(x)
#             xe_out = F.elu(xe_out)

        
#         if self.coarsen:
#             return x, xe_out, xsg, torch.exp(e.T) 
#         else:
#             return x, xe_out, xsg

#     def __repr__(self):
#         return self.__class__.__name__ + ' (' + str(self.in_ch_n) + ' -> ' + str(self.out_ch) + ')'

# class HyperGraphConvolution(nn.Module):
#     """
#     Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
#     """

#     def __init__(self, a, b, reapproximate=True, cuda=True):
        
#         super(HyperGraphConvolution, self).__init__()
#         self.a, self.b = a, b
#         self.reapproximate, self.cuda = reapproximate, cuda

#         self.W = Parameter(torch.FloatTensor(a, b))
#         self.bias = Parameter(torch.FloatTensor(b))
#         self.reset_parameters()
        


#     def reset_parameters(self):
#         std = 1. / math.sqrt(self.W.size(1))
#         self.W.data.uniform_(-std, std)
#         self.bias.data.uniform_(-std, std)



#     def forward(self, structure, H, m=True):
#         W, b = self.W, self.bias
#         HW = torch.mm(H, W)

#         if self.reapproximate:
#             n, X = H.shape[0], HW.cpu().detach().numpy()
#             A = Laplacian(n, structure, X, m)
#         else: A = structure

#         if self.cuda: A = A.cuda()
#         A = Variable(A)

#         AHW = SparseMM.apply(A, HW)     #torch.Size([1438, 150])
#         return AHW + b



#     def __repr__(self):
#         return self.__class__.__name__ + ' (' \
#                + str(self.a) + ' -> ' \
#                + str(self.b) + ')'



# class SparseMM(torch.autograd.Function):
#     """
#     Sparse x dense matrix multiplication with autograd support.
#     Implementation by Soumith Chintala:
#     https://discuss.pytorch.org/t/
#     does-pytorch-support-autograd-on-sparse-matrix/6156/7
#     """
#     @staticmethod
#     def forward(ctx, M1, M2):
#         ctx.save_for_backward(M1, M2)
#         return torch.mm(M1, M2)

#     @staticmethod
#     def backward(ctx, g):
#         M1, M2 = ctx.saved_tensors
#         g1 = g2 = None

#         if ctx.needs_input_grad[0]:
#             g1 = torch.mm(g, M2.t())

#         if ctx.needs_input_grad[1]:
#             g2 = torch.mm(M1.t(), g)

#         return g1, g2



# def Laplacian(V, E, X, m):
#     """
#     approximates the E defined by the E Laplacian with/without mediators

#     arguments:
#     V: number of vertices
#     E: dictionary of hyperedges (key: hyperedge, value: list/set of hypernodes)
#     X: features on the vertices
#     m: True gives Laplacian with mediators, while False gives without

#     A: adjacency matrix of the graph approximation
#     returns: 
#     updated data with 'graph' as a key and its value the approximated hypergraph 
#     """
    
#     edges, weights = [], {}
#     rv = np.random.rand(X.shape[1])

#     for k in E.keys():
#         hyperedge = list(E[k])
        
#         p = np.dot(X[hyperedge], rv)   
#         s, i = np.argmax(p), np.argmin(p)
#         Se, Ie = hyperedge[s], hyperedge[i]

        
#         c = 2*len(hyperedge) - 3    
#         if m:
            
            
#             edges.extend([[Se, Ie], [Ie, Se]])
            
#             if (Se,Ie) not in weights:
#                 weights[(Se,Ie)] = 0
#             weights[(Se,Ie)] += float(1/c)

#             if (Ie,Se) not in weights:
#                 weights[(Ie,Se)] = 0
#             weights[(Ie,Se)] += float(1/c)
            
            
#             for mediator in hyperedge:
#                 if mediator != Se and mediator != Ie:
#                     edges.extend([[Se,mediator], [Ie,mediator], [mediator,Se], [mediator,Ie]])
#                     weights = update(Se, Ie, mediator, weights, c)
#         else:
#             edges.extend([[Se,Ie], [Ie,Se]])
#             e = len(hyperedge)
            
#             if (Se,Ie) not in weights:
#                 weights[(Se,Ie)] = 0
#             weights[(Se,Ie)] += float(1/e)

#             if (Ie,Se) not in weights:
#                 weights[(Ie,Se)] = 0
#             weights[(Ie,Se)] += float(1/e)    
    
#     return adjacency(edges, weights, V)



# def update(Se, Ie, mediator, weights, c):
#     """
#     updates the weight on {Se,mediator} and {Ie,mediator}
#     """    
    
#     if (Se,mediator) not in weights:
#         weights[(Se,mediator)] = 0
#     weights[(Se,mediator)] += float(1/c)

#     if (Ie,mediator) not in weights:
#         weights[(Ie,mediator)] = 0
#     weights[(Ie,mediator)] += float(1/c)

#     if (mediator,Se) not in weights:
#         weights[(mediator,Se)] = 0
#     weights[(mediator,Se)] += float(1/c)

#     if (mediator,Ie) not in weights:
#         weights[(mediator,Ie)] = 0
#     weights[(mediator,Ie)] += float(1/c)

#     return weights



# def adjacency(edges, weights, n):
#     """
#     computes an sparse adjacency matrix

#     arguments:
#     edges: list of pairs
#     weights: dictionary of edge weights (key: tuple representing edge, value: weight on the edge)
#     n: number of nodes

#     returns: a scipy.sparse adjacency matrix with unit weight self loops for edges with the given weights
#     """
    
#     dictionary = {tuple(item): index for index, item in enumerate(edges)}
#     edges = [list(itm) for itm in dictionary.keys()]   
#     organised = []

#     for e in edges:
#         i,j = e[0],e[1]
#         w = weights[(i,j)]
#         organised.append(w)

#     edges, weights = np.array(edges), np.array(organised)
#     adj = sp.coo_matrix((weights, (edges[:, 0], edges[:, 1])), shape=(n, n), dtype=np.float32)
#     adj = adj + sp.eye(n)

#     A = symnormalise(sp.csr_matrix(adj, dtype=np.float32))
#     A = ssm2tst(A)
#     return A



# def symnormalise(M):
#     """
#     symmetrically normalise sparse matrix

#     arguments:
#     M: scipy sparse matrix

#     returns:
#     D^{-1/2} M D^{-1/2} 
#     where D is the diagonal node-degree matrix
#     """
    
#     d = np.array(M.sum(1))
    
#     dhi = np.power(d, -1/2).flatten()
#     dhi[np.isinf(dhi)] = 0.
#     DHI = sp.diags(dhi)    
    
#     return (DHI.dot(M)).dot(DHI) 



# def ssm2tst(M):
#     """
#     converts a scipy sparse matrix (ssm) to a torch sparse tensor (tst)

#     arguments:
#     M: scipy sparse matrix

#     returns:
#     a torch sparse tensor of M
#     """
    
#     M = M.tocoo().astype(np.float32)
    
#     indices = torch.from_numpy(np.vstack((M.row, M.col))).long()
#     values = torch.from_numpy(M.data)
#     shape = torch.Size(M.shape)
    
#     return torch.sparse.FloatTensor(indices, values, shape)



# class GraphConvolution(nn.Module):
#     '''
#     GCN torch from https://github.com/dragen1860/GCN-PyTorch/blob/master/layer.py
#     '''

#     def __init__(self, input_dim, output_dim, num_features_nonzero,
#                  dropout=0.,
#                  is_sparse_inputs=False,
#                  bias=False,
#                  activation = F.relu,
#                  featureless=False):
#         super(GraphConvolution, self).__init__()


#         self.dropout = dropout
#         self.bias = bias
#         self.activation = activation
#         self.is_sparse_inputs = is_sparse_inputs
#         self.featureless = featureless
#         self.num_features_nonzero = num_features_nonzero

#         self.weight = nn.Parameter(torch.randn(input_dim, output_dim))
#         self.bias = None
#         if bias:
#             self.bias = nn.Parameter(torch.zeros(output_dim))


#     def forward(self, inputs):
#         # print('inputs:', inputs)
#         x, support = inputs

#         if self.training and self.is_sparse_inputs:
#             x = self.sparse_dropout(x, self.dropout, self.num_features_nonzero)
#         elif self.training:
#             x = F.dropout(x, self.dropout)

#         # convolve
#         if not self.featureless: # if it has features x
#             if self.is_sparse_inputs:
#                 xw = torch.sparse.mm(x, self.weight)
#             else:
#                 xw = torch.mm(x, self.weight)
#         else:
#             xw = self.weight

#         out = torch.sparse.mm(support, xw)

#         if self.bias is not None:
#             out += self.bias

#         return self.activation(out), support


#     def sparse_dropout(x, rate, noise_shape):
#         """

#         :param x:
#         :param rate:
#         :param noise_shape: int scalar
#         :return:
#         """
#         random_tensor = 1 - rate
#         random_tensor += torch.rand(noise_shape).to(x.device)
#         dropout_mask = torch.floor(random_tensor).byte()
#         i = x._indices() # [2, 49216]
#         v = x._values() # [49216]

#         # [2, 4926] => [49216, 2] => [remained node, 2] => [2, remained node]
#         i = i[:, dropout_mask]
#         v = v[dropout_mask]

#         out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)

#         out = out * (1./ (1-rate))

#         return out
