'''
Function:
    Implementation of FeaturesMemory
Author:
    Zhenchao Jin
'''
from math import trunc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from ..base import SelfAttentionBlock
from ...backbones import BuildActivation, BuildNormalization, constructnormcfg
from timm.models.layers import trunc_normal_
from einops import rearrange, repeat

'''FeaturesMemory'''
class FeaturesMemory(nn.Module):
    def __init__(self, num_classes, feats_channels, transform_channels, out_channels,num_feats_per_cls,anchor_pixels,negative_pixels,use_context_within_image=True, 
                  use_hard_aggregate=False, norm_cfg=None, act_cfg=None):
        super(FeaturesMemory, self).__init__()
        assert num_feats_per_cls > 0, 'num_feats_per_cls should be larger than 0'
        # set attributes
        self.num_classes = num_classes
        self.feats_channels = feats_channels
        self.transform_channels = transform_channels
        self.out_channels = out_channels
        self.num_feats_per_cls = num_feats_per_cls
        self.anchor_pixels = anchor_pixels
        self.negative_pixels = negative_pixels
        self.use_context_within_image = use_context_within_image
        self.use_hard_aggregate = use_hard_aggregate

        # init memory  nn.parameter()与tensor差不多，唯一的就是nn.parameter()定义的张量被认为是module的可训练参数 
        # self.memory = nn.Parameter(torch.zeros(num_classes, num_feats_per_cls, feats_channels, dtype=torch.float), requires_grad=False)
        self.memory = nn.Parameter(torch.zeros(num_classes, num_feats_per_cls, feats_channels, dtype=torch.float),requires_grad=False)
        trunc_normal_(self.memory,std=0.02) # 内存库中的数据一开始设置在一定范围内

        # define self_attention module
        self.attention = SelfAttentionBlock(
            key_in_channels=feats_channels,
            query_in_channels=feats_channels,
            transform_channels=transform_channels,
            out_channels=feats_channels,
            share_key_query=False,
            query_downsample=None,
            key_downsample=None,
            key_query_num_convs=2,
            value_out_num_convs=1,
            key_query_norm=True,
            value_out_norm=True,
            matmul_norm=True,
            with_out_project=True,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        self.fuse_memory_conv = nn.Sequential(
            nn.Conv2d(feats_channels * self.num_feats_per_cls, feats_channels, kernel_size=1, stride=1, padding=0, bias=False),
            BuildNormalization(constructnormcfg(placeholder=feats_channels, norm_cfg=norm_cfg)),
            BuildActivation(act_cfg),
        )
        # whether need to fuse the contextual information within the input image
        self.bottleneck = nn.Sequential(
            nn.Conv2d(feats_channels * 2, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            BuildNormalization(constructnormcfg(placeholder=out_channels, norm_cfg=norm_cfg)),
            BuildActivation(act_cfg),
        )
        if use_context_within_image:
            self.self_attention_ms = SelfAttentionBlock(
                key_in_channels=feats_channels,
                query_in_channels=feats_channels,
                transform_channels=transform_channels,
                out_channels=feats_channels,
                share_key_query=False,
                query_downsample=None,
                key_downsample=None,
                key_query_num_convs=2,
                value_out_num_convs=1,
                key_query_norm=True,
                value_out_norm=True,
                matmul_norm=True,
                with_out_project=True,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            )
            self.bottleneck_ms = nn.Sequential(
                nn.Conv2d(feats_channels * 2, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                BuildNormalization(constructnormcfg(placeholder=out_channels, norm_cfg=norm_cfg)),
                BuildActivation(act_cfg),
            )
    '''forward'''
    def forward(self, feats, preds=None, feats_ms=None):
        batch_size, num_channels, h, w = feats.size()
        # extract the history features
        # --(B, num_classes, H, W) --> (B*H*W, num_classes)
        weight_cls = preds.permute(0, 2, 3, 1).contiguous()
        weight_cls = weight_cls.reshape(-1, self.num_classes)
        weight_cls = F.softmax(weight_cls, dim=-1)# N * num_classes
        if self.use_hard_aggregate:
            labels = weight_cls.argmax(-1).reshape(-1, 1)
            onehot = torch.zeros_like(weight_cls).scatter_(1, labels.long(), 1)
            weight_cls = onehot
        # --(B*H*W, num_classes) * (num_classes, C) --> (B*H*W, C)
        selected_memory_list = []
        # 定义一个均值列表
        avg_list = []
        for idx in range(self.num_feats_per_cls):
            memory = self.memory.data[:, idx, :] # 表示的是 num_classes *
            selected_memory = torch.matmul(weight_cls, memory)
            selected_memory_list.append(selected_memory.unsqueeze(1))
        
        

        # calculate selected_memory according to the num_feats_per_cls
        relation_selected_memory_list = []
        for idx, selected_memory in enumerate(selected_memory_list):
            # --(B*H*W, C) --> (B, H, W, C)
            selected_memory = selected_memory.view(batch_size, h, w, num_channels)
            # --(B, H, W, C) --> (B, C, H, W)
            selected_memory = selected_memory.permute(0, 3, 1, 2).contiguous()
            # --append
            relation_selected_memory_list.append(selected_memory)
        # --concat
        selected_memory = torch.cat(relation_selected_memory_list, dim=1)
        selected_memory = self.fuse_memory_conv(selected_memory) # 投票
        selected_memory = self.attention(feats,selected_memory)
        # return
        memory_output = self.bottleneck(torch.cat([feats, selected_memory], dim=1))
        if self.use_context_within_image:
            feats_ms = self.self_attention_ms(feats, feats_ms)
            memory_output = self.bottleneck_ms(torch.cat([feats_ms, memory_output], dim=1))
        return  memory_output
    
    
    def prototype_learning(self,_c, out_seg, gt_seg, masks,targets):
        pred_seg = torch.max(out_seg, 1)[1] # 获得最大值的索引
        mask = (gt_seg == pred_seg.view(-1)) # 获得分类正确的mask
        
        #(b*h*w) *c    *   c* (num_classes * proto_num)  ===>  (b*h*w) * (num_classes * proto_num)
        cosine_similarity = torch.mm(_c, self.memory.view(-1, self.memory.shape[-1]).t()) # _c的尺寸是 K * C  proto的尺寸为c * (n*proto_num) 

        proto_logits = cosine_similarity  #原型的相似度矩阵，当前特征与原型的相似度矩阵
        proto_target = gt_seg.clone().float() # 尺寸是1 * N

        protos = self.memory.data.clone()
        for k in range(self.num_classes):
            init_q = masks[..., k] # （b*h*w） * (num_proto)  * (num_class)  init_q尺寸为 （b*h*w） * (num_proto)
            init_q = init_q[gt_seg == k, ...] # 选择类别为k的像素点对应的原型
            if init_q.shape[0] == 0: # 如果等于0说明该类没有出现
                continue

            q, indexs = self.distributed_sinkhorn(init_q)# 输入类别为K的像素点 N*(num_proto) 返回最近的原型是第几个，以及

            m_k = mask[gt_seg == k] # 在分类正确的像素中选择类别为k的掩码

            c_k = _c[gt_seg == k, ...]# 在当前的特征中选择类别为k的特征 N * C

            m_k_tile = repeat(m_k, 'n -> n tile', tile=self.num_feats_per_cls)

            m_q = q * m_k_tile  # n x self.num_prototype

            c_k_tile = repeat(m_k, 'n -> n tile', tile=c_k.shape[-1])

            c_q = c_k * c_k_tile  # n x embedding_dim

            f = m_q.transpose(0, 1) @ c_q  # self.num_prototype x embedding_dim

            n = torch.sum(m_q, dim=0)

            if torch.sum(n) > 0:
                f = F.normalize(f, p=2, dim=-1) # 经过正则化之后的特征 
                new_value = self.momentum_update(old_value=protos[k, n != 0, :], new_value=f[n != 0, :],momentum=0.999, debug=False) # 对原来的原型进行更新
                protos[k, n != 0, :] = new_value
            #proto_target[gt_seg == k] = indexs.float() + (self.num_feats_per_cls * k) # 第几个类，加上原型的个数乘以当前是哪个类
            #proto_target[gt_seg == k] = (self.num_feats_per_cls * k)

        pp = F.normalize(protos,p=2,dim=-1)
        self.memory = nn.Parameter(pp,requires_grad=False)

        memory_calculate = nn.Parameter(pp)
        avg_list = []
        # for num_cls in range(self.num_classes):
        #     avg = memory_calculate.data[num_cls,:,:]  # 每个类的均值特征  N  ×  通道数目
        #     avg_list.append(avg.mean(dim=0).unsqueeze(1)) # 得到每个类的均值特征

        # avg_feature_per_class = torch.cat(avg_list,dim=1) # C × F
        # distance_matrix = torch.cdist(avg_feature_per_class,avg_feature_per_class)
        # distance_matrix.fill_diagonal_(float('inf'))
        # min_distance = (1 - torch.min(distance_matrix)) / self.num_classes
        for num_cls in range(self.num_classes):
            avg = memory_calculate.data[num_cls,:,:]  # 每个类的均值特征  N  ×  通道数目
            avg_list.append(avg.mean(dim=0).unsqueeze(1)) # 得到每个类的均值特征

        avg_feature_per_class = torch.cat(avg_list,dim=1) # C × F
        similarity_matrix = F.cosine_similarity(avg_feature_per_class.unsqueeze(1), avg_feature_per_class.unsqueeze(0), dim=2)
        similarity_matrix.fill_diagonal_(0)
        non_zero_similarities = similarity_matrix[similarity_matrix != 0]
        min_similarity = non_zero_similarities.min() # 取最小值
        min_distance = ( 1 + min_similarity ) / self.num_classes


        # loss_ce
        h, w = targets.size(1), targets.size(2) # targets尺为B * H * W
        pred = F.interpolate(input=out_seg, size=(h, w), mode='bilinear', align_corners=True)
        ce_loss  = nn.CrossEntropyLoss(ignore_index=255)
        loss_ce = ce_loss(pred, targets.long())

        # loss_ppc 
        loss_ppc = 0
        count = 0
        for cl in range(self.num_classes):
            current_logists = proto_logits[gt_seg == cl] # 取属于类别cl的像素点
            if current_logists.shape[0] == 0:
                continue
            count = count + 1
            #mm = proto_target[proto_target == cl]
            if cl==0:
                k = self.num_feats_per_cls * cl
                for num in range(self.num_feats_per_cls):
                    logists_per = current_logists[:,k:k+1]
                    k = k + 1
                    all_logists = torch.cat([logists_per,current_logists[:,self.num_feats_per_cls*(cl+1):]],dim=1)
                    #mm[gt_seg == cl] = 0
                    mm = torch.zeros(all_logists.shape[0])
                    loss_ppc = loss_ppc + F.cross_entropy(all_logists,mm.long().cuda(),ignore_index=255)
            elif cl == self.num_classes-1:
                k = self.num_feats_per_cls*cl
                for num in range(self.num_feats_per_cls):
                    logists_per = current_logists[:,k:k+1]
                    k = k + 1
                    all_logists = torch.cat([logists_per,current_logists[:,:self.num_feats_per_cls*(cl)]],dim=1)
                    mm = torch.zeros(all_logists.shape[0])
                    loss_ppc = loss_ppc + F.cross_entropy(all_logists,mm.long().cuda(),ignore_index=255)
            else:
                k = self.num_feats_per_cls*cl
                for num in range(self.num_feats_per_cls):
                    logists_per = current_logists[:,k:k+1]
                    k = k + 1
                    all_logists = torch.cat([logists_per,torch.cat([current_logists[:,:self.num_feats_per_cls*(cl)],current_logists[:,self.num_feats_per_cls*(cl+1):]],dim=1)],dim=1)
                    mm = torch.zeros(all_logists.shape[0])
                    loss_ppc = loss_ppc + F.cross_entropy(all_logists,mm.long().cuda(),ignore_index=255)
        loss_ppc = loss_ppc / count

        loss_ppc = loss_ppc /self.num_feats_per_cls

        #loss_ppc = F.cross_entropy(proto_logits,proto_target.long(),ignore_index=255)

        # # loss_ppd
        # contrast_logits = proto_logits[proto_target != 255, :]
        # contrast_target = proto_target[proto_target!=255]
        # logits = torch.gather(contrast_logits, 1, contrast_target[:, None].long())
        # loss_ppd = (1 - logits).pow(2).mean()

        if dist.is_available() and dist.is_initialized():
            protos  = self.memory.data.clone()
            dist.all_reduce(protos.div_(dist.get_world_size()))
            self.memory = nn.Parameter(protos,requires_grad=False)
        return loss_ce,loss_ppc,min_distance



    # '''update'''
    # def update(self, features, segmentation, ignore_index=255, strategy='cosine_similarity', momentum_cfg=None, learning_rate=None):
    #     assert strategy in ['mean', 'cosine_similarity']
    #     batch_size, num_channels, h, w = features.size()
    #     momentum = momentum_cfg['base_momentum'] # 0.9
    #     if momentum_cfg['adjust_by_learning_rate']:
    #         momentum = momentum_cfg['base_momentum'] / momentum_cfg['base_lr'] * learning_rate
    #     # use features to update memory
    #     segmentation = segmentation.long()
    #     features = features.detach()
    #     features = features.permute(0, 2, 3, 1).contiguous() # 变为B * H * W * C
    #     features = features.view(batch_size * h * w, num_channels) # 变成 K * C
    #     clsids = segmentation.unique() #获得该图中的类别
    #     for clsid in clsids:
    #         if clsid == ignore_index: continue
    #         # --(B, H, W) --> (B*H*W,)
    #         seg_cls = segmentation.view(-1)
    #         # --extract the corresponding feats: (K, C)
    #         feats_cls = features[seg_cls == clsid]  # 获得确定类的特征
    #         # --init memory by using extracted features
    #         need_update = True # 默认是需要更新的
    #         for idx in range(self.num_feats_per_cls):
    #             if (self.memory[clsid][idx] == 0).sum() == self.feats_channels: #表明这个类的特征是空的，直接送进去
    #                 self.memory[clsid][idx].data.copy_(feats_cls.mean(0)) # 一开始每个初始化都一样
    #                 need_update = False
    #                 break
    #         if not need_update: continue

    #         # --update according to the selected strategy
    #         if self.num_feats_per_cls == 1:
    #             if strategy == 'mean':
    #                 feats_cls = feats_cls.mean(0)

    #             elif strategy == 'cosine_similarity':  #应用向量相似性，
    #                 # a.expand_as(b) 表示将a的尺寸扩充到与b相同，
    #                 similarity = F.cosine_similarity(feats_cls, self.memory[clsid].data.expand_as(feats_cls)) # 结果是尺度为 K
    #                 weight = (1 - similarity) / (1 - similarity).sum()  # 存储的是每个类的权重

    #                 #feat_cls的尺寸为 K×C，weight的尺寸是K,给weight添加一维度为K×1， 与feat_cls相乘，等同于给每列乘上相同的值
    #                 '''
    #                    例如：
    #                    feat_cls = tensor([[1., 1., 1., 1.],
    #                                       [1., 1., 1., 1.],
    #                                       [1., 1., 1., 1.]])

    #                     weight = tensor([1., 2., 3.])
    #                     weight.unsqueeze(-1) = tensor([[1.],
    #                                                    [2.],
    #                                                    [3.]])
    #                     feat_cls * weight.unsqueeze(-1) = tensor([[1., 1., 1., 1.],
    #                                                               [2., 2., 2., 2.],
    #                                                               [3., 3., 3., 3.]]) 
                    
    #                 '''
    #                 feats_cls = (feats_cls * weight.unsqueeze(-1)).sum(0)  # 权重 * 向量， 加权求和 尺寸变为 K
    #             feats_cls = (1 - momentum) * self.memory[clsid].data + momentum * feats_cls.unsqueeze(0)
    #             self.memory[clsid].data.copy_(feats_cls)

    #         else:
    #             assert strategy in ['cosine_similarity']
    #             # ----(K, C) * (C, num_feats_per_cls) --> (K, num_feats_per_cls)
    #             relation = torch.matmul(
    #                 F.normalize(feats_cls, p=2, dim=1), 
    #                 F.normalize(self.memory[clsid].data.permute(1, 0).contiguous(), p=2, dim=0),
    #             ) # 求得是跟当前的的数据集级别的特征对应的最相似的特征
    #             argmax = relation.argmax(dim=1)# 返回最大值的索引
    #             # ----for saving memory during training
    #             for idx in range(self.num_feats_per_cls):
    #                 mask = (argmax == idx)
    #                 feats_cls_iter = feats_cls[mask]
    #                 memory_cls_iter = self.memory[clsid].data[idx].unsqueeze(0).expand_as(feats_cls_iter)
    #                 similarity = F.cosine_similarity(feats_cls_iter, memory_cls_iter)
    #                 weight = (1 - similarity) / (1 - similarity).sum()
    #                 feats_cls_iter = (feats_cls_iter * weight.unsqueeze(-1)).sum(0)
    #                 self.memory[clsid].data[idx].copy_(self.memory[clsid].data[idx] * (1 - momentum) + feats_cls_iter * momentum)
                    
    #     # syn the memory
    #     if dist.is_available() and dist.is_initialized():
    #         memory = self.memory.data.clone()
    #         dist.all_reduce(memory.div_(dist.get_world_size()))
    #         self.memory = nn.Parameter(memory, requires_grad=False)
    
    def distributed_sinkhorn(self,out, sinkhorn_iterations=3, epsilon=0.05):
        L = torch.exp(out / epsilon).t() # K x (num_proto) ==> num_proto × K
        B = L.shape[1]  # 改类多少个像素点
        K = L.shape[0]  # 多少个原型

        # make the matrix sums to 1
        sum_L = torch.sum(L) #求所有矩阵中元素的和 
        L /= sum_L

        for _ in range(sinkhorn_iterations):
            L /= torch.sum(L, dim=1, keepdim=True)
            L /= K

            L /= torch.sum(L, dim=0, keepdim=True)
            L /= B

        L *= B 
        L = L.t() # K * num_proto

        indexs = torch.argmax(L, dim=1) # 与当前像素点最近的原型的下标 返回最大值的索引 应该是1 * N
        # L = torch.nn.functional.one_hot(indexs, num_classes=L.shape[1]).float()
        L = F.gumbel_softmax(L, tau=0.5, hard=True)

        return L, indexs
    def momentum_update(self,old_value, new_value, momentum, debug=False):
        update = momentum * old_value + (1 - momentum) * new_value
        if debug:
            print("old prot: {:.3f} x |{:.3f}|, new val: {:.3f} x |{:.3f}|, result= |{:.3f}|".format(
                momentum, torch.norm(old_value, p=2), (1 - momentum), torch.norm(new_value, p=2),
                torch.norm(update, p=2)))
        return update