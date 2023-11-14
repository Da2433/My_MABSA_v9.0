import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from torch.nn.modules.transformer import MultiheadAttention, TransformerEncoderLayer, TransformerEncoder
from MultimodalBert.modules.multihead_attention1 import MultiheadAttention_mask
from MultimodalBert.modules.trans_block_with_cross_att import TransWithCrossAtt
from MultimodalBert.modules.MM_fusion_trans import MmFusionTrans_mask, MmFusionTrans
from MultimodalBert.modules.my_transformer_encoder import my_transformer_encoder
from MultimodalBert.modules.contrastive_module import get_loss
import random
import math
class RegionBertClassification(nn.Module):
    def __init__(self, args):
        super(RegionBertClassification, self).__init__()
        self.args = args
        self.bert = BertModel.from_pretrained(args.pretrain_model_name)
        self.drop = nn.Dropout(p=args.bert_hidden_dropout_prob)

        self.as_lin = nn.Linear(args.bert_hidden_size, args.att_hidden_size)
        self.sent_lin = nn.Linear(args.bert_hidden_size, args.att_hidden_size)
        self.img_region_lin = nn.Linear(768, args.att_hidden_size)
        self.all_lin = nn.Linear(args.att_hidden_size*2, args.att_hidden_size)

        self.region_att = MultiheadAttention_mask(embed_dim=args.att_hidden_size, num_heads=args.num_head, batch_first=True)

        self.as2sent_AAT1 = TransWithCrossAtt(args, dropout=args.bert_hidden_dropout_prob)
        self.as2sent_AAT2 = TransWithCrossAtt(args, dropout=args.bert_hidden_dropout_prob)
        self.as2region_AAT1 = TransWithCrossAtt(args, dropout=args.bert_hidden_dropout_prob)
        self.as2region_AAT2 = TransWithCrossAtt(args, dropout=args.bert_hidden_dropout_prob)

        self.region2sent_MFT = MmFusionTrans(args, dropout=args.bert_hidden_dropout_prob)
        self.sent2region_MFT = MmFusionTrans(args, dropout=args.bert_hidden_dropout_prob)

        # self.as_sent_region_att = TransformerEncoderLayer(d_model=args.att_hidden_size,
        #                                                   nhead=args.num_head@
        #                                                   batch_first=True)
        self.as_sent_region_att = my_transformer_encoder(args)

        self.classifier = nn.Linear(args.att_hidden_size, args.num_labels)
        self.loss_fn = nn.CrossEntropyLoss()
        # self.loss_mse = nn.MSELoss()
        # self.relu1 = nn.ReLU()
#        self.contr_loss = SimSiam(args)
        # self.sentlin_forfusion = nn.Linear(args.att_hidden_size, 1, bias=False)
        # self.regionlin_forfusion = nn.Linear(args.att_hidden_size, 1, bias=False)


    # def multimodal_mix(self, x, img, batch_len):
    #
    #     pseudo_features = img[torch.LongTensor(np.random.randint(0, img.size(0), batch_len))]
    #     alpha = torch.tensor([random.betavariate(1, 1) for _ in range(x.size(1))]).unsqueeze(0).unsqueeze(-1).type_as(x)
    #     mixed_x = alpha * x[:batch_len] + (1 - alpha) * pseudo_features
    #     x = torch.cat([x[:batch_len], img, mixed_x], dim=0)
    #
    #     return x

    def my_multimodal_mix(self, current_input: torch.Tensor, cross_input: torch.Tensor):
        '''
        beta 分布来将两种相同length的模态融合
        '''
        alpha = torch.tensor([random.betavariate(1, 1) for _ in range(current_input.size(1))]).unsqueeze(0).unsqueeze(-1).type_as(current_input)
        mixed_x = alpha * current_input + (1 - alpha) * cross_input
        # mixed_x = mixed_x + current_input
        return mixed_x

    def forward(self, input_ids,
                input_mask=None,
                segment_ids=None,
                s2_input_ids=None,
                s2_input_mask=None,
                s2_segment_ids=None,
                img_region_feat=None,  # img raw datas
                labels=None
                ):
        # Feature extraction
        sent_bert_outs = self.bert(input_ids=input_ids,
                                   token_type_ids=segment_ids,
                                   attention_mask=input_mask)

        s2_bert_outs = self.bert(input_ids=s2_input_ids,
                                 token_type_ids=s2_segment_ids,
                                 attention_mask=s2_input_mask)
        sent_bert_out = self.sent_lin(sent_bert_outs[0])
        s2_bert_out = self.as_lin(s2_bert_outs[0])
        img_region_feat =  self.img_region_lin(img_region_feat)

        text_exp = torch.cat([sent_bert_out, s2_bert_out], dim=1)
        mask_tmp = self.mask(img_region_feat, text_exp)
        img_region_feat = self.region_att(query=img_region_feat, key=img_region_feat, value=img_region_feat,mask_matrix_tmp = mask_tmp )[0]


        # img_region_feat = self.region_att(query=img_region_feat, key=img_region_feat, value=img_region_feat)[0]


        #对比loss

#        la = get_loss(s2_bert_out, sent_bert_out)
#        lb = get_loss(s2_bert_out, img_region_feat)
#        la = self.contr_loss(s2_bert_out, sent_bert_out)
#        lb = self.contr_loss(s2_bert_out, img_region_feat)

        #Aspect-Aware Transformer Layer
        as2sent = self.as2sent_AAT1(query=s2_bert_out, key_value=sent_bert_out, key_padding_mask = input_mask)
        as2sent = self.as2sent_AAT2(query=s2_bert_out, key_value=as2sent, key_padding_mask=s2_input_mask)
       # as2sent = self.region_att(query=as2sent, key=as2sent, value=as2sent)[0]
        as2region = self.as2region_AAT1(query=s2_bert_out, key_value=img_region_feat)
        as2region = self.as2region_AAT2(query=s2_bert_out, key_value=as2region, key_padding_mask=s2_input_mask)
        #as2region = self.region_att(query=as2region, key=as2region, value=as2region)[0]


        #Auxiliary Reconstruction Module

        # 1.auxiliary reconstruction module in the paper
        # H_ts = torch.mean(as2sent, dim=1)
        # H_ti = torch.mean(as2region, dim=1)
        # ave_sent_bertout = torch.mean(sent_bert_out, dim=1)
        # Recon_loss = self.loss_mse(H_ts, ave_sent_bertout) + self.loss_mse(H_ti, ave_sent_bertout)

        # 2.(a1, a2, mix_a) triplet loss
        # H_ts = torch.mean(as2sent, dim=1)
        # H_ti = torch.mean(as2region, dim=1)
        mix_aspect = 0.5 * as2sent + 0.5 * as2region
        # mix_aspect = torch.mean(mix_aspect)
        # Recon_loss = (5e-1) * self.loss_mse(H_ti, H_ts) + 1 * self.loss_mse(H_ts, mix_aspect) + 1 * self.loss_mse(H_ti, mix_aspect)

        ###########simclr
#        lc = self.contr_loss(mix_aspect, as2sent)

        lc = get_loss(mix_aspect, as2sent)
        ld = get_loss(mix_aspect, as2region)
        le = get_loss(as2sent, as2region)
#        ld = self.contr_loss(mix_aspect, as2region)
#        le = self.contr_loss(as2sent, as2region)
        Recon_loss = lc + ld + le
        #
        # sorse = sim_sorce(as2sent, as2region)
        # print(sorse)


        #Multimodal Fusion Transformer
        sent_mix_input = self.args.w_mix*as2sent + (1-self.args.w_mix)*as2region
        region_mix_input = self.args.w_mix*as2region + (1-self.args.w_mix)*as2sent
        sent2region = self.sent2region_MFT(former_input=sent_mix_input, cross_input=as2region,
                                           former_padding_mask=s2_input_mask, cross_padding_mask=s2_input_mask)
        region2sent = self.region2sent_MFT(former_input=region_mix_input, cross_input=as2sent,
                                           former_padding_mask=s2_input_mask, cross_padding_mask=s2_input_mask)

      ###MMF交互后加一个对比损失
        lm = get_loss(region2sent, sent2region)
#        lm = self.contr_loss(region2sent, sent2region)


        # Output

        # 1.gating mechanism with weight
        # gate = torch.sigmoid(self.sentlin_forfusion(sent2region) + self.regionlin_forfusion(region2sent))
        # mmfusion = torch.mul(gate, sent2region) + torch.mul((1-gate), region2sent)
        # as_sent_region_out = self.as_sent_region_att(src=mmfusion)


        # 2.gating mechanism add directly
        # gate = torch.sigmoid(sent2region + region2sent)
        # mmfusion = torch.mul(gate, sent2region) + torch.mul((1-gate), region2sent)
        # as_sent_region_out = self.as_sent_region_att(src=mmfusion)

        # 3.two modal concat directly
        as_sent_region = self.all_lin(torch.cat([sent2region, region2sent], dim=-1))
        as_sent_region_out = self.as_sent_region_att(src=as_sent_region)

        # 4.two channel(gating+concat)
        # gate = torch.sigmoid(self.sentlin_forfusion(sent2region) + self.regionlin_forfusion(region2sent))
        # gate = torch.sigmoid(sent2region + region2sent)
        # mmfusion = torch.mul(gate, sent2region) + torch.mul((1-gate), region2sent)
        # as_sent_region = self.all_lin(torch.cat([sent2region, region2sent], dim=-1))
        # multi_channel_out = torch.tanh(0.05*mmfusion+0.95*as_sent_region)
        # as_sent_region_out = self.as_sent_region_att(src=multi_channel_out)

        outs = torch.mean(as_sent_region_out, dim=1)
     
        logits = self.classifier(outs)

        if labels is not None:
            loss = self.loss_fn(logits.view(-1, self.args.num_labels), labels.view(-1))
            loss = loss + self.args.trade_off*Recon_loss  + self.args.weight_m * lm
            return loss, logits
        return logits



        #############加入mask

    def getBinaryTensor(self, i, boundary):
        one_matrix = torch.ones_like(i)
        zero_matrix = torch.zeros_like(i)

        return torch.where(i > boundary, one_matrix, zero_matrix)

    def mask(self, x, src_img_features):

        # x = x.transpose(0, 1)  # batch * len * dim     MultiheadAttention_Image
        # src_img_features = src_img_features.transpose(0, 1)  # batch * 49 * dim   MultiheadAttention_Image
        ########  mask  img ######### batch * 49 * len
        mask_img = torch.bmm(src_img_features, x.transpose(1, 2)) / math.sqrt(128)
        mask_img = F.softmax(mask_img, dim=-1)

        # mask_matrix = torch.mean(mask_img, dim=2, keepdim=True).repeat(1,1,49)
        # mask_img = F.sigmoid(mask_img)
        # mask_img = self.getBinaryTensor(mask_img,0.015)

        ########  mask  txt ######### batch * len * 49
        mask_txt = torch.bmm(x, src_img_features.transpose(1, 2)) / math.sqrt(128)
        mask_txt = F.softmax(mask_txt, dim=-1)

        # mask_txt = F.sigmoid(mask_txt)

        # mask_txt = self.getBinaryTensor(mask_txt,0.015)

        mask_matrix = torch.bmm(mask_txt, mask_img).cuda()

        # mask_matrix = F.sigmoid(mask_matrix)
        # mask_matrix_output = torch.bmm(mask_img, mask_txt).cuda()
        # mask_matrix = F.softmax(mask_matrix, dim=-1)

        # mask_matrix_alpha = torch.sort(mask_matrix[1].view(-1),dim=-1,descending=True).values[int(mask_matrix[0].view(-1).size()[0] * 0.7)].detach().cpu().numpy().tolist()
        # mask_matrix_alpha = torch.sort(mask_matrix.view(-1),dim=-1,descending=True).values[int(mask_matrix.view(-1).size()[0] * 0.8)].detach().cpu().numpy().tolist()

        # mask_matrix_output = self.getBinaryTensor(mask_matrix, -1)  ## 最好的值设置为0.02  ## 0.01其中decay = 0.1时结果特别好但是非正常停了
        mask_matrix_output = self.getBinaryTensor(mask_matrix, 0.003).eq(0)
        # mask_matrix_output1 = torch.full_like(mask_matrix,0.5).eq(0)
        # mask_matrix_output1
        # mask_matrix_output = []
        # for i in mask_matrix:
        #     mask_list = i.reshape(1, -1)  # Ascending    # or i.view(src_img_features.size(1) * src_img_features.size(1))
        #     mask_list = sorted(mask_list.squeeze().tolist())
        #     num_tmp = int(len(mask_list) * 0.15)
        #     mask_matrix_tmp = self.getBinaryTensor(i, mask_list[num_tmp])
        #     mask_matrix_output.append(mask_matrix_tmp.tolist())

        return mask_matrix_output.detach()


