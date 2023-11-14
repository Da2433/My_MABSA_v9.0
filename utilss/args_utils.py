import os
import argparse
import logging
import torch
import random
import numpy as np


def get_args():
    parser = argparse.ArgumentParser()
    ## data path
    parser.add_argument("--dataset", default='tw15', type=str)
    parser.add_argument("--train_data_path", default='train.tsv', type=str)
    parser.add_argument("--valid_data_path", default='dev.tsv', type=str)
    parser.add_argument("--test_data_path", default='test.tsv', type=str)
    parser.add_argument("--processed_feature_path", default="./datas/processed_feature_path/")
    parser.add_argument('--crop_size', type=int, default=224, help='crop size of image')


    parser.add_argument("--num_labels", default=3, type=int)


    # output_path
    parser.add_argument("--ckpt_out_file", default='outs/', type=str)
    parser.add_argument("--output_log_path", default='outs/performance.txt', type=str)
    parser.add_argument("--output_network_path", default='outs/network.txt', type=str)
    parser.add_argument("--output_parameter_path", default='outs/parameter.txt', type=str)


    # modal paras
    parser.add_argument("--task_name", default='MABSA', type=str, help="")
    parser.add_argument('--pretrain_model_name', default='/home/gb/yzd/YZD_Code/bert-base-uncased', type=str)
    parser.add_argument('--output_attentions', type=bool, default=False)
    parser.add_argument('--output_hidden_states', type=bool, default=False)
    parser.add_argument("--do_lower_case", type=bool, default=True)
    parser.add_argument("--bert_hidden_dropout_prob", type=float, default=0.4)

    parser.add_argument("--bert_hidden_size", type=int, default=768)
    parser.add_argument("--att_hidden_size", type=int, default=512)
    parser.add_argument("--num_head", type=int, default=8)
    parser.add_argument("--dim_feedforward", type=int, default=2048)


    #training paras
    parser.add_argument("--num_epoch", default=100, type=int)
    # parser.add_argument("--save_proportion", default=0.5, type=float,
    #                     help="Proportion of steps to save models for. E.g., 0.5 = 50% of training.")
    parser.add_argument("--bert_learning_rate", default=1e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam.")

    # parser.add_argument("--trade_off", default=1e-2, type=float, help="the weight of the reconstruction loss")
    # parser.add_argument("--w_mix", default=0.85, type=float, help="the weight of mixed up")


    parser.add_argument("--batch_size", default=32, type=int, help="Total batch size for training.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8)
    parser.add_argument('--fp16', default=False, type=bool)
    parser.add_argument('--is_cuda', default=None, type=str)
    parser.add_argument('--seed', default=123, type=int)

    parser.add_argument("--max_seq_length", default=64, type=int)
    parser.add_argument("--max_entity_length", default=16, type=int)

    parser.add_argument("--num_warmup_steps", default=37, type=int)
    parser.add_argument("--patience", default=20, type=int)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)


    parser.add_argument('--loss_scale', type=float, default=0)

    return parser.parse_args()


def update_args(args):

    args.dataset = 'tw15'
    # args.dataset = 'tw17'
    args.learning_rate = 1e-5
    args.trade_off = 0.001
    args.w_mix = 0.82
    args.weight_m = 0.001


    if args.dataset == 'tw15':
        args.data_dir = '/home/gb/yzd/YZD_Code/datas/twitter2015'
        #faster-rcnn
        # args.image_region_dir = '/home/gb/yzd/YZD_Code/code/vit-pytorch-main/Twitter15-17_Faster_Rcnn/15/'
        ###masad
        # args.data_dir = '/home/gb/yzd/YZD_Code/code/vit-pytorch-main/MASAD_Datas/MASAD_preprocessed/preprocessed/text/Food'
        args.image_path = 'E:/CYJcodes/datasets/IJCAI2019_data/twitter2015_images'
         #vit_base_16_224
        args.image_region_dir = '/home/gb/yzd/YZD_Code/datas/vit_feature/'
        ###masad
        # args.image_region_dir = '/home/gb/yzd/YZD_Code/code/vit-pytorch-main/MASAD_Datas/image_features/Food/'
        # resnet_feature
        # args.image_region_dir = '/home/gb/yzd/YZD_Code/datas/region_box/15/'


    elif args.dataset == 'tw17':
        args.data_dir = '/home/gb/yzd/YZD_Code/datas/twitter2017'
        # faster-rcnn
        # args.image_region_dir = '/home/gb/yzd/YZD_Code/code/vit-pytorch-main/Twitter15-17_Faster_Rcnn/17/'
        args.image_path = 'E:/CYJcodes/datasets/IJCAI2019_data/twitter2017_images'
         #vit_base_16_224
        args.image_region_dir = '/home/gb/yzd/YZD_Code/datas/vit_feature17/'
        # resnet_feature
        # args.image_region_dir = '/home/gb/yzd/YZD_Code/datas/region_box/17/'
    if torch.cuda.is_available():
        args.is_cuda = True
    else:
        args.is_cuda = False

    if args.is_cuda:
        args.device = torch.device('cuda:0')
    else:
        args.device = torch.device('cpu')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    return args