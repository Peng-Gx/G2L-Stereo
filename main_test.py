# from __future__ import print_function, division
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torchvision.utils as vutils
import torch.nn.functional as F
import numpy as np
import time
from tensorboardX import SummaryWriter
from datasets import __datasets__
# from models import __models__, model_loss_train_attn_only, model_loss_train_freeze_attn, model_loss_train, model_loss_test
from utils import *
from torch.utils.data import DataLoader
import gc
# from apex import amp
import cv2
# from datasets import sceneflow_listfile

# from models.coexnet.myCoEx import myCoEx
# from models.coexnet.CoEx import CoEx
from tqdm import tqdm

import torch
import torch.nn.functional as F
import shutil


def disp_to_color(img):
    b,c,h,w = img.shape

    mapMatrix = np.array([[0,0,0,114],[0,0,1,185],[1,0,0,114],[1,0,1,174],
                            [0,1,0,114],[0,1,1,185],[1,1,0,114],[1,1,1,0]])
    bins = mapMatrix[:-1,-1]
    cbins = np.cumsum(bins)
    bins = bins/cbins[-1]
    cbins = cbins[:-1]/cbins[-1]

    cbins_ = np.repeat(np.repeat(np.repeat(cbins.reshape(1,-1,1,1), b, axis=0), h, axis=2), w, axis=3)
    img_ = np.repeat(img, len(cbins), axis=1)
    ind = np.sum(img_>cbins_, axis=1, keepdims=True)
    cbins = np.insert(cbins, 0, 0.)
    bins = 1/bins

    img_left = (img-cbins[ind])*bins[ind]
    img_left = np.repeat(img_left, 3, axis=1)
    img_right = 1-img_left

    color_left = np.concatenate((mapMatrix[ind, 2], mapMatrix[ind, 1], mapMatrix[ind, 0]), axis=1)
    color_right = np.concatenate((mapMatrix[ind+1, 2], mapMatrix[ind+1, 1], mapMatrix[ind+1, 0]), axis=1)
    color_img = color_left*img_right+color_right*img_left
    return color_img

def test():
    # testing
    avg_test_scalars = AverageMeterDict()
    with tqdm(total=len(TestImgLoader),desc="Testing") as pbar:
        for batch_idx, sample in enumerate(TestImgLoader):    
            start_time = time.time()
            loss, scalar_outputs = test_sample(sample)
            avg_test_scalars.update(scalar_outputs)
            del scalar_outputs 
            pbar.update(1)
            pbar.set_postfix({'test loss':'{:.3f}'.format(loss),'time':'{:.3f}'.format(time.time() - start_time)})
    avg_test_scalars = avg_test_scalars.mean()

    save_scalars(logger, 'fulltest', avg_test_scalars, 0)
    with open(os.path.join(args.logdir,'test_result.txt'),'a') as file:
        file.write(f'epoch_idx:0\n')
        for k,v in avg_test_scalars.items():
            file.write(f'{k}:{v}\n')
        file.write('\n')
    print("avg_test_scalars", avg_test_scalars)
    gc.collect() 

# test one sample
@make_nograd_func
def test_sample(sample):
    model.eval()
    imgL, imgR, disp_gt = sample['left'], sample['right'], sample['disparity']
    if torch.cuda.is_available() and args.cuda:
        imgL = imgL.cuda()
        imgR = imgR.cuda()
        disp_gt = disp_gt.cuda()

    mask = (disp_gt < args.maxdisp) & (disp_gt > 0)
    disp_est = model(imgL, imgR)[0]

    if mask.float().mean()<0.01:
        loss=torch.tensor(0, dtype=torch.float32, device=disp_gt.device)
    else:
        loss = F.smooth_l1_loss(disp_est[mask], disp_gt[mask], reduction='mean')

    scalar_outputs = {"loss": loss}
    scalar_outputs["EPE"] = [EPE_metric(disp_est, disp_gt, mask)]
    scalar_outputs["D1"] = [D1_metric(disp_est, disp_gt, mask)]
    scalar_outputs["Thres1"] = [Thres_metric(disp_est, disp_gt, mask, 1.0)]
    scalar_outputs["Thres2"] = [Thres_metric(disp_est, disp_gt, mask, 2.0)]
    scalar_outputs["Thres3"] = [Thres_metric(disp_est, disp_gt, mask, 3.0)]

    if args.visual:
        top_pad, right_pad = sample['top_pad'], sample['right_pad']

        imgL_name_list = [x.split('/') for x in sample['left_filename']]
        imgL_name_list = ['_'.join(x) for x in imgL_name_list]

        #保存色彩真值图
        color_gt_path = os.path.join(args.logdir,'color_disp_gt')
        os.makedirs(color_gt_path, exist_ok=True)
        disp_gt_np = disp_gt.cpu().numpy()
        for i in range(disp_gt_np.shape[0]):
            disp_gt_np[i] = np.where(np.isinf(disp_gt_np[i]), 0, disp_gt_np[i])
            img = disp_gt_np[i]
            img = img[top_pad[i]:,:img.shape[-1]-right_pad[i]]
            # color_img = img.reshape(1,1,img.shape[-2],img.shape[-1]).astype(np.float64)/args.maxdisp
            color_img = img.reshape(1,1,img.shape[-2],img.shape[-1]).astype(np.float64)/np.amax(img)
            color_img = np.clip(color_img, 0, 1)
            color_img = (disp_to_color(color_img)*255).transpose(0,2,3,1)
            cv2.imwrite(os.path.join(color_gt_path, imgL_name_list[i]), color_img[0])

        #保存色彩视差图
        color_disp_path = os.path.join(args.logdir,'color_disp_est')
        os.makedirs(color_disp_path, exist_ok=True)
        disp_est_np = disp_est.cpu().numpy()
        for i in range(disp_est_np.shape[0]):
            img = disp_est_np[i]
            img = img[top_pad[i]:,:img.shape[-1]-right_pad[i]]
            # color_img = img.reshape(1,1,img.shape[-2],img.shape[-1]).astype(np.float64)/args.maxdisp
            color_img = img.reshape(1,1,img.shape[-2],img.shape[-1]).astype(np.float64)/np.amax(disp_gt_np[i])
            color_img = np.clip(color_img, 0, 1)
            color_img = (disp_to_color(color_img)*255).transpose(0,2,3,1)
            cv2.imwrite(os.path.join(color_disp_path, imgL_name_list[i]), color_img[0])

        #保存error
        error_path = os.path.join(args.logdir,'disp_error')
        os.makedirs(error_path, exist_ok=True)
        errormap = disp_error_image_func.apply(disp_est, disp_gt)
        for i in range(errormap.shape[0]):
            img = errormap[i]
            img = img[:,top_pad[i]:,:img.shape[-1]-right_pad[i]]
            img = img.cpu().numpy().transpose(1,2,0)*255
            cv2.imwrite(os.path.join(error_path, imgL_name_list[i]), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


    return tensor2float(loss), tensor2float(scalar_outputs)


if __name__ == '__main__':
    cudnn.benchmark = True
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'


    parser = argparse.ArgumentParser(description='Stereo Matching')
    parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')

    parser.add_argument('--dataset', default='sceneflow', help='dataset name', choices=__datasets__.keys())
    parser.add_argument('--datapath', default="../sub_sceneflow", help='data path')
    parser.add_argument('--datapath_another', default="../sub_sceneflow", help='data path')
    parser.add_argument('--testlist',default='filenames/sub_sceneflow_test_2024-12-08.txt', help='testing list')
    
    parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
    parser.add_argument('--num_workers', type=int, default=8, help='num_workers')

    parser.add_argument('--logdir',default='./log_test', help='the directory to save logs and checkpoints')
    parser.add_argument('--loadckpt', default='pretrained_model/checkpoint_000039.ckpt',help='load the weights from a specific checkpoint')
    parser.add_argument('--seed', type=int, default=42, metavar='S', help='random seed (default: 1)')

    parser.add_argument('--cuda', default=True, action='store_true', help='use cuda to train the model')
    parser.add_argument('--visual', default=False, action='store_true', help='visualization the map')

    # parse arguments, set seeds
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and args.cuda:
        torch.cuda.manual_seed(args.seed)

    # dataset, dataloader
    StereoDataset = __datasets__[args.dataset]
    if args.dataset == 'kitti':
        test_dataset = StereoDataset(args.datapath, args.datapath_another, args.testlist, False)
    else:
        test_dataset = StereoDataset(args.datapath, args.testlist, False)
    TestImgLoader = DataLoader(test_dataset, args.test_batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)

    # model
    from models.g2l.g2l import G2l
    model = G2l()
    model = nn.DataParallel(model)
    if torch.cuda.is_available() and args.cuda:
        model.cuda()

    # create summary logger
    os.makedirs(args.logdir, exist_ok=True)
    args.logdir=os.path.join(args.logdir,time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime()))
    logger = SummaryWriter(args.logdir)


    #保存命令行参数
    with open(os.path.join(args.logdir,'config.txt'),'a') as file:
        for k,v in vars(args).items():
            file.write(f'{k}:{v}\n')
        file.write('\n')

    #加载参数
    print("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt, 
                            map_location=torch.device('cuda') if torch.cuda.is_available() and args.cuda else torch.device('cpu'))
    model.load_state_dict(state_dict['model'])
    
    test()