from __future__ import print_function
from __future__ import division

import os
import sys
import time
import datetime
import os.path as osp
import numpy as np
import numpy.ma as ma
import warnings
import pickle
from skimage import io, transform
from PIL import Image
import os.path as osp
import random

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms, utils

import models
from losses import triplet_loss, xent_loss
from loss import CrossEntropyLoss, TripletLoss
from utils.avgmeter import AverageMeter
from utils.iotools import check_isfile
from utils.loggers import Logger, RankLogger
from utils.torchtools import count_num_param, accuracy, load_pretrained_weights, save_checkpoint, resume_from_checkpoint
from utils.visualtools import visualize_ranked_results
from utils.generaltools import set_random_seed
from optimizers import init_optimizer
from lr_schedulers import init_lr_scheduler
from functions import keyfromval, strint, ranges, search
from datasets import init_imgreid_dataset
from data_loading import VeriDataset as vd
#from transform import Rescale, RandomCrop, ToTensor 
from transform import train_transforms
from test_loading import ImageDataManager
from evaluation import evaluate

from data_loading import VerispanDataset as vdspan
from utils.reranking import re_ranking,re_ranking_numpy

def test(model, queryloader, galleryloader, batch_size, use_gpu, ranks=[1, 5, 10], return_distmat=False):
    batch_time = AverageMeter()

    model.eval()

    with torch.no_grad():
        qf, q_pids, q_camids = [], [], []
        for batch_idx, (imgs, pids, camids, _) in enumerate(queryloader):
            if use_gpu:
                imgs = imgs.cuda()

            end = time.time()
            features = model(imgs)
            batch_time.update(time.time() - end)

            features = features.data.cpu()
            qf.append(features)
            q_pids.extend(pids)
            q_camids.extend(camids)
        qf = torch.cat(qf, 0)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)

        print('Extracted features for query set, obtained {}-by-{} matrix'.format(qf.size(0), qf.size(1)))

        gf, g_pids, g_camids = [], [], []
        for batch_idx, (imgs, pids, camids, _) in enumerate(galleryloader):
            if use_gpu:
                imgs = imgs.cuda()

            end = time.time()
            features = model(imgs)
            batch_time.update(time.time() - end)

            features = features.data.cpu()
            gf.append(features)
            g_pids.extend(pids)
            g_camids.extend(camids)
        gf = torch.cat(gf, 0)
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)

        print('Extracted features for gallery set, obtained {}-by-{} matrix'.format(gf.size(0), gf.size(1)))

    print('=> BatchTime(s)/BatchSize(img): {:.3f}/{}'.format(batch_time.avg, batch_size))

    m, n = qf.size(0), gf.size(0)
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, qf, gf.t())
    distmat = distmat.numpy()

    print('Computing CMC and mAP')
    # cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids, target_names)
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids, 10)

    print('Results ----------')
    print('mAP: {:.1%}'.format(mAP))
    print('CMC curve')
    for r in ranks:
        print('Rank-{:<3}: {:.1%}'.format(r, cmc[r - 1]))
    print('------------------')
    
    return cmc[0], distmat
def test_rerank(model, queryloader, galleryloader, batch_size, use_gpu, ranks=[1, 5, 10], return_distmat=False):
    batch_time = AverageMeter()

    model.eval()

    with torch.no_grad():
        qf, q_pids, q_camids = [], [], []
        for batch_idx, (imgs, pids, camids, _) in enumerate(queryloader):
            if use_gpu:
                imgs = imgs.cuda()

            end = time.time()
            features = model(imgs)
            batch_time.update(time.time() - end)

            features = features.data.cpu()
            qf.append(features)
            q_pids.extend(pids)
            q_camids.extend(camids)
        qf = torch.cat(qf, 0)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)

        print('Extracted features for query set, obtained {}-by-{} matrix'.format(qf.size(0), qf.size(1)))

        gf, g_pids, g_camids = [], [], []
        for batch_idx, (imgs, pids, camids, _) in enumerate(galleryloader):
            if use_gpu:
                imgs = imgs.cuda()

            end = time.time()
            features = model(imgs)
            batch_time.update(time.time() - end)

            features = features.data.cpu()
            gf.append(features)
            g_pids.extend(pids)
            g_camids.extend(camids)
        gf = torch.cat(gf, 0)
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)

        print('Extracted features for gallery set, obtained {}-by-{} matrix'.format(gf.size(0), gf.size(1)))

    print('=> BatchTime(s)/BatchSize(img): {:.3f}/{}'.format(batch_time.avg, batch_size))

    m, n = qf.size(0), gf.size(0)
    # distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
    #           torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    # distmat.addmm_(1, -2, qf, gf.t())
    # distmat = distmat.numpy()

    distmat = re_ranking(qf, gf, k1=50, k2=15, lambda_value=0.3)
    print('Computing CMC and mAP')
    # cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids, target_names)
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids, 10)

    print('Results ----------')
    print('mAP: {:.1%}'.format(mAP))
    print('CMC curve')
    for r in ranks:
        print('Rank-{:<3}: {:.1%}'.format(r, cmc[r - 1]))
    print('------------------')
    
    return cmc[0], distmat


def main():
    #GENERAL
    torch.cuda.empty_cache()

    root = "/home/kuru/Desktop/veri-gms-master_noise/"
    train_dir = '/home/kuru/Desktop/veri-gms-master_noise/VeRispan/image_train/'
    source = {'verispan'}
    target = {'verispan'}
    workers = 4
    height = 280
    width  = 280
    train_size = 32
    train_sampler = 'RandomSampler'

    #AUGMENTATION
    random_erase = True
    jitter = True
    aug = True

    #OPTIMIZATION
    opt = 'adam'
    lr = 0.0003
    weight_decay = 5e-4
    momentum = 0.9
    sgd_damp = 0.0
    nesterov = True
    warmup_factor = 0.01
    warmup_method = 'linear'

    #HYPERPARAMETER
    max_epoch = 80
    start = 0
    train_batch_size = 8
    test_batch_size = 100

    #SCHEDULER
    lr_scheduler = 'multi_step'
    stepsize = [30, 60]
    gamma = 0.1

    #LOSS
    margin = 0.3
    num_instances = 4
    lambda_tri = 1

    #MODEL
    #arch = 'resnet101'
    arch='resnet101_ibn_a'
    no_pretrained = False

    #TEST SETTINGS
    load_weights = '/home/kuru/Desktop/veri-gms-master/IBN-Net_pytorch0.4.1/resnet101_ibn_a.pth'
    #load_weights = None
    start_eval = 0
    eval_freq = -1

    #MISC
    use_gpu = True
    print_freq = 10
    seed = 1
    resume = ''
    save_dir = '/home/kuru/Desktop/veri-gms-master_noise/spanningtree_verinoise_101_stride2/'
    gpu_id = 0,1
    vis_rank = True
    query_remove = True
    evaluate = False

    dataset_kwargs = {
        'source_names': source,
        'target_names': target,
        'root': root,
        'height': height,
        'width': width,
        'train_batch_size': train_batch_size,
        'test_batch_size': test_batch_size,
        'train_sampler': train_sampler,
        'random_erase': random_erase,
        'color_jitter': jitter,
        'color_aug': aug
        }
    transform_kwargs = {
        'height': height,
        'width': width,
        'random_erase': random_erase,
        'color_jitter': jitter,
        'color_aug': aug
    }

    optimizer_kwargs = {
        'optim': opt,
        'lr': lr,
        'weight_decay': weight_decay,
        'momentum': momentum,
        'sgd_dampening': sgd_damp,
        'sgd_nesterov': nesterov
        }

    lr_scheduler_kwargs = {
        'lr_scheduler': lr_scheduler,
        'stepsize': stepsize,
        'gamma': gamma
        }
    
    use_gpu = torch.cuda.is_available()
    log_name = 'log_test.txt' if evaluate else 'log_train.txt'
    sys.stdout = Logger(osp.join(save_dir, log_name))
    print('Currently using GPU ', gpu_id)
    cudnn.benchmark = True

    print('Initializing image data manager')
    dataset = init_imgreid_dataset(root='/home/kuru/Desktop/veri-gms-master_noise/', name='verispan')
    train = []
    num_train_pids = 0
    num_train_cams = 0

    print(len( dataset.train))

    for img_path, pid, camid, subid, countid in dataset.train:
        #print(img_path)
        path = img_path[56+6:90+6]
        #print(path)
        folder = path[1:4]
        #print(folder)
        pid += num_train_pids
        newidd=0
        train.append((path, folder, pid, camid,subid,countid))

    num_train_pids += dataset.num_train_pids
    num_train_cams += dataset.num_train_cams

    pid = 0
    pidx = {}
    for img_path, pid, camid, subid, countid in dataset.train:
        path = img_path[56+6:90+6]
        
        folder = path[1:4]
        pidx[folder] = pid
        pid+= 1

    sub=[]
    final=0
    xx=dataset.train
    newids=[]
    print(train[0:2])
    train2={}
    for k in range(0,770):
        for img_path, pid, camid, subid, countid in dataset.train:
            if k==pid:
                newid=final+subid
                sub.append(newid)
                #print(pid,subid,newid)
                newids.append(newid)
                train2[img_path]= newid
                #print(img_path, pid, camid, subid, countid, newid)

                

        final=max(sub)
        #print(final)
    print(len(newids),final)

    #train=train2
    #print(train2)
    train3=[]
    for img_path, pid, camid, subid, countid in dataset.train:
        #print(img_path,pid,train2[img_path])
        path = img_path[56:90+6]
        #print(path)
        folder = path[1:4]
        newid=train2[img_path]
        #print((path, folder, pid, camid, subid, countid,newid ))
        train3.append((path, folder, pid, camid, subid, countid,newid ))

    train = train3

    
    path = '/home/kuru/Desktop/adhi/veri-final-draft-master_noise/gmsNoise776/'
    pkl = {}
    #pkl[0] = pickle.load('/home/kuru/Desktop/veri-gms-master/gms/620.pkl')

    entries = os.listdir(path)
    for name in entries:
        f = open((path+name), 'rb')
        ccc=(path+name)
        #print(ccc)
        if name=='featureMatrix.pkl':
            s = name[0:13]
        else:
            s = name[0:3]
        #print(s)
        #with open (ccc,"rb") as ff:
        #    pkl[s] = pickle.load(ff)
            #print(pkl[s])
        pkl[s] = pickle.load(f)
        f.close
        #print(len(pkl))

    with open('cids.pkl', 'rb') as handle:
        b = pickle.load(handle)
        #print(b)

    with open('index.pkl', 'rb') as handle:
        c = pickle.load(handle)



    transform_t = train_transforms(**transform_kwargs)

    data_tfr = vdspan(pkl_file='index_veryspan_noise.pkl', dataset = train, root_dir='/home/kuru/Desktop/veri-gms-master_noise/VeRispan/image_train/', transform=transform_t)
    print("lllllllllllllllllllllllllllllllllllllllllllline 433")
    df2=[]
    data_tfr_old=data_tfr
    for (img,label,index,pid, cid,subid,countid,newid) in data_tfr :
        #print((img,label,index,pid, cid,subid,countid,newid) )
        #print("datframe",(label))
        #print(countid)
        if countid > 4 :
            #print(countid)
            df2.append((img,label,index,pid, cid,subid,countid,newid))
    print("filtered final trainset length",len(df2))
    
    data_tfr=df2
    
    
    
    
    trainloader = DataLoader(data_tfr, sampler=None,batch_size=train_batch_size, shuffle=True, num_workers=workers,pin_memory=True, drop_last=True)

    #data_tfr = vd(pkl_file='index.pkl', dataset = train, root_dir=train_dir,transform=transforms.Compose([Rescale(64),RandomCrop(32),ToTensor()]))
    #dataloader = DataLoader(data_tfr, batch_size=batch_size, shuffle=False, num_workers=0)

    for batch_idx, (img,label,index,pid, cid,subid,countid,newid) in enumerate(trainloader):
        #print("trainloader",batch_idx, (label,index,pid, cid,subid,countid,newid))
        print("trainloader",batch_idx, (label))
        break

    print('Initializing test data manager')
    dm = ImageDataManager(use_gpu, **dataset_kwargs)
    testloader_dict = dm.return_dataloaders()

    print('Initializing model: {}'.format(arch))
    model = models.init_model(name=arch, num_classes=num_train_pids, loss={'xent', 'htri'},
                              pretrained=not no_pretrained, last_stride =2 )
    print('Model size: {:.3f} M'.format(count_num_param(model)))

    if load_weights is not None:
        print("weights loaded")
        load_pretrained_weights(model, load_weights)

    print(torch.cuda.device_count())
    model = nn.DataParallel(model).cuda() if use_gpu else model
    optimizer = init_optimizer(model, **optimizer_kwargs)
    #optimizer = init_optimizer(model)
    
    scheduler = init_lr_scheduler(optimizer, **lr_scheduler_kwargs)

    criterion_xent = CrossEntropyLoss(num_classes=num_train_pids, use_gpu=use_gpu, label_smooth=True)
    criterion_htri = TripletLoss(margin=margin)
    ranking_loss = nn.MarginRankingLoss(margin = margin)

    if evaluate:
        print('Evaluate only')

        for name in target:
            print('Evaluating {} ...'.format(name))
            queryloader = testloader_dict[name]['query']
            galleryloader = testloader_dict[name]['gallery']
            _, distmat = test(model, queryloader, galleryloader, train_batch_size, use_gpu, return_distmat=True)

            if vis_rank:
                visualize_ranked_results(
                    distmat, dm.return_testdataset_by_name(name),
                    save_dir=osp.join(save_dir, 'ranked_results', name),
                    topk=20
                )
        return    

    time_start = time.time()
    ranklogger = RankLogger(source, target)
    print('=> Start training')

    data_index = search(pkl)
    print(len(data_index))
    
    for epoch in range(start, max_epoch):
        losses = AverageMeter()
        #xent_losses = AverageMeter()
        htri_losses = AverageMeter()
        accs = AverageMeter()
        batch_time = AverageMeter()
        xent_losses=AverageMeter()

        model.train()
        for p in model.parameters():
            p.requires_grad = True    # open all layers

        end = time.time()
        for batch_idx,  (img,label,index,pid, cid,subid,countid,newid)  in enumerate(trainloader):
            trainX, trainY = torch.zeros((train_batch_size*3,3,height, width), dtype=torch.float32), torch.zeros((train_batch_size*3), dtype = torch.int64)
            #pids = torch.zeros((batch_size*3), dtype = torch.int16)
            for i in range(train_batch_size):
                #print("dfdsfs")
                labelx = label[i]
                indexx = index[i]
                cidx = pid[i]
                if indexx >len(pkl[labelx])-1:
                    indexx = len(pkl[labelx])-1

                #maxx = np.argmax(pkl[labelx][indexx])
                a = pkl[labelx][indexx]
                minpos = np.argmin(ma.masked_where(a==0, a)) 
                #print(minpos)
                #print(np.array(data_index).shape)
                #print(data_index[cidx][1])
                pos_dic = data_tfr_old[data_index[cidx][1]+minpos]

                neg_label = int(labelx)
                while True:
                    neg_label = random.choice(range(1, 770))
                    #print(neg_label)
                    if neg_label is not int(labelx) and os.path.isdir(os.path.join('/home/kuru/Desktop/adiusb/veri-split/train', strint(neg_label))) is True:
                        break
                negative_label = strint(neg_label)
                neg_cid = pidx[negative_label]
                neg_index = random.choice(range(0, len(pkl[negative_label])))

                neg_dic = data_tfr_old[data_index[neg_cid][1]+neg_index]
                trainX[i] = img[i]
                trainX[i+train_batch_size] = pos_dic[0]
                trainX[i+(train_batch_size*2)] = neg_dic[0]
                trainY[i] = cidx
                trainY[i+train_batch_size] = pos_dic[3]
                trainY[i+(train_batch_size*2)] = neg_dic[3]
            
            trainX = trainX.cuda()
            trainY = trainY.cuda()
            outputs, features = model(trainX)
            xent_loss = criterion_xent(outputs[0:train_batch_size], trainY[0:train_batch_size])
            htri_loss = criterion_htri(features, trainY)

            #tri_loss = ranking_loss(features)
            #ent_loss = xent_loss(outputs[0:batch_size], trainY[0:batch_size], num_train_pids)
            
            loss = htri_loss+xent_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            losses.update(loss.item(), trainY.size(0))
            htri_losses.update(htri_loss.item(), trainY.size(0))
            xent_losses.update(xent_loss.item(), trainY.size(0))
            accs.update(accuracy(outputs[0:train_batch_size], trainY[0:train_batch_size])[0])
    
            if (batch_idx) % 50 == 0:
                print('Train ', end=" ")
                print('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'TriLoss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'XLoss {xloss.val:.4f} ({xloss.avg:.4f})\t'
                    'OveralLoss {oloss.val:.4f} ({oloss.avg:.4f})\t'
                    'Acc {acc.val:.2f} ({acc.avg:.2f})\t'
                    'lr {lrrr} \t'.format(
                    epoch + 1, batch_idx + 1, len(trainloader),
                    batch_time=batch_time,
                    loss = htri_losses,
                    xloss = xent_losses,
                    oloss = losses,
                    acc=accs ,
                    lrrr=lrrr,
                ))
                

            end = time.time()

        
        scheduler.step()            
        print('=> Test')

        for name in target:
            print('Evaluating {} ...'.format(name))
            queryloader = testloader_dict[name]['query']
            galleryloader = testloader_dict[name]['gallery']
            rank1, distmat = test(model, queryloader, galleryloader, test_batch_size, use_gpu)
            ranklogger.write(name, epoch + 1, rank1)
            rank2, distmat2 = test_rerank(model, queryloader, galleryloader, test_batch_size, use_gpu)
            ranklogger.write(name, epoch + 1, rank2)
            
        #if (epoch + 1) == max_epoch:
        if (epoch + 1) % 2 == 0:
            print('=> Test')

            for name in target:
                print('Evaluating {} ...'.format(name))
                queryloader = testloader_dict[name]['query']
                galleryloader = testloader_dict[name]['gallery']
                rank1, distmat = test(model, queryloader, galleryloader, test_batch_size, use_gpu)
                ranklogger.write(name, epoch + 1, rank1)

                # if vis_rank:
                #     visualize_ranked_results(
                #         distmat, dm.return_testdataset_by_name(name),
                #         save_dir=osp.join(save_dir, 'ranked_results', name),
                #         topk=20)

            save_checkpoint({
                'state_dict': model.state_dict(),
                'rank1': rank1,
                'epoch': epoch + 1,
                'arch': arch,
                'optimizer': optimizer.state_dict(),
            }, save_dir)
    

if __name__ == '__main__':
    main()