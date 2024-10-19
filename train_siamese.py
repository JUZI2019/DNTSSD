import torch
import torch.nn as nn
import os
import numpy as np
import loss
import cv2
import func_utils
from tensorboardX import SummaryWriter
# from torch.utils.tensorboard import SummaryWriter 
from torchvision.utils import make_grid


def collater(data):
    out_data_dict = {}
    for name in data[0]:
        out_data_dict[name] = []
    for sample in data:
        for name in sample:
            out_data_dict[name].append(torch.from_numpy(sample[name]))
    for name in out_data_dict:
        out_data_dict[name] = torch.stack(out_data_dict[name], dim=0)
    return out_data_dict

class TrainModule(object):
    def __init__(self, dataset, num_classes, model, decoder, down_ratio):
        torch.manual_seed(317)
        self.dataset = dataset
        self.dataset_phase = {'dota': ['train','test'],
                              'dota_ship': ['train','test'],
                              'dota_ship_siamese':['train','test'],
                            #   'dota_ship_siamese':['train','test','val'],
                              'hrsc': ['train', 'test']}
        self.num_classes = num_classes
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.decoder = decoder
        self.down_ratio = down_ratio
        self.writer = SummaryWriter('log')


    def save_model(self, path, epoch, model, optimizer):
        if isinstance(model, torch.nn.DataParallel):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()
        torch.save({
            'epoch': epoch,
            'model_state_dict': state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            # 'loss': loss
        }, path)

    def load_model(self, model, optimizer, resume, strict=True):
        checkpoint = torch.load(resume, map_location=lambda storage, loc: storage)
        print('loaded weights from {}, epoch {}'.format(resume, checkpoint['epoch']))
        state_dict_ = checkpoint['model_state_dict']
        state_dict = {}
        for k in state_dict_:
            if k.startswith('module') and not k.startswith('module_list'):
                state_dict[k[7:]] = state_dict_[k]
            else:
                state_dict[k] = state_dict_[k]
        model_state_dict = model.state_dict()
        if not strict:
            for k in state_dict:
                if k in model_state_dict:
                    if state_dict[k].shape != model_state_dict[k].shape:
                        print('Skip loading parameter {}, required shape{}, ' \
                              'loaded shape{}.'.format(k, model_state_dict[k].shape, state_dict[k].shape))
                        state_dict[k] = model_state_dict[k]
                else:
                    print('Drop parameter {}.'.format(k))
            for k in model_state_dict:
                if not (k in state_dict):
                    print('No param {}.'.format(k))
                    state_dict[k] = model_state_dict[k]
        model.load_state_dict(state_dict, strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
        epoch = checkpoint['epoch']
        # loss = checkpoint['loss']
        return model, optimizer, epoch

    def train_network(self, args):
        self.optimizer = torch.optim.Adam(self.model.parameters(), args.init_lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.96, last_epoch=-1)

        # self.optimizer = torch.optim.AdamW(self.model.parameters(), args.init_lr, weight_decay=1e-4)
        # self.optimizer = torch.optim.AdamW(self.model.parameters(), args.init_lr)
        # self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.96, last_epoch=-1)

        start_epoch = 1
        
        # add resume part for continuing training when break previously, 10-16-2020
        if args.resume_train:
            self.model, self.optimizer, start_epoch = self.load_model(self.model, 
                                                                        self.optimizer, 
                                                                        args.resume_train, 
                                                                        strict=True)
        # end 
        # if not os.path.exists(os.path.join(args.data_dir,'output')):
        #     os.mkdir(os.path.join(args.data_dir,'output'))
        # data_dir_name = args.data_dir.split('/')[-1]

        # save_file = args.data_dir.split('/')[-2]+'_'+args.data_dir.split('/')[-1]
        # save_path = os.path.join(args.save_dir, args.data_dir.split('/')[-2]+'_'+args.data_dir.split('/')[-1])
        save_path = os.path.join(args.save_dir, args.data_dir.split('/')[-1]+'_module'+str(args.module))
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        if args.ngpus>1:
            if torch.cuda.device_count() > 1:
                print("Let's use", torch.cuda.device_count(), "GPUs!")
                # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
                self.model = nn.DataParallel(self.model)
        self.model.to(self.device)

        criterion = loss.LossAll()
        print('Setting up data...')

        dataset_module = self.dataset[args.dataset]

        dsets = {x: dataset_module(data_dir=args.data_dir,
                                   phase=x,
                                   input_h=args.input_h,
                                   input_w=args.input_w,
                                   down_ratio=self.down_ratio)
                 for x in self.dataset_phase[args.dataset]}

        dsets_loader = {}
        dsets_loader['train'] = torch.utils.data.DataLoader(dsets['train'],
                                                           batch_size=args.batch_size,
                                                           shuffle=True,
                                                           num_workers=args.num_workers,
                                                           pin_memory=True,
                                                           drop_last=True,
                                                           collate_fn=collater)

        print('Starting training...')
        train_loss = []
        ap_list = []

        for epoch in range(start_epoch, args.num_epoch+1):
            print('-'*10)
            print('Epoch: {}/{} '.format(epoch, args.num_epoch))
            epoch_loss = self.run_epoch(phase='train',
                                        data_loader=dsets_loader['train'],
                                        criterion=criterion,
                                        epoch=epoch)
            train_loss.append(epoch_loss)
            # self.scheduler.step(epoch) 旧版本被淘汰的写法
            self.scheduler.step()

            np.savetxt(os.path.join(save_path, 'train_loss.txt'), train_loss, fmt='%.6f')

            # if epoch % 5 == 0:
            #     self.save_model(os.path.join(save_path, 'model_{}.pth'.format(epoch)),
            #                     epoch,
            #                     self.model,
            #                     self.optimizer)

            self.save_model(os.path.join(save_path, 'model_{}.pth'.format(epoch)),
                epoch,
                self.model,
                self.optimizer)

            # if 'val' in self.dataset_phase[args.dataset]:
            #     mAP = self.dec_eval(args, 'val', dsets['val'], epoch)
            #     val_ap_list.append(mAP)
            #     # ap_list = np.array(ap_list, dtype=float)
            #     np.savetxt(os.path.join(save_path, 'val_ap_list.txt'), val_ap_list, fmt='%.6f')

            # 每5次test一次
            if 'test' in self.dataset_phase[args.dataset] :
                mAP = self.dec_eval(args, dsets['test'], epoch)
                ap_list.append(mAP)
                # ap_list = np.array(ap_list, dtype=float)
                np.savetxt(os.path.join(save_path, 'ap_list.txt'), ap_list, fmt='%.6f')

            # self.save_model(os.path.join(save_path, 'model_last.pth'),
            #                 epoch,
            #                 self.model,
            #                 self.optimizer)

    def run_epoch(self, phase, data_loader, criterion, epoch):
        if phase == 'train':
            self.model.train()
        else:
            self.model.eval()

            # #下面这段sy加的 //
            # for i, data in enumerate(data_loader, 0):
            #     # 获取训练数据
            #     inputs = data['input']
            #     inputs = inputs.to(self.device)
            #     x = inputs
            #     # x = inputs[0].unsqueeze(0)  # x 在这里呀
            #     break
            # self.writer.add_image('image',x[0], global_step=0)
            # for name, layer in self.model._modules.items():
            #     # 查看卷积层的特征图
            #     if 'base_network' in name:
            #         for thename, thelayer in layer._modules.items():
            #             if 'layer' in thename or 'conv' in thename:
            #                 # x = x.view(x.size(0), -1) if 'fc' in name else x
            #                 # x = thelayer(x)
            #                 # x1 = x.transpose(0, 1)  # C，B, H, W ---> B，C, H, W
            #                 x = thelayer(x)
            #                 x1 = x[0].detach().cpu().unsqueeze(dim=1)
            #                 img_grid = make_grid(x1, normalize=False, scale_each=False, nrow=5)
            #                 self.writer.add_image(f'{thename}_feature_maps', img_grid, global_step=0)
            #     else:
            #         x = layer(x)  # //


        running_loss = 0.
        for i, data_dict in enumerate(data_loader):
            for name in data_dict:
                data_dict[name] = data_dict[name].to(device=self.device, non_blocking=True)
            if phase == 'train':
                self.optimizer.zero_grad()
                with torch.enable_grad():
                    pr_decs = self.model(data_dict['input'], data_dict['bginput'])
                    loss = criterion(pr_decs, data_dict)
                    loss.backward()
                    self.optimizer.step()
            else:
                with torch.no_grad():
                    pr_decs = self.model(data_dict['input'], data_dict['bginput'])
                    loss = criterion(pr_decs, data_dict)

            running_loss += loss.item()
            # 每100个batch显示一次当前的loss
            if i % 10 == 9:
                print('[epoch: %d, batch: %5d] loss: %.3f' %
                      (epoch, i + 1, loss.item()))
            # 每100个batch画个点用于loss曲线
            if i % 100 == 0:
                niter = (epoch - 1) * len(data_loader) + i
                self.writer.add_scalar('Train/Loss', loss.item(), niter)
        epoch_loss = running_loss / len(data_loader)
        print('{} loss: {}'.format(phase, epoch_loss))
        return epoch_loss

        # for data_dict in data_loader:
        #     for name in data_dict:
        #         data_dict[name] = data_dict[name].to(device=self.device, non_blocking=True)
        #     if phase == 'train':
        #         self.optimizer.zero_grad()
        #         with torch.enable_grad():
        #             pr_decs = self.model(data_dict['input'])

        #             #(pr_decs, gt_batch)
        #             loss = criterion(pr_decs, data_dict)
        #             loss.backward()
        #             self.optimizer.step()
        #     else:
        #         with torch.no_grad():
        #             pr_decs = self.model(data_dict['input'])
        #             loss = criterion(pr_decs, data_dict)

        #     running_loss += loss.item()
            
        # epoch_loss = running_loss / len(data_loader)
        # print('{} loss: {}'.format(phase, epoch_loss))
        # return epoch_loss


    def dec_eval(self, args, dsets, epoch):
        # result_path = 'result_'+args.dataset
        result_path = os.path.join(args.save_dir, args.data_dir.split('/')[-1]+'_module'+str(args.module), 'evalTxt_epoch_'+str(epoch)) 
        # result_path = os.path.join(args.save_dir, args.data_dir.split('/')[-2]+'_'+args.data_dir.split('/')[-1], 'evalTxt_epoch_'+str(epoch)) 

        if not os.path.exists(result_path):
            os.mkdir(result_path)

        self.model.eval()
        func_utils.write_results_siamese(args,
                                 self.model,
                                 dsets,
                                 self.down_ratio,
                                 self.device,
                                 self.decoder,
                                 result_path)
        
        if 'dota' in args.dataset:
            merge_path = os.path.join(args.save_dir, args.data_dir.split('/')[-1]+'_module'+str(args.module), 'evalTxt_merge_epoch_'+str(epoch))
            # merge_path = os.path.join(args.save_dir, args.data_dir.split('/')[-2]+'_'+args.data_dir.split('/')[-1], 'evalTxt_merge_epoch_'+str(epoch))

            if not os.path.exists(merge_path):
                os.mkdir(merge_path)
            
            dsets.merge_crop_image_results(result_path, merge_path)
            ap = dsets.dec_evaluation(merge_path)
            return ap
        else:
            ap = dsets.dec_evaluation(result_path)
            return ap

        # ap = dsets.dec_evaluation(result_path)
        # return ap