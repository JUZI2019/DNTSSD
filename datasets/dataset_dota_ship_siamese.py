from matplotlib import pyplot as plt
from .base import BaseDataset
import os
import cv2
import numpy as np
from DOTA_devkit.ResultMerge_multi_process import mergebypoly_multiprocess
from DOTA_devkit.dota_evaluation_task1 import voc_eval, voc_eval_inoffshore
import math
from . import data_augment

from .draw_gaussian import draw_umich_gaussian, gaussian_radius
from .transforms import load_affine_matrix, random_crop_info, ex_box_jaccard
from glob import glob
import csv

def random_flip(image, bgimg, gt_pts, crop_center=None):
    # image: h x w x c
    # gt_pts: num_obj x 4 x 2
    h,w,c = image.shape
    if np.random.random()<0.5:
        image = image[:,::-1,:]
        bgimg = bgimg[:,::-1,:]

        if gt_pts.shape[0]:
            gt_pts[:,:,0] = w-1 - gt_pts[:,:,0]
        if crop_center is not None:
            crop_center[0] = w-1 - crop_center[0]
    if np.random.random()<0.5:
        image = image[::-1,:,:]
        bgimg = bgimg[::-1,:,:]
        
        if gt_pts.shape[0]:
            gt_pts[:,:,1] = h-1 - gt_pts[:,:,1]
        if crop_center is not None:
            crop_center[1] = h-1 - crop_center[1]
    return image, bgimg, gt_pts, crop_center

class DOTA_SHIP_SIAMESE(BaseDataset):
    def __init__(self, data_dir, phase, input_h=None, input_w=None, down_ratio=None):
        super(DOTA_SHIP_SIAMESE, self).__init__(data_dir, phase, input_h, input_w, down_ratio)
        self.polarize = 'vh'
        self.category = [
                         'ship',
                        # 'ship_inshore',
                        # 'ship_offshore'
        ]
        
        self.color_pans = [
                           (255,255,0)]
        self.num_classes = len(self.category)
        self.cat_ids = {cat:i for i,cat in enumerate(self.category)}
        self.img_ids = self.load_img_ids()
        self.image_distort =  data_augment.PhotometricDistort2()

    
        # 版本3
        # TODO:分开VH和BG的图像加载路径（包括路径和加载函数)
        # 注意：这里图像的路径和标签改成字典(如果vh和vv同时使用的话可能需要改这里)

        # self.image_path = {'vh':os.path.join(self.data_dir, 'vh', self.phase, 'images'),
        #                    'vv':os.path.join(self.data_dir, 'vv', self.phase, 'images')}
        # self.bg_path = {'vh':os.path.join(self.data_dir, 'vhbg', 'images'),
        #                    'vv':os.path.join(self.data_dir, 'vvbg','images')}        
        # self.label_path = {'vh':os.path.join(self.data_dir, 'vh', self.phase, 'labelTxt'),
        #                    'vv':os.path.join(self.data_dir, 'vv', self.phase, 'labelTxt')}
        

        self.image_path = {'vh':os.path.join(self.data_dir, 'vh', 'inshore_'+self.phase, 'images'),
                            'vv':os.path.join(self.data_dir, 'vv', 'inshore_'+self.phase, 'images')}        
        self.bg_path = {'vh':os.path.join(self.data_dir, 'vhbg', 'images'),
                           'vv':os.path.join(self.data_dir, 'vvbg','images')}        
        self.label_path = {'vh':os.path.join(self.data_dir, 'vh', 'inshore_'+self.phase, 'labelTxt'),
                           'vv':os.path.join(self.data_dir, 'vv', 'inshore_'+self.phase, 'labelTxt')}
        

        # self.image_path = {'vh':os.path.join(self.data_dir, 'vh', 'offshore_'+self.phase, 'images'),
        #                     'vv':os.path.join(self.data_dir, 'vv', 'offshore_'+self.phase, 'images')}
        # self.bg_path = {'vh':os.path.join(self.data_dir, 'vhbg', 'images'),
        #                    'vv':os.path.join(self.data_dir, 'vvbg','images')}        
        # self.label_path = {'vh':os.path.join(self.data_dir, 'vh', 'offshore_'+self.phase, 'labelTxt'),
        #                    'vv':os.path.join(self.data_dir, 'vv', 'offshore_'+self.phase, 'labelTxt')}



    def load_img_ids(self):

        # # all
        # image_set_index_file = os.path.join(self.data_dir, self.polarize, self.phase, self.phase + '.txt')

        # #  inshore
        image_set_index_file = os.path.join(self.data_dir, self.polarize, 'inshore_'+self.phase, self.phase + '.txt')

        # offshore
        # image_set_index_file = os.path.join(self.data_dir, self.polarize, 'offshore_'+self.phase, self.phase + '.txt')

        # assert os.path.exists(image_set_index_file), 'Path does not exist: {}'.format(image_set_index_file)
        with open(image_set_index_file, 'r') as f:
            lines = f.readlines()
        image_lists = [line.strip() for line in lines]
        return image_lists

    def load_image(self, index):
        img_id = self.img_ids[index]
        imgFile = os.path.join(self.image_path[self.polarize], img_id+'.jpg')
        assert os.path.exists(imgFile), 'image {} not existed'.format(imgFile)
        img = cv2.imread(imgFile)
        return img
    
    def load_image_next(self, index):
        img_id = self.img_ids[index]
        bgi = img_id.split('_')
        # 查询下一个时序的影像
        bgimgFile = glob(self.image_path[self.polarize]+'/'+bgi[0]+'*'+bgi[3]+'*'+bgi[5]+'*'+bgi[8]+'.jpg')[0]
        assert os.path.exists(bgimgFile), 'image {} not existed'.format(bgimgFile)
        bgimg = cv2.imread(bgimgFile)
        return bgimg


    def load_bgimg(self, index):
        img_id = self.img_ids[index]
        bgi = img_id.split('_')
        # 查询对应背景
        bgimgFile = glob(self.bg_path[self.polarize]+'/'+bgi[0]+'*'+bgi[3]+'*'+bgi[5]+'*'+bgi[8]+'.jpg')[0]
        assert os.path.exists(bgimgFile), 'image {} not existed'.format(bgimgFile)
        bgimg = cv2.imread(bgimgFile)
        return bgimg


    def load_annoFolder(self, img_id):
        return os.path.join(self.label_path[self.polarize], img_id+'.txt')

    def load_annotation(self, index):
        image = self.load_image(index)
        h,w,c = image.shape
        valid_pts = []
        valid_cat = []
        valid_dif = []
        with open(self.load_annoFolder(self.img_ids[index]), 'r') as f:
            for i, line in enumerate(f.readlines()):
                obj = line.split(' ')  # list object
                if len(obj)>8:
                    x1 = min(max(float(obj[0]), 0), w - 1)
                    y1 = min(max(float(obj[1]), 0), h - 1)
                    x2 = min(max(float(obj[2]), 0), w - 1)
                    y2 = min(max(float(obj[3]), 0), h - 1)
                    x3 = min(max(float(obj[4]), 0), w - 1)
                    y3 = min(max(float(obj[5]), 0), h - 1)
                    x4 = min(max(float(obj[6]), 0), w - 1)
                    y4 = min(max(float(obj[7]), 0), h - 1)
                    # TODO: filter small instances
                    xmin = max(min(x1, x2, x3, x4), 0)
                    xmax = max(x1, x2, x3, x4)
                    ymin = max(min(y1, y2, y3, y4), 0)
                    ymax = max(y1, y2, y3, y4)
                    if ((xmax - xmin) > 10) and ((ymax - ymin) > 10):
                        valid_pts.append([[x1,y1], [x2,y2], [x3,y3], [x4,y4]])
                        if self.category==['ship']:
                            valid_cat.append(self.cat_ids[obj[8].split('_')[0]])  # ship
                        else: 
                            valid_cat.append(self.cat_ids[obj[8]])   # ship_offshore / ship_inshore
                        valid_dif.append(int(obj[9]))
        f.close()
        annotation = {}
        annotation['pts'] = np.asarray(valid_pts, np.float32)
        annotation['cat'] = np.asarray(valid_cat, np.int32)
        annotation['dif'] = np.asarray(valid_dif, np.int32)
        # pts0 = np.asarray(valid_pts, np.float32)
        # img = self.load_image(index)
        # for i in range(pts0.shape[0]):
        #     pt = pts0[i, :, :]
        #     tl = pt[0, :]
        #     tr = pt[1, :]
        #     br = pt[2, :]
        #     bl = pt[3, :]
        #     cv2.line(img, (int(tl[0]), int(tl[1])), (int(tr[0]), int(tr[1])), (0, 0, 255), 1, 1)
        #     cv2.line(img, (int(tr[0]), int(tr[1])), (int(br[0]), int(br[1])), (255, 0, 255), 1, 1)
        #     cv2.line(img, (int(br[0]), int(br[1])), (int(bl[0]), int(bl[1])), (0, 255, 255), 1, 1)
        #     cv2.line(img, (int(bl[0]), int(bl[1])), (int(tl[0]), int(tl[1])), (255, 0, 0), 1, 1)
        #     cv2.putText(img, '{}:{}'.format(valid_dif[i], self.category[valid_cat[i]]), (int(tl[0]), int(tl[1])), cv2.FONT_HERSHEY_TRIPLEX, 0.6,
        #                 (0, 0, 255), 1, 1)
        # cv2.imshow('img', np.uint8(img))
        # k = cv2.waitKey(0) & 0xFF
        # if k == ord('q'):
        #     cv2.destroyAllWindows()
        #     exit()
        return annotation


    def merge_crop_image_results(self, result_path, merge_path):
        mergebypoly_multiprocess(result_path, merge_path)
 

    # 要补一个验证集上计算mAP的函数
    def dec_evaluation_inoffshore(self, merge_path):
        detpath = os.path.join(merge_path, 'Task1_{}.txt')
        # annopath = os.path.join(self.label_path, '{}.txt')
        # imagesetfile = os.path.join(self.data_dir, 'test', 'test.txt')

        # 使用未切割的test数据进行验证
        annopath = os.path.join('/workstation/fyy/sen1ship_dota_vhbg/vh/test/labelTxt', '{}.txt')  
        imagesetfile = os.path.join('/workstation/fyy/sen1ship_dota_vhbg/vh/test', 'test.txt')

        classaps = []
        map = 0
        categories = ["ship_inshore", "ship_offshore"]
        # for classname in self.category:
        for classname in self.category:
            if classname == 'background':
                continue
            print('classname:', classname)
            rec, prec, ap = voc_eval_inoffshore(detpath,
                                     annopath,
                                     imagesetfile,
                                     categories,
                                     ovthresh=0.5,
                                     use_07_metric=True)
            map = map + ap
            # print('rec: ', rec, 'prec: ', prec, 'ap: ', ap)
            print('{}:{} '.format(classname, ap*100))
            classaps.append(ap)
            # # umcomment to show p-r curve of each category
            # plt.figure(figsize=(8,4))
            # plt.xlabel('recall')
            # plt.ylabel('precision')
            # plt.plot(rec, prec)
            # plt.savefig(os.path.join(merge_path,'p-r_curve.png',))
            # plt.close()
        # plt.show()
        map = map / len(self.category)
        print('mAP:', map*100)
        # classaps = 100 * np.array(classaps)
        # print('classaps: ', classaps)
        return map
    

# 要补一个验证集上计算mAP的函数
    def dec_evaluation(self, detPath):
        detpath = os.path.join(detPath, 'Task1_{}.txt')
        # annopath = os.path.join(self.label_path, '{}.txt')
        # imagesetfile = os.path.join(self.data_dir, 'test', 'test.txt')

        # 使用未切割的test数据进行验证
        # annopath = os.path.join('/workstation/fyy/sen1ship_dota_vhbg/vh/test/labelTxt', '{}.txt')  
        # imagesetfile = os.path.join('/workstation/fyy/sen1ship_dota_vhbg/vh/test', 'test.txt')

        # offshore
        # annopath = os.path.join('/workstation/fyy/sen1ship_dota_vhbg_608_single_2/vh/offshore_test/labelTxt', '{}.txt')  
        # imagesetfile = os.path.join('/workstation/fyy/sen1ship_dota_vhbg_608_single_2/vh/offshore_test', 'test.txt')


        # inshore
        annopath = os.path.join('/workstation/fyy/sen1ship_dota_vhbg_608_single_2/vh/inshore_test/labelTxt', '{}.txt')  
        imagesetfile = os.path.join('/workstation/fyy/sen1ship_dota_vhbg_608_single_2/vh/inshore_test', 'test.txt')


        classaps = []
        map = 0
        # for classname in self.category:
        for classname in self.category:
            if classname == 'background':
                continue
            print('classname:', classname)
            rec, prec, ap = voc_eval(detpath,
                                     annopath,
                                     imagesetfile,
                                     classname,
                                     ovthresh=0.5,
                                     use_07_metric=True)
            map = map + ap
            recall = rec[-1]
            precision = prec[-1]
            f1 = 2 * (precision * recall) / (precision + recall)
            print('recall: ', recall, 'precision: ', precision, 'f1: ', f1, 'ap: ', ap)
            print('{}:{} '.format(classname, ap*100))
            # 保存 precision 和 recall 到 CSV 文件
            csv_filename = os.path.join(detPath, 'precision_recall.csv')
            with open(csv_filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['recall', 'precision'])
                for r, p in zip(rec, prec):
                    writer.writerow([r, p])
            classaps.append(ap)
            # # umcomment to show p-r curve of each category
            plt.figure(figsize=(8,4))
            plt.xlabel('recall')
            plt.ylabel('precision')
            plt.plot(rec, prec)
            plt.savefig(os.path.join(detPath,'p-r_curve.png',))
            plt.close()
        # plt.show()
        map = map / len(self.category)
        print('mAP:', map*100)
        classaps = 100 * np.array(classaps)
        print('classaps: ', classaps)
        return map    


    def generate_ground_truth(self, image, bgimg, annotation):
        image = np.asarray(np.clip(image, a_min=0., a_max=255.), np.float32)
        bgimg = np.asarray(np.clip(bgimg, a_min=0., a_max=255.), np.float32)

        image, bgimg = self.image_distort(np.asarray(image, np.float32), np.asarray(bgimg, np.float32))

        image = np.asarray(np.clip(image, a_min=0., a_max=255.), np.float32)
        image = np.transpose(image / 255. - 0.5, (2, 0, 1))

        bgimg = np.asarray(np.clip(bgimg, a_min=0., a_max=255.), np.float32)
        bgimg = np.transpose(bgimg / 255. - 0.5, (2, 0, 1))

        image_h = self.input_h // self.down_ratio
        image_w = self.input_w // self.down_ratio

        hm = np.zeros((self.num_classes, image_h, image_w), dtype=np.float32)
        wh = np.zeros((self.max_objs, 10), dtype=np.float32)
        ## add
        cls_theta = np.zeros((self.max_objs, 1), dtype=np.float32)
        ## add end
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
        num_objs = min(annotation['rect'].shape[0], self.max_objs)
        # ###################################### view Images #######################################
        # copy_image1 = cv2.resize(image, (image_w, image_h))
        # copy_image2 = copy_image1.copy()
        # ##########################################################################################
        for k in range(num_objs):
            rect = annotation['rect'][k, :]
            cen_x, cen_y, bbox_w, bbox_h, theta = rect
            # print(theta)

            # 高斯热图（原
            radius = gaussian_radius((math.ceil(bbox_h), math.ceil(bbox_w)))
            radius = max(0, int(radius))
            ct = np.asarray([cen_x, cen_y], dtype=np.float32)
            ct_int = ct.astype(np.int32)
            draw_umich_gaussian(hm[annotation['cat'][k]], ct_int, radius)

            ind[k] = ct_int[1] * image_w + ct_int[0]
            reg[k] = ct - ct_int
            reg_mask[k] = 1
            # generate wh ground_truth
            pts_4 = cv2.boxPoints(((cen_x, cen_y), (bbox_w, bbox_h), theta))  # 4 x 2

            bl = pts_4[0,:]
            tl = pts_4[1,:]
            tr = pts_4[2,:]
            br = pts_4[3,:]

            tt = (np.asarray(tl,np.float32)+np.asarray(tr,np.float32))/2
            rr = (np.asarray(tr,np.float32)+np.asarray(br,np.float32))/2
            bb = (np.asarray(bl,np.float32)+np.asarray(br,np.float32))/2
            ll = (np.asarray(tl,np.float32)+np.asarray(bl,np.float32))/2

            if theta in [-90.0, -0.0, 0.0]:  # (-90, 0]
                tt,rr,bb,ll = self.reorder_pts(tt,rr,bb,ll)
            # rotational channel
            wh[k, 0:2] = tt - ct
            wh[k, 2:4] = rr - ct
            wh[k, 4:6] = bb - ct
            wh[k, 6:8] = ll - ct
            #####################################################################################
            # # draw
            # cv2.line(copy_image1, (cen_x, cen_y), (int(tt[0]), int(tt[1])), (0, 0, 255), 1, 1)
            # cv2.line(copy_image1, (cen_x, cen_y), (int(rr[0]), int(rr[1])), (255, 0, 255), 1, 1)
            # cv2.line(copy_image1, (cen_x, cen_y), (int(bb[0]), int(bb[1])), (0, 255, 255), 1, 1)
            # cv2.line(copy_image1, (cen_x, cen_y), (int(ll[0]), int(ll[1])), (255, 0, 0), 1, 1)
            #####################################################################################
            # horizontal channel
            w_hbbox, h_hbbox = self.cal_bbox_wh(pts_4)
            wh[k, 8:10] = 1. * w_hbbox, 1. * h_hbbox
            #####################################################################################
            # # draw
            # cv2.line(copy_image2, (cen_x, cen_y), (int(cen_x), int(cen_y-wh[k, 9]/2)), (0, 0, 255), 1, 1)
            # cv2.line(copy_image2, (cen_x, cen_y), (int(cen_x+wh[k, 8]/2), int(cen_y)), (255, 0, 255), 1, 1)
            # cv2.line(copy_image2, (cen_x, cen_y), (int(cen_x), int(cen_y+wh[k, 9]/2)), (0, 255, 255), 1, 1)
            # cv2.line(copy_image2, (cen_x, cen_y), (int(cen_x-wh[k, 8]/2), int(cen_y)), (255, 0, 0), 1, 1)
            #####################################################################################
            # v0
            # if abs(theta)>3 and abs(theta)<90-3:
            #     cls_theta[k, 0] = 1
            # v1
            jaccard_score = ex_box_jaccard(pts_4.copy(), self.cal_bbox_pts(pts_4).copy())
            if jaccard_score<0.95:
                cls_theta[k, 0] = 1
        # ###################################### view Images #####################################
        # # hm_show = np.uint8(cv2.applyColorMap(np.uint8(hm[0, :, :] * 255), cv2.COLORMAP_JET))
        # # copy_image = cv2.addWeighted(np.uint8(copy_image), 0.4, hm_show, 0.8, 0)
        #     if jaccard_score>0.95:
        #         print(theta, jaccard_score, cls_theta[k, 0])
        #         cv2.imshow('img1', cv2.resize(np.uint8(copy_image1), (image_w*4, image_h*4)))
        #         cv2.imshow('img2', cv2.resize(np.uint8(copy_image2), (image_w*4, image_h*4)))
        #         key = cv2.waitKey(0)&0xFF
        #         if key==ord('q'):
        #             cv2.destroyAllWindows()
        #             exit()
        # #########################################################################################

        ret = {'input': image,
               'bginput': bgimg,
               'hm': hm,
               'reg_mask': reg_mask,
               'ind': ind,
               'wh': wh,
               'reg': reg,
               'cls_theta':cls_theta,
               }
        
        # #插入一个显示heatmap图像的函数。。
        # imshow_heatmap(ret, image)

        return ret

    
    def data_transform(self, image, bgimg, annotation):
        # only do random_flip augmentation to original images
        crop_size = None
        crop_center = None
        crop_size, crop_center = random_crop_info(h=image.shape[0], w=image.shape[1])
        image, bgimg, annotation['pts'], crop_center = random_flip(image, bgimg, annotation['pts'], crop_center)

        if crop_center is None:
            crop_center = np.asarray([float(image.shape[1])/2, float(image.shape[0])/2], dtype=np.float32)
        if crop_size is None:
            crop_size = [max(image.shape[1], image.shape[0]), max(image.shape[1], image.shape[0])]  # init
        M = load_affine_matrix(crop_center=crop_center,
                               crop_size=crop_size,
                               dst_size=(self.input_w, self.input_h),
                               inverse=False,
                               rotation=True)
        image = cv2.warpAffine(src=image, M=M, dsize=(self.input_w, self.input_h), flags=cv2.INTER_LINEAR)
        bgimg = cv2.warpAffine(src=bgimg, M=M, dsize=(self.input_w, self.input_h), flags=cv2.INTER_LINEAR)
        # plt.imsave('image-trans.png',image)
        # plt.imsave('bgimg-trans.png',bgimg)

        if annotation['pts'].shape[0]:
            annotation['pts'] = np.concatenate([annotation['pts'], np.ones((annotation['pts'].shape[0], annotation['pts'].shape[1], 1))], axis=2)
            annotation['pts'] = np.matmul(annotation['pts'], np.transpose(M))
            annotation['pts'] = np.asarray(annotation['pts'], np.float32)

        out_annotations = {}
        size_thresh = 3
        out_rects = []
        out_cat = []
        for pt_old, cat in zip(annotation['pts'] , annotation['cat']):
            if (pt_old<0).any() or (pt_old[:,0]>self.input_w-1).any() or (pt_old[:,1]>self.input_h-1).any():
                pt_new = pt_old.copy()
                pt_new[:,0] = np.minimum(np.maximum(pt_new[:,0], 0.), self.input_w - 1)
                pt_new[:,1] = np.minimum(np.maximum(pt_new[:,1], 0.), self.input_h - 1)
                iou = ex_box_jaccard(pt_old.copy(), pt_new.copy())
                if iou>0.6:
                    rect = cv2.minAreaRect(pt_new/self.down_ratio)
                    if rect[1][0]>size_thresh and rect[1][1]>size_thresh:
                        out_rects.append([rect[0][0], rect[0][1], rect[1][0], rect[1][1], rect[2]])
                        out_cat.append(cat)
            else:
                rect = cv2.minAreaRect(pt_old/self.down_ratio)
                if rect[1][0]<size_thresh and rect[1][1]<size_thresh:
                    continue
                out_rects.append([rect[0][0], rect[0][1], rect[1][0], rect[1][1], rect[2]])
                out_cat.append(cat)
        out_annotations['rect'] = np.asarray(out_rects, np.float32)
        out_annotations['cat'] = np.asarray(out_cat, np.uint8)
        return image, bgimg, out_annotations
    

    def processing_test_withbg(self, image, bgimg, input_h, input_w):
        import torch

        image = cv2.resize(image, (input_w, input_h))
        bgimg = cv2.resize(bgimg, (input_w, input_h))

        out_image = image.astype(np.float32) / 255.
        out_image = out_image - 0.5
        out_image = out_image.transpose(2, 0, 1).reshape(1, 3, input_h, input_w)
        out_image = torch.from_numpy(out_image)

        out_bgimage = bgimg.astype(np.float32) / 255.
        out_bgimage = out_bgimage - 0.5
        out_bgimage = out_bgimage.transpose(2, 0, 1).reshape(1, 3, input_h, input_w)
        out_bgimage = torch.from_numpy(out_bgimage)
        return out_image, out_bgimage


    def __getitem__(self, index):
        image = self.load_image(index)
        bgimg = self.load_bgimg(index)
        image_h, image_w, c = image.shape

        #  测试的图像
        if self.phase == 'test':
            img_id = self.img_ids[index]
            image,bgimg = self.processing_test_withbg(image, bgimg, self.input_h, self.input_w)
            return {'image': image,
                    'bgimg': bgimg,
                    'img_id': img_id,
                    'image_w': image_w,
                    'image_h': image_h}
        
        #  验证的图像
        if self.phase == 'val':
            img_id = self.img_ids[index]
            image,bgimg = self.processing_test_withbg(image, bgimg, self.input_h, self.input_w)
            return {'image': image,
                    'bgimg': bgimg,
                    'img_id': img_id,
                    'image_w': image_w,
                    'image_h': image_h}


        # 训练的图像
        elif self.phase == 'train':
            annotation = self.load_annotation(index)

            # ？是否可以在这里添加对背景的处理
            image, bgimg, annotation = self.data_transform(image, bgimg, annotation)
            
            data_dict = self.generate_ground_truth(image, bgimg, annotation)
            return data_dict
