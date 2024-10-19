import argparse
import train
import train_siamese
import test
import eval
import eval_siamese
from datasets.dataset_dota import DOTA
from datasets.dataset_hrsc import HRSC
from datasets.dataset_dota_ship_siamese import DOTA_SHIP_SIAMESE

from models import ctrbox_net, ctrbox_siamese
import decoder
import os

def parse_args():
    parser = argparse.ArgumentParser(description='BBAVectors Implementation')
    parser.add_argument('--num_epoch', type=int, default=60, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Number of batch size')  # batch_size 过小会出现loss=nan，4不行，8好像也有出现的
    parser.add_argument('--num_workers', type=int, default=1, help='Number of workers')
    parser.add_argument('--init_lr', type=float, default=1.25e-4, help='Initial learning rate')
    parser.add_argument('--input_h', type=int, default=608, help='Resized image height')
    parser.add_argument('--input_w', type=int, default=608, help='Resized image width')
    parser.add_argument('--K', type=int, default=500, help='Maximum of objects')
    parser.add_argument('--conf_thresh', type=float, default=0.18, help='Confidence threshold, 0.1 for general evaluation')
    parser.add_argument('--ngpus', type=int, default=1, help='Number of gpus, ngpus>1 for multigpu')
    parser.add_argument('--resume_train', type=str, default='', help='Weights resumed in training')
    parser.add_argument('--resume', type=str, default='', help='Weights resumed in testing and evaluation')
    parser.add_argument('--dataset', type=str, default='dota_ship_siamese', help='Name of dataset')
    parser.add_argument('--data_dir', type=str, default='', help='Data directory')
    parser.add_argument('--save_dir', type=str, default='', help='result directory')
    parser.add_argument('--phase', type=str, default='train', help='Phase choice= {train, test, eval}')
    parser.add_argument('--wh_channels', type=int, default=8, help='Number of channels for the vectors (4x2)')
    parser.add_argument('--siamese', type=bool, default=True, help='Using dual-branch model?')
    # parser.add_argument('--module', type=int, help='消融实验模块')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    dataset = {'dota': DOTA, 'hrsc': HRSC, 'dota_ship_siamese':DOTA_SHIP_SIAMESE}
    num_classes = {'dota': 15, 'hrsc': 1, 'dota_ship_siamese': 1 }  

    heads = {'hm': num_classes[args.dataset],  
             'wh': 10,
             'reg': 2,
             'cls_theta': 1
             }
        #  hm: heatmap 尺寸为 1 x num_classes x 152 x 152 (表示num_classes个类别)
        #  wh: 偏移尺寸为 1 x 10 x 152 x 152 (四个点的长宽（4*2）+ 旋转框的长宽（2）=10 )
        #  reg: 预测中心点偏移量尺寸为 1 x 2 x 152 x 152 (2表示x, y) 
        #  cls_theta：旋转角度1 * 152* 152
    down_ratio = 4
    model = ctrbox_net.CTRBOX(heads=heads,
                              pretrained=True,
                              down_ratio=down_ratio,
                              final_kernel=1,
                              head_conv=256)
    if args.siamese:        
        model = ctrbox_siamese.CTRBOX_SIAMESE(heads=heads,
                        pretrained=True,
                        down_ratio=down_ratio,
                        final_kernel=1,
                        head_conv=256)
    

    decoder = decoder.DecDecoder(K=args.K,
                                 conf_thresh=args.conf_thresh,
                                 num_classes=num_classes[args.dataset])
    if args.phase == 'train' :
        if args.siamese:
            ctrbox_obj = train_siamese.TrainModule(dataset=dataset,
                                        num_classes=num_classes,
                                        model=model,
                                        decoder=decoder,
                                        down_ratio=down_ratio)

            ctrbox_obj.train_network(args)
        else:            
            ctrbox_obj = train.TrainModule(dataset=dataset,
                                        num_classes=num_classes,
                                        model=model,
                                        decoder=decoder,
                                        down_ratio=down_ratio)

            ctrbox_obj.train_network(args)
    elif args.phase == 'test':
        ctrbox_obj = test.TestModule(dataset=dataset, num_classes=num_classes, model=model, decoder=decoder)
        ctrbox_obj.test(args, down_ratio=down_ratio)
    else:
        if args.siamese:
            ctrbox_obj = eval_siamese.EvalModule(dataset=dataset, num_classes=num_classes, model=model, decoder=decoder)
            ctrbox_obj.evaluation(args, down_ratio=down_ratio)
        else:
            ctrbox_obj = eval.EvalModule(dataset=dataset, num_classes=num_classes, model=model, decoder=decoder)
            ctrbox_obj.evaluation(args, down_ratio=down_ratio)