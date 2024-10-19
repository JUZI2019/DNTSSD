import torch
import os
import func_utils


class EvalModule(object):
    def __init__(self, dataset, num_classes, model, decoder):
        torch.manual_seed(317)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dataset = dataset
        self.num_classes = num_classes
        self.model = model
        self.decoder = decoder


    def load_model(self, model, resume):
        checkpoint = torch.load(resume, map_location=lambda storage, loc: storage)
        print('loaded weights from {}, epoch {}'.format(resume, checkpoint['epoch']))
        state_dict_ = checkpoint['model_state_dict']
        model.load_state_dict(state_dict_, strict=False)
        return model

    def evaluation(self, args, down_ratio):
        # save_path = 'weights_'+args.dataset
        # self.model = self.load_model(self.model, os.path.join(save_path, args.resume))


        self.model = self.load_model(self.model, args.resume)
        self.model = self.model.to(self.device)
        self.model.eval()

        result_path = os.path.join(args.save_dir,  'evalTxt')
        if not os.path.exists(result_path):
            os.mkdir(result_path)


        # result_path = os.path.join(args.data_dir,'result_'+args.dataset)
        # if not os.path.exists(result_path):
        #     os.mkdir(result_path)

        dataset_module = self.dataset[args.dataset]
        dsets = dataset_module(data_dir=args.data_dir,
                               phase='test',
                               input_h=args.input_h,
                               input_w=args.input_w,
                               down_ratio=down_ratio)
        
        # # 注释掉是因为有结果再evalTxt里了
        func_utils.write_results_siamese(args,
                                 self.model,
                                 dsets,
                                 down_ratio,
                                 self.device,
                                 self.decoder,
                                 result_path,
                                 print_ps=True)

        # if args.dataset == 'dota':
        if args.dataset =='dota_ship_siamese':
            merge_path = os.path.join(args.save_dir,  'evalTxt_merge')
            if not os.path.exists(merge_path):
                os.mkdir(merge_path)
            dsets.merge_crop_image_results(result_path, merge_path)   # 注释掉是因为有结果再evalTxt_merge里了
            
            # # 全部的在这里统计
            # ap = dsets.dec_evaluation(merge_path)
            
            # in/offshore在这里统计
            ap = dsets.dec_evaluation(result_path)
            

            return None
        else:
            ap = dsets.dec_evaluation(result_path)
            return ap