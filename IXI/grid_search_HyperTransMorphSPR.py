import glob
import os, utils
from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
import torch
from torchvision import transforms
from natsort import natsorted
from models.TransMorphHyper import CONFIGS as CONFIGS_TM
import models.TransMorphHyper as TransMorph
import torch.nn.functional as F

def eval_test(model, test_loader, reg_model, beta_hyper, reg_hyper):
    eval_dsc = utils.AverageMeter()
    for data in test_loader:
        model.eval()
        data = [t.cuda() for t in data]
        x = data[0]
        y = data[1]
        x_seg = data[2]
        y_seg = data[3]
        x_half = F.avg_pool3d(x, 2).cuda()
        y_half = F.avg_pool3d(y, 2).cuda()
        
        beta_code = torch.tensor([beta_hyper], dtype=x.dtype, device=x.device).unsqueeze(dim=0)
        reg_code = torch.tensor([reg_hyper], dtype=x.dtype, device=x.device).unsqueeze(dim=0)
        hyper_code = torch.cat((beta_code, reg_code), -1)
        flow, _ = model((x_half, y_half), hyper_code)
        
        flow = F.interpolate(flow.cuda(), scale_factor=2, mode='trilinear', align_corners=False) * 2
        
        def_seg = reg_model([x_seg.cuda().float(), flow.cuda()])
        dsc = utils.dice_val_VOI(def_seg.long(), y_seg.long())
        eval_dsc.update(dsc.item(), x.size(0))
    return eval_dsc.avg
        
def main():
    atlas_dir = '/scratch/jchen/DATA/IXI/atlas.pkl'
    test_dir = '/scratch/jchen/DATA/IXI/Val/'
    model_folder = 'HyperTransMorphSPR_beta/'
    model_dir = '/scratch2/jchen/IXI/experiments/' + model_folder
    step_size = 0.02
    '''
    Initialize model
    '''
    H, W, D = 160, 192, 224
    config = CONFIGS_TM['TransMorph-3-LVL']
    config.img_size = (H//2, W//2, D//2)
    config.window_size = (H // 32, W // 32, D // 32)
    config.out_chan = 3
    model = TransMorph.TransMorphTVFSPR(config, SVF=True)
    best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[0])['state_dict']
    print('Best model: {}'.format(natsorted(os.listdir(model_dir))[0]))
    model.load_state_dict(best_model)
    model.cuda()
    
    reg_model = utils.register_model((H, W, D), 'nearest')
    reg_model.cuda()
    test_composed = transforms.Compose([trans.Seg_norm(),
                                        trans.NumpyType((np.float32, np.int16)),
                                        ])
    test_set = datasets.IXIBrainInferDataset(glob.glob(test_dir + '*.pkl'), atlas_dir, transforms=test_composed)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, drop_last=True)
    dsc_landscape = np.zeros((len(np.arange(0, 1.1, step_size)), len(np.arange(0, 1.1, step_size))))
    with torch.no_grad():
        for i, beta in enumerate(np.arange(0, 1.1, step_size)):
            for j, reg in enumerate(np.arange(0, 1.1, step_size)):
                dsc_avg = eval_test(model.eval(), test_loader, reg_model, beta, reg)
                dsc_landscape[i,j] = dsc_avg
                print(beta, reg, dsc_avg)
    np.savez_compressed('SPR_landscape', dsc=dsc_landscape)

if __name__ == '__main__':
    '''
    GPU configuration
    '''
    GPU_iden = 0
    GPU_num = torch.cuda.device_count()
    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    torch.cuda.set_device(GPU_iden)
    GPU_avai = torch.cuda.is_available()
    print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
    print('If the GPU is available? ' + str(GPU_avai))
    main()