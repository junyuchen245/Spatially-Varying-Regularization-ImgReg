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
import digital_diffeomorphism as dd
import torch.nn as nn

def main():
    atlas_dir = '/scratch/jchen/DATA/IXI/atlas.pkl'
    test_dir = '/scratch/jchen/DATA/IXI/Test/'
    model_folder = 'HyperTransMorphSPR_beta/'
    model_dir = '/scratch2/jchen/IXI/experiments/' + model_folder
    dict = utils.process_label()
    if not os.path.exists('Quantitative_Results/'):
        os.makedirs('Quantitative_Results/')
    if os.path.exists('Quantitative_Results/'+model_folder[:-1]+'.csv'):
        os.remove('Quantitative_Results/'+model_folder[:-1]+'.csv')
    csv_writter(model_folder[:-1], 'Quantitative_Results/' + model_folder[:-1])
    line = ''
    for i in range(46):
        line = line + ',' + dict[i]
    csv_writter(line +','+'ndv'+','+'ndp', 'Quantitative_Results/' + model_folder[:-1])

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
    reg_model = utils.register_model((H, W, D))
    reg_model.cuda()
    test_composed = transforms.Compose([trans.Seg_norm(),
                                        trans.NumpyType((np.float32, np.int16)),
                                        ])
    test_set = datasets.IXIBrainInferDataset(glob.glob(test_dir + '*.pkl'), atlas_dir, transforms=test_composed)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, drop_last=True)
    eval_dsc_def = utils.AverageMeter()
    eval_dsc_raw = utils.AverageMeter()
    
    with torch.no_grad():
        stdy_idx = 0
        for data in test_loader:
            model.eval()
            data = [t.cuda() for t in data]
            x = data[0]
            y = data[1]
            x_seg = data[2]
            y_seg = data[3]
            x_half = nn.functional.avg_pool3d(x, 2).cuda()
            y_half = nn.functional.avg_pool3d(y, 2).cuda()
            beta_code = torch.tensor([1.06], dtype=x.dtype, device=x.device).unsqueeze(dim=0)
            reg_code = torch.tensor([1.08], dtype=x.dtype, device=x.device).unsqueeze(dim=0)
            hyper_code = torch.cat((beta_code, reg_code), -1)
            flow, _ = model((x_half, y_half), hyper_code)
            flow = nn.functional.interpolate(flow.cuda(), scale_factor=2, mode='trilinear', align_corners=False) * 2
            x_seg_oh = nn.functional.one_hot(x_seg.long(), num_classes=46)
            x_seg_oh = torch.squeeze(x_seg_oh, 1)
            x_seg_oh = x_seg_oh.permute(0, 4, 1, 2, 3).contiguous()
            x_segs = []
            for i in range(46):
                def_seg = reg_model([x_seg_oh[:, i:i + 1, ...].float(), flow.float()])
                x_segs.append(def_seg)
            x_segs = torch.cat(x_segs, dim=1)
            def_out = torch.argmax(x_segs, dim=1, keepdim=True)
            del x_segs, x_seg_oh
            
            mask = x.detach().cpu().numpy()[0, 0, 1:-1, 1:-1, 1:-1]
            mask = mask > 0
            disp_field = flow.cpu().detach().numpy()[0]
            trans_ = disp_field + dd.get_identity_grid(disp_field)
            jac_dets = dd.calc_jac_dets(trans_)
            non_diff_voxels, non_diff_tetrahedra, non_diff_volume = dd.calc_measurements(jac_dets, mask)
            total_voxels = np.sum(mask)
            ndv = non_diff_volume / total_voxels * 100
            ndp = non_diff_voxels / total_voxels * 100
            
            line = utils.dice_val_substruct(def_out.long(), y_seg.long(), stdy_idx)
            line = line +','+str(ndv)+','+str(ndp)
            csv_writter(line, 'Quantitative_Results/' + model_folder[:-1])
            dsc_trans = utils.dice_val(def_out.long(), y_seg.long(), 46)
            dsc_raw = utils.dice_val(x_seg.long(), y_seg.long(), 46)
            print('Trans dsc: {:.4f}, Raw dsc: {:.4f}'.format(dsc_trans.item(),dsc_raw.item()))
            eval_dsc_def.update(dsc_trans.item(), x.size(0))
            eval_dsc_raw.update(dsc_raw.item(), x.size(0))
            stdy_idx += 1

        print('Deformed DSC: {:.3f} +- {:.3f}, Affine DSC: {:.3f} +- {:.3f}'.format(eval_dsc_def.avg,
                                                                                    eval_dsc_def.std,
                                                                                    eval_dsc_raw.avg,
                                                                                    eval_dsc_raw.std))

def csv_writter(line, name):
    with open(name+'.csv', 'a') as file:
        file.write(line)
        file.write('\n')

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