from main import train
import pandas as pd
import torch
import numpy as np
import random
from pathlib import Path
from argparse import ArgumentParser

# parameters
DATASETS = ["visa"]
CATEGORIES = {
    # "mvtec_ad": ['screw', 'pill', 'capsule', 'carpet', 'grid', 'tile', 'wood', 'zipper', 'cable', 'toothbrush', 'transistor', 'metal_nut', 'bottle', 'hazelnut', 'leather'],
    # "visa": ["candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1", "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum"]
    # "visa": ["pipe_fryum", "pcb4", "pcb3", "pcb2", "pcb1", "macaroni2", "macaroni1", "fryum", "chewinggum", "cashew", "capsules", "candle"]
    "visa": ["candle", "capsules", "cashew", "fryum"]
}
# CATEGORIES = {
#     "mvtec_ad": ['bottle'],
#     "visa": ["candle"]
# }
DATASET_PATHS = {
    "mvtec_ad": "/home/djameln/datasets/MVTec/",
    "visa": "/home/djameln/datasets/visa/visa_pytorch/"
}

BATCH_SIZE=16
IMAGE_SIZE=256
PROJ_LR=0.001
DISTILL_LR=0.005
WEIGHT_PROJ=0.2

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--gpu', default=0, type=int)
    pars = parser.parse_args()
    return pars

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    setup_seed(42)
    metrics = {'class': [], 'AUROC_sample':[], 'AUROC_pixel': []}

    args=get_args()
    
    # train all_classes
    # for c in all_classes
    for d in DATASETS:
        for c in CATEGORIES[d]:
            save_folder = Path(f"results/rd++_wr50_ext4") / d / c
            save_folder.mkdir(exist_ok=True, parents=True)
            auroc_sp, auroc_px, aupro_px = train(c, 
                                                 d,
                                                 DATASET_PATHS[d],
                                                 str(save_folder),
                                                 batch_size=BATCH_SIZE,
                                                 image_size=IMAGE_SIZE,
                                                 proj_lr=PROJ_LR,
                                                 distill_lr=DISTILL_LR,
                                                 weight_proj=WEIGHT_PROJ,
                                                 gpu=args.gpu)
            print('Best score of class: {}, Auroc sample: {:.4f}, Auroc pixel:{:.4f}, Pixel Aupro: {:.4f}'.format(c, auroc_sp, auroc_px, 0))
            metrics['class'].append(c)
            metrics['AUROC_sample'].append(auroc_sp)
            metrics['AUROC_pixel'].append(auroc_px)
            pd.DataFrame(metrics).to_csv(f'{save_folder}/metrics_results.csv', index=False)
