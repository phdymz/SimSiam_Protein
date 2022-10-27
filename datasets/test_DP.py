import torch
from protein_dataset import Protein
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
import torch.utils.data.distributed
import os
from models.dmasif import dMaSIF

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

if __name__ == "__main__":
    data_set = Protein(phase = 'train')
    dataloader = DataLoader(
        data_set,
        batch_size=32,
        num_workers = 12
    )

    for xyz, normal, label, curvature, dists, atom_type_sel  in tqdm(dataloader):
        xyz, label, curvature, dists, atom_type_sel = xyz.cuda(), label.cuda(), \
                                                                     curvature.cuda(), dists.cuda(), \
                                                                     atom_type_sel.cuda()
        break
