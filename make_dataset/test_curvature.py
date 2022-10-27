# Standard imports:
import os

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split
from torch_geometric.data import DataLoader
from torch_geometric.transforms import Compose
from pathlib import Path
from tqdm import tqdm
from geometry_processing import curvatures


# Custom data loader and model:
from data import ProteinPairsSurfaces, PairData, CenterPairAtoms
from data import RandomRotationPairAtoms, NormalizeChemFeatures, iface_valid_filter
from helper import *
from Arguments import parser
from preprocess import iterate_surface_precompute


# Parse the arguments, prepare the TensorBoard writer:
args = parser.parse_args()
save_root = '/media/ymz/11c16692-3687-420b-9787-72a7f9e3bdaf/Protein/site_processed'

def gen_rot():
    anglex = np.random.uniform() * np.pi
    angley = np.random.uniform() * np.pi
    anglez = np.random.uniform() * np.pi
    cosx = np.cos(anglex)
    cosy = np.cos(angley)
    cosz = np.cos(anglez)
    sinx = np.sin(anglex)
    siny = np.sin(angley)
    sinz = np.sin(anglez)
    Rx = np.array([[1, 0, 0], [0, cosx, -sinx], [0, sinx, cosx]])
    Ry = np.array([[cosy, 0, siny], [0, 1, 0], [-siny, 0, cosy]])
    Rz = np.array([[cosz, -sinz, 0], [sinz, cosz, 0], [0, 0, 1]])
    R_ab = Rx.dot(Ry).dot(Rz)
    return R_ab.astype('float32')

if __name__ == "__main__":

    # Ensure reproducibility:
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # We load the train and test datasets.
    # Random transforms, to ensure that no network/baseline overfits on pose parameters:
    transformations = (
        Compose([NormalizeChemFeatures(), CenterPairAtoms(), RandomRotationPairAtoms()])
        if args.random_rotation
        else Compose([NormalizeChemFeatures()])
    )

    # PyTorch geometric expects an explicit list of "batched variables":
    batch_vars = ["xyz_p1", "xyz_p2", "atom_coords_p1", "atom_coords_p2"]
    # Load the train dataset:


    # Load the test dataset:
    test_dataset = ProteinPairsSurfaces(
        "/media/ymz/2b933929-0294-4162-9385-4fe3eec72189/protein/dmasif/surface_data", ppi=args.search, train=False, transform=transformations
    )
    test_dataset = [data for data in test_dataset if iface_valid_filter(data)]
    test_loader = DataLoader(
        test_dataset, batch_size=1, follow_batch=batch_vars, shuffle=True
    )
    print("Preprocessing testing dataset")
    test_dataset = iterate_surface_precompute(test_loader, args)



    test_loader = DataLoader(test_dataset, batch_size=1, follow_batch=batch_vars)
    for i, P in enumerate(tqdm(test_loader)):

        xyz = P['gen_xyz_p1']
        label = P['gen_labels_p1']
        normal = P['gen_normals_p1']
        batch = P['gen_batch_p1']

        atom = P['atom_coords_p1']
        atom_type = P['atom_types_p1']
        atom_batch = P['atom_coords_p1_batch']

        atom_center = atom.mean(0)
        xyz = xyz - atom_center
        atom = atom - atom_center


        P_curvatures = curvatures(
            xyz,
            triangles=P["triangles"] if args.use_mesh else None,
            normals=None if args.use_mesh else normal,
            scales=args.curvature_scales,
            batch=batch,
        )

        rot = gen_rot()
        rot = torch.from_numpy(rot)

        rot_normal = (torch.matmul(rot, normal.T).T).contiguous()
        rot_xyz = (torch.matmul(rot, xyz.T).T).contiguous()



        P_curvatures_rot = curvatures(
            rot_xyz,
            triangles=P["triangles"] if args.use_mesh else None,
            normals=None if args.use_mesh else rot_normal,
            scales=args.curvature_scales,
            batch=batch,
        )

        print(((P_curvatures_rot - P_curvatures)**2).sum())

        save_name = os.path.join(save_root, 'test', '{}.npz'.format(i))
        np.savez(save_name,
                 xyz = xyz.cpu().numpy().astype('float32'),
                 normal = normal.cpu().numpy().astype('float32'),
                 label = label.cpu().numpy().astype('float32'),
                 curvature = P_curvatures.cpu().numpy().astype('float32'),
                 atom = atom.cpu().numpy().astype('float32'),
                 atom_type = atom_type.cpu().numpy().astype('float32'),
                 atom_center = atom_center.cpu().numpy().astype('float32'))
















