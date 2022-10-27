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
    train_dataset = ProteinPairsSurfaces(
        "/media/ymz/2b933929-0294-4162-9385-4fe3eec72189/protein/dmasif/surface_data", ppi=args.search, train=True, transform=transformations
    )
    train_dataset = [data for data in train_dataset if iface_valid_filter(data)]
    train_loader = DataLoader(
        train_dataset, batch_size=1, follow_batch=batch_vars, shuffle=True
    )
    print("Preprocessing training dataset")
    train_dataset = iterate_surface_precompute(train_loader, args)

    # Train/Validation split:
    train_nsamples = len(train_dataset)
    val_nsamples = int(train_nsamples * args.validation_fraction)
    train_nsamples = train_nsamples - val_nsamples
    train_dataset, val_dataset = random_split(
        train_dataset, [train_nsamples, val_nsamples]
    )

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


    # PyTorch_geometric data loaders:
    train_loader = DataLoader(
        train_dataset, batch_size=1, follow_batch=batch_vars, shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=1, follow_batch=batch_vars)
    test_loader = DataLoader(test_dataset, batch_size=1, follow_batch=batch_vars)



    for i, P in enumerate(tqdm(train_loader)):

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

        save_name = os.path.join(save_root, 'train', '{}.npz'.format(i))
        np.savez(save_name,
                 xyz = xyz.cpu().numpy().astype('float32'),
                 normal = normal.cpu().numpy().astype('float32'),
                 label = label.cpu().numpy().astype('float32'),
                 curvature = P_curvatures.cpu().numpy().astype('float32'),
                 atom = atom.cpu().numpy().astype('float32'),
                 atom_type = atom_type.cpu().numpy().astype('float32'),
                 atom_center = atom_center.cpu().numpy().astype('float32'))


    for i, P in enumerate(tqdm(val_loader)):

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

        save_name = os.path.join(save_root, 'val', '{}.npz'.format(i))
        np.savez(save_name,
                 xyz = xyz.cpu().numpy().astype('float32'),
                 normal = normal.cpu().numpy().astype('float32'),
                 label = label.cpu().numpy().astype('float32'),
                 curvature = P_curvatures.cpu().numpy().astype('float32'),
                 atom = atom.cpu().numpy().astype('float32'),
                 atom_type = atom_type.cpu().numpy().astype('float32'),
                 atom_center = atom_center.cpu().numpy().astype('float32'))


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

        save_name = os.path.join(save_root, 'test', '{}.npz'.format(i))
        np.savez(save_name,
                 xyz = xyz.cpu().numpy().astype('float32'),
                 normal = normal.cpu().numpy().astype('float32'),
                 label = label.cpu().numpy().astype('float32'),
                 curvature = P_curvatures.cpu().numpy().astype('float32'),
                 atom = atom.cpu().numpy().astype('float32'),
                 atom_type = atom_type.cpu().numpy().astype('float32'),
                 atom_center = atom_center.cpu().numpy().astype('float32'))
















