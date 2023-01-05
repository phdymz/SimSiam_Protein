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
from make_dataset.geometry_processing import curvatures


# Custom data loader and model:
from make_dataset.data import ProteinPairsSurfaces, PairData, CenterPairAtoms
from make_dataset.data import RandomRotationPairAtoms, NormalizeChemFeatures, iface_valid_filter
from make_dataset.helper import *
from make_dataset.Arguments import parser
from make_dataset.preprocess import iterate_surface_precompute
from Bio.PDB import *
from make_dataset.geometry_processing import (
    curvatures,
    mesh_normals_areas,
    tangent_vectors,
    atoms_to_points_normals,
)



labels_dict = {"ADP": 0, "COA": 1, "FAD": 2, "HEM": 3, "NAD": 4, "NAP": 5, "SAM": 6}





# Parse the arguments, prepare the TensorBoard writer:
args = parser.parse_args()
source_root = '/media/ymz/11c16692-3687-420b-9787-72a7f9e3bdaf/Protein/Ligand/atom'
save_root = '/media/ymz/11c16692-3687-420b-9787-72a7f9e3bdaf/Protein/Ligand/processed'


if __name__ == "__main__":

    # Ensure reproducibility:
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    train_list = np.load('/home/ymz/桌面/protein/masif/data/masif_ligand/lists/train_pdbs_sequence.npy')
    val_list = np.load('/home/ymz/桌面/protein/masif/data/masif_ligand/lists/val_pdbs_sequence.npy')
    test_list = np.load('/home/ymz/桌面/protein/masif/data/masif_ligand/lists/test_pdbs_sequence.npy')


    # train set
    print("Preprocessing training dataset")
    phase = 'train'
    for item in tqdm(train_list):
        item = item.decode('utf-8') + '.npz'
        pdb_root = pdb_root = os.path.join(source_root, item)
        if not os.path.exists(pdb_root):
            print(item)
            continue

        data = np.load(pdb_root, allow_pickle=True)
        atom = data['atoms']
        type = data['types']
        atom_type = np.concatenate([
            type[:, 1][:, None],
            type[:, 15][:, None],
            type[:, 2][:, None],
            type[:, 0][:, None],
            type[:, 3][:, None],
            type[:, 4][:, None],
        ], axis=-1)

        atom = torch.from_numpy(atom).cuda()
        atom_type = torch.from_numpy(atom_type).cuda()
        atom_batch = torch.zeros_like(atom)[:,0]
        atom_batch = atom_batch.int()

        xyz, normal, batch = atoms_to_points_normals(
            atom,
            atom_batch,
            atomtypes=atom_type,
            resolution=args.resolution,
            sup_sampling=args.sup_sampling,
            distance=args.distance,
        )

        atom_center = atom.mean(0)
        xyz = xyz - atom_center
        atom = atom - atom_center


        P_curvatures = curvatures(
            xyz,
            triangles=None if args.use_mesh else None,
            normals=None if args.use_mesh else normal,
            scales=args.curvature_scales,
            batch=batch,
        )

        data_type = data['data_type']
        data_coord = data['data_coord']

        atom_center = atom_center.cpu().reshape(1,3).numpy().astype('float32')
        for i in range(len(data_type)):
            label = labels_dict[data_type[i]]
            ligand = data_coord[i] - atom_center

            save_name = os.path.join(save_root, phase, item.replace('.npz', '_' + data_type[i] + '_{}.npz'.format(i)))
            np.savez(save_name,
                 xyz = xyz.cpu().numpy().astype('float32'),
                 normal = normal.cpu().numpy().astype('float32'),
                 curvature = P_curvatures.cpu().numpy().astype('float32'),
                 atom = atom.cpu().numpy().astype('float32'),
                 atom_type = atom_type.cpu().numpy().astype('float32'),
                 atom_center = atom_center,
                label = np.array(label).astype('float32'),
                     ligand = ligand.astype('float32'))



    # valid
    print("Preprocessing valid dataset")
    phase = 'valid'
    for item in tqdm(val_list):
        item = item.decode('utf-8') + '.npz'
        pdb_root = pdb_root = os.path.join(source_root, item)
        if not os.path.exists(pdb_root):
            print(item)
            continue

        data = np.load(pdb_root, allow_pickle=True)
        atom = data['atoms']
        type = data['types']
        atom_type = np.concatenate([
            type[:, 1][:, None],
            type[:, 15][:, None],
            type[:, 2][:, None],
            type[:, 0][:, None],
            type[:, 3][:, None],
            type[:, 4][:, None],
        ], axis=-1)

        atom = torch.from_numpy(atom).cuda()
        atom_type = torch.from_numpy(atom_type).cuda()
        atom_batch = torch.zeros_like(atom)[:, 0]
        atom_batch = atom_batch.int()

        xyz, normal, batch = atoms_to_points_normals(
            atom,
            atom_batch,
            atomtypes=atom_type,
            resolution=args.resolution,
            sup_sampling=args.sup_sampling,
            distance=args.distance,
        )

        atom_center = atom.mean(0)
        xyz = xyz - atom_center
        atom = atom - atom_center

        P_curvatures = curvatures(
            xyz,
            triangles=None if args.use_mesh else None,
            normals=None if args.use_mesh else normal,
            scales=args.curvature_scales,
            batch=batch,
        )

        data_type = data['data_type']
        data_coord = data['data_coord']

        atom_center = atom_center.cpu().reshape(1, 3).numpy().astype('float32')
        for i in range(len(data_type)):
            label = labels_dict[data_type[i]]
            ligand = data_coord[i] - atom_center

            save_name = os.path.join(save_root, phase,
                                     item.replace('.npz', '_' + data_type[i] + '_{}.npz'.format(i)))
            np.savez(save_name,
                     xyz=xyz.cpu().numpy().astype('float32'),
                     normal=normal.cpu().numpy().astype('float32'),
                     curvature=P_curvatures.cpu().numpy().astype('float32'),
                     atom=atom.cpu().numpy().astype('float32'),
                     atom_type=atom_type.cpu().numpy().astype('float32'),
                     atom_center=atom_center,
                     label=np.array(label).astype('float32'),
                     ligand=ligand.astype('float32'))

    # test
    print("Preprocessing test dataset")
    phase = 'test'
    for item in tqdm(test_list):
        item = item.decode('utf-8') + '.npz'
        pdb_root = pdb_root = os.path.join(source_root, item)
        if not os.path.exists(pdb_root):
            print(item)
            continue

        data = np.load(pdb_root, allow_pickle=True)
        atom = data['atoms']
        type = data['types']
        atom_type = np.concatenate([
            type[:, 1][:, None],
            type[:, 15][:, None],
            type[:, 2][:, None],
            type[:, 0][:, None],
            type[:, 3][:, None],
            type[:, 4][:, None],
        ], axis=-1)

        atom = torch.from_numpy(atom).cuda()
        atom_type = torch.from_numpy(atom_type).cuda()
        atom_batch = torch.zeros_like(atom)[:, 0]
        atom_batch = atom_batch.int()

        xyz, normal, batch = atoms_to_points_normals(
            atom,
            atom_batch,
            atomtypes=atom_type,
            resolution=args.resolution,
            sup_sampling=args.sup_sampling,
            distance=args.distance,
        )

        atom_center = atom.mean(0)
        xyz = xyz - atom_center
        atom = atom - atom_center

        P_curvatures = curvatures(
            xyz,
            triangles=None if args.use_mesh else None,
            normals=None if args.use_mesh else normal,
            scales=args.curvature_scales,
            batch=batch,
        )

        data_type = data['data_type']
        data_coord = data['data_coord']

        atom_center = atom_center.cpu().reshape(1, 3).numpy().astype('float32')
        for i in range(len(data_type)):
            label = labels_dict[data_type[i]]
            ligand = data_coord[i] - atom_center

            save_name = os.path.join(save_root, phase,
                                     item.replace('.npz', '_' + data_type[i] + '_{}.npz'.format(i)))
            np.savez(save_name,
                     xyz=xyz.cpu().numpy().astype('float32'),
                     normal=normal.cpu().numpy().astype('float32'),
                     curvature=P_curvatures.cpu().numpy().astype('float32'),
                     atom=atom.cpu().numpy().astype('float32'),
                     atom_type=atom_type.cpu().numpy().astype('float32'),
                     atom_center=atom_center,
                     label=np.array(label).astype('float32'),
                     ligand=ligand.astype('float32'))

