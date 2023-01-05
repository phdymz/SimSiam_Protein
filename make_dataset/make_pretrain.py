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




begin = 104420




# Parse the arguments, prepare the TensorBoard writer:
args = parser.parse_args()
source_root = '/media/ymz/11c16692-3687-420b-9787-72a7f9e3bdaf/Protein/atom'
save_root = '/media/ymz/11c16692-3687-420b-9787-72a7f9e3bdaf/Protein/pdb_processed'


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
    # train_dataset = ProteinPairsSurfaces(
    #     "/media/ymz/2b933929-0294-4162-9385-4fe3eec72189/protein/dmasif/surface_data", ppi=args.search, train=True, transform=transformations
    # )
    # train_dataset = [data for data in train_dataset if iface_valid_filter(data)]
    # train_loader = DataLoader(
    #     train_dataset, batch_size=1, follow_batch=batch_vars, shuffle=True
    # )

    for _, _, files in tqdm(os.walk(source_root)):
        break

    files.sort()
    print("Preprocessing training dataset")

    for i, item in enumerate(tqdm(files)):
        if i < begin:
            continue
        try:
            pdb_root = os.path.join(source_root, item)
            data = np.load(pdb_root)
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

            if len(atom) > 20000:
                continue

            if type.sum() != atom_type.sum():
                continue

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

            save_name = os.path.join(save_root, item)
            np.savez(save_name,
                     xyz = xyz.cpu().numpy().astype('float32'),
                     normal = normal.cpu().numpy().astype('float32'),
                     curvature = P_curvatures.cpu().numpy().astype('float32'),
                     atom = atom.cpu().numpy().astype('float32'),
                     atom_type = atom_type.cpu().numpy().astype('float32'),
                     atom_center = atom_center.cpu().numpy().astype('float32'))

        except Exception as e:
            print("Problem with cuda")
            print(e)
            continue













