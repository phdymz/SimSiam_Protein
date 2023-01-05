import numpy as np
import json
from tqdm import tqdm
from matplotlib import pyplot as plt
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
import os
import numpy as np
import torch
from tqdm  import tqdm
from Bio.PDB import *
import open3d as o3d
import warnings

warnings.filterwarnings("ignore")




ele2num = {
'N':0,
'C':1,
'O':2,
'S':3,
'SE':4,
'MG':5,
'CL':6,
'ZN':7,
'MN':8,
'K':9,
'B':10,
'P':11,
'AS':12,
'NA':13,
'CU':14,
'H':15,
'BR':16,
'F':17,
'CA':18,
'CD':19,
'NI':20,
'U':21,
'D':22,
'FE':23,
'I':24,
'BA':25,
'RB':26,
'CO':27,
'MO':28,
'W':29,
'HG':30,
'TL':31,
'YB':32,
'GD':33,
'CS':34,
'KR':35,
'PT':36,
'':37,
'V':38,
'XE':39,
'GA':40,
'TI':41,
'AU':42,
'SR':43,
'AL':44,
'BE':45,
'SM':46,
'PB':47,
'AG':48,
'PR':49,
'Y':50,
'AR':51,
'RU':52,
'LI':53,
'RE':54,
'LA':55,
'LU':56,
'TH':57,
'TB':58,
'IN':59,
'SI':60,
'CF':61,
'OS':62,
'SN':63,
'TA':64,
'EU':65,
'RH':66,
'SB':67,
'IR':68,
'TE':69,
'CR':70,
'PD':71,
'HO':72,
'ER':73,
'BI':74,
'AM':75,
'CE':76,
'HF':77,
'ZR':78,
'ND':79,
'NE':80,
'DY':81,
'PU':82,
'SC':83,
'CM':84,
}


def get_atoms(fname, target_root):
    parser = PDBParser()
    structure = parser.get_structure("structure", fname)
    atoms = structure.get_atoms()

    pdb_atoms = []
    pdb_types = []

    for atom in atoms:
        pdb_atoms.append(atom.get_coord())
        pdb_types.append(ele2num[atom.element])


    coords = np.stack(pdb_atoms)
    types_array = np.zeros((len(pdb_atoms), len(ele2num)), dtype=np.float32)
    for i, t in enumerate(pdb_types):
        types_array[i, t] = 1.0


    np.savez(target_root,
            atoms = coords,
            types = types_array)


def load_idx(mode):
    with open(f"/media/ymz/11c16692-3687-420b-9787-72a7f9e3bdaf/Protein/enzyme/metadata/base_split.json", "r") as f:
        splits = json.load(f)
    ids = splits[mode]

    with open(f"/media/ymz/11c16692-3687-420b-9787-72a7f9e3bdaf/Protein/enzyme/metadata/function_labels.json",
              "r") as f:
        labels_all = json.load(f)

    with open(f"/media/ymz/11c16692-3687-420b-9787-72a7f9e3bdaf/Protein/enzyme/metadata/labels_to_idx.json", "r") as f:
        labels_to_idx = json.load(f)

    idx_to_labels = {idx: label for label, idx in labels_to_idx.items()}

    return ids, labels_all, labels_to_idx, idx_to_labels




args = parser.parse_args()
source_root = '/media/ymz/11c16692-3687-420b-9787-72a7f9e3bdaf/Protein/enzyme/pdb_files'
save_root = '/media/ymz/11c16692-3687-420b-9787-72a7f9e3bdaf/Protein/enzyme_processed'


if __name__ == "__main__":
    mode = 'test'

    count = []

    # Ensure reproducibility:
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    transformations = (
        Compose([NormalizeChemFeatures(), CenterPairAtoms(), RandomRotationPairAtoms()])
        if args.random_rotation
        else Compose([NormalizeChemFeatures()])
    )


    print("Preprocessing dataset")
    idx, labels_all, labels_to_idx, idx_to_labels = load_idx(mode)

    for i, item in enumerate(tqdm(idx)):
        label = labels_all[item]
        label = labels_to_idx[label]
        # try:
        pdb_root = os.path.join(source_root, item, item+'_fixed.pdb')
        if not os.path.exists(pdb_root):
            print(item)
            pdb_root = os.path.join(source_root, item, item + '.pdb')
        else:
            save_npz = os.path.join(save_root,'raw', mode, item+'.npz')
            get_atoms(pdb_root, save_npz)
            data = np.load(save_npz)
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

            count.append(len(atom))

            # if type.sum() != atom_type.sum():
            #     print(item)
            #
            # atom = torch.from_numpy(atom).cuda()
            # atom_type = torch.from_numpy(atom_type).cuda()
            # atom_batch = torch.zeros_like(atom)[:,0]
            # atom_batch = atom_batch.int()
            #
            # xyz, normal, batch = atoms_to_points_normals(
            #     atom,
            #     atom_batch,
            #     atomtypes=atom_type,
            #     resolution=args.resolution,
            #     sup_sampling=args.sup_sampling,
            #     distance=args.distance,
            # )
            #
            # atom_center = atom.mean(0)
            # xyz = xyz - atom_center
            # atom = atom - atom_center
            #
            #
            # P_curvatures = curvatures(
            #     xyz,
            #     triangles=None if args.use_mesh else None,
            #     normals=None if args.use_mesh else normal,
            #     scales=args.curvature_scales,
            #     batch=batch,
            # )
            #
            # save_name = os.path.join(save_root, mode, item+'.npz')
            # np.savez(save_name,
            #          xyz = xyz.cpu().numpy().astype('float32'),
            #          normal = normal.cpu().numpy().astype('float32'),
            #          curvature = P_curvatures.cpu().numpy().astype('float32'),
            #          atom = atom.cpu().numpy().astype('float32'),
            #          atom_type = atom_type.cpu().numpy().astype('float32'),
            #          atom_center = atom_center.cpu().numpy().astype('float32'))

    plt.hist(np.array(count))





