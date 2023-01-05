import numpy as np
from tqdm import tqdm
import os
from scipy import spatial
from Bio.PDB import *
import warnings

warnings.filterwarnings("ignore")

def show_ligand(atom, data_coord):
    import open3d as o3d
    pcd0 = o3d.geometry.PointCloud()
    pcd0.points = o3d.utility.Vector3dVector(atom)
    pcd0.paint_uniform_color([1, 0, 0])

    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(data_coord)
    pcd1.paint_uniform_color([0, 0, 1])

    o3d.visualization.draw_geometries([pcd0, pcd1])


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


def get_atoms(fname):
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

    return coords, types_array

labels_dict = {"ADP": 1, "COA": 2, "FAD": 3, "HEM": 4, "NAD": 5, "NAP": 6, "SAM": 7}






if __name__ == "__main__":
    for _, _, files in os.walk('/media/ymz/11c16692-3687-420b-9787-72a7f9e3bdaf/Protein/Ligand/pdb'):
        break

    for item in tqdm(files):
        data_path = os.path.join('/media/ymz/11c16692-3687-420b-9787-72a7f9e3bdaf/Protein/Ligand/pdb', item)

        data_coord = os.path.join('/media/ymz/11c16692-3687-420b-9787-72a7f9e3bdaf/Protein/Ligand/masif_ligand_pdbs_and_ply_files/00c-ligand_coords',
                                  item.split('_')[0]+'_ligand_coords.npy')
        data_type = os.path.join(
            '/media/ymz/11c16692-3687-420b-9787-72a7f9e3bdaf/Protein/Ligand/masif_ligand_pdbs_and_ply_files/00c-ligand_coords',
            item.split('_')[0] + '_ligand_types.npy')


        if not os.path.exists(data_coord):
            print(item)
        if not os.path.exists(data_type):
            print(item)

        type =  np.load(data_type)
        data_type = []
        for i in range(len(type)):
            data_type.append(type[i].decode('utf-8'))
        data_type = np.array(data_type)
        data_coord = np.load(data_coord, encoding='latin1', allow_pickle=True)

        # some pdb don't include ligand
        if len(data_type) < 1:
            continue

        coords, type = get_atoms(data_path)

        atom_path = os.path.join('/media/ymz/11c16692-3687-420b-9787-72a7f9e3bdaf/Protein/Ligand/atom',
                                 item.replace('.pdb', '.npz'))
        # show_ligand(coords, data_coord[0])
        np.savez(atom_path,
                 atoms = coords,
                 types = type,
                 data_type = data_type,
                 data_coord = data_coord)

        # atom_type = np.concatenate([
        #     type[:, 1][:, None],
        #     type[:, 15][:, None],
        #     type[:, 2][:, None],
        #     type[:, 0][:, None],
        #     type[:, 3][:, None],
        #     type[:, 4][:, None],
        # ], axis=-1)











