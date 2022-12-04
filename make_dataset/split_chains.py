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


def show_pcd(points):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.paint_uniform_color([1,0,0])
    o3d.visualization.draw_geometries([point_cloud])



def get_atoms(fname, target_root):
    parser = PDBParser()
    structure = parser.get_structure("structure", fname)
    atoms = structure.get_atoms()

    chains = {}


    for atom in atoms:
        chain_id = atom.get_full_id()[2]
        if chain_id not in chains:
            chains[chain_id] = {
                'id': chain_id,
                'atoms':[],
                'atom_types':[]}

        chains[chain_id]['atoms'].append(atom.get_coord())
        chains[chain_id]['atom_types'].append(ele2num[atom.element])


    for chain_id in chains:
        coords = np.stack(chains[chain_id]['atoms'])
        types_array = np.zeros((len(chains[chain_id]['atom_types']), len(ele2num)), dtype=np.float32)
        for i, t in enumerate(chains[chain_id]['atom_types']):
            types_array[i, t] = 1.0


        save_name = fname[-8:].replace('.pdb', '_' + chain_id + '.npz')
        save_name = os.path.join(target_root, save_name)
        np.savez(save_name,
                atoms = coords,
                types = types_array)











if __name__ == '__main__':
    # args = parser.parse_args()

    pdb_root = '/media/ymz/11c16692-3687-420b-9787-72a7f9e3bdaf/Protein/pdb'
    target_root = '/media/ymz/11c16692-3687-420b-9787-72a7f9e3bdaf/Protein/atom'

    for _, _, files in os.walk(pdb_root):
        break

    for item in tqdm(files):
        get_atoms(os.path.join(pdb_root, item), target_root)

        break


    # get_atoms(os.path.join(pdb_root, '1aa7.pdb'), target_root)




