import os
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import shutil

if __name__ == "__main__":

    # names = []
    # length_pcd = []
    # length_atom = []
    #
    # root = '/media/ymz/11c16692-3687-420b-9787-72a7f9e3bdaf/Protein/pdb_processed'
    # for _, _, files in os.walk(root):
    #     break
    # files.sort()
    #
    # for item in tqdm(files):
    #     name = os.path.join(root, item)
    #     data = np.load(name)
    #
    #     names.append(name)
    #     length_atom.append(len(data['atom']))
    #     length_pcd.append(len(data['xyz']))
    #
    # names = np.array(names)
    # length_atom = np.array(length_atom)
    # length_pcd = np.array(length_pcd)
    #
    # np.savez('pre_train.npz', names = names, length_atom = length_atom, length_pcd = length_pcd)


    root = '/media/ymz/11c16692-3687-420b-9787-72a7f9e3bdaf/Protein/pdb_processed'
    target = '/media/ymz/11c16692-3687-420b-9787-72a7f9e3bdaf/Protein/Pretrain_processed'

    total = np.load('./pre_train.npz')
    names = total['names']
    atom_len = total['length_atom']
    xyz_len = total['length_pcd']

    mask = (atom_len < 8000) * (xyz_len < 10000) * (xyz_len > 1000)
    names = names[mask]
    atom_len = atom_len[mask]
    xyz_len = xyz_len[mask]

    plt.hist(atom_len)
    plt.hist(xyz_len)
    plt.show()



    for i in tqdm(range(len(names))):
        save_name = os.path.join(target, names[i].split('processed/')[-1])
        shutil.copy(names[i], save_name)














