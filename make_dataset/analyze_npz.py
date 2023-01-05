import os
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt


if __name__ == "__main__":

    length_pcd = []
    length_atom = []

    root = '/media/ymz/11c16692-3687-420b-9787-72a7f9e3bdaf/Protein/site_processed/test'
    for _, _, files in os.walk(root):
        break

    for item in tqdm(files):
        name = os.path.join(root, item)
        data = np.load(name)
        atom = data['atom']
        pcd = data['xyz']

        length_atom.append(len(atom))
        length_pcd.append(len(pcd))


    print(np.mean(length_atom))
    print(np.std(length_atom))
    print(np.mean(length_pcd))
    print(np.std(length_pcd))

    plt.hist(np.array(length_atom))
    plt.hist(np.array(length_pcd))
    plt.show()





