import numpy as np
from tqdm import tqdm
import os















if __name__ == "__main__":
    mode = 'train'
    count = 0


    root = '/media/ymz/11c16692-3687-420b-9787-72a7f9e3bdaf/Protein/enzyme_processed/raw/' + mode
    for _, _, files in os.walk(root):
        break

    for item in tqdm(files):
        data = os.path.join(root, item)
        data = np.load(data)
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

        if atom_type.sum() != type.sum():
            print(atom_type.sum(), type.sum(), atom_type.sum()/ type.sum(), type.sum() - atom_type.sum())
            count += 1



    print(count/len(files), count)






