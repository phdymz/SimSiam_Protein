import numpy as np
import os
from tqdm import tqdm


labels_dict = {"ADP": 1, "COA": 2, "FAD": 3, "HEM": 4, "NAD": 5, "NAP": 6, "SAM": 7}



if __name__ == "__main__":
    count = np.zeros(7)
    #
    # for _, _, files in os.walk('/media/ymz/11c16692-3687-420b-9787-72a7f9e3bdaf/Protein/Ligand/processed/test'):
    #     break
    #
    # for item in files:
    #     root = os.path.join('/media/ymz/11c16692-3687-420b-9787-72a7f9e3bdaf/Protein/Ligand/processed/test', item)
    #     data = np.load(root, allow_pickle=True)
    #     label = int(data['label'])
    #     count[label] += 1
    #
    #
    # print(count)
    #

    count = np.zeros(7)
    test_list = np.load('/home/ymz/桌面/protein/masif/data/masif_ligand/lists/test_pdbs_sequence.npy')
    for item in tqdm(test_list):
        item  = item.decode('utf-8')
        data_type = os.path.join(
            '/media/ymz/11c16692-3687-420b-9787-72a7f9e3bdaf/Protein/Ligand/masif_ligand_pdbs_and_ply_files/00c-ligand_coords',
            item.split('_')[0] + '_ligand_types.npy')

        type = np.load(data_type)
        data_type = []
        for i in range(len(type)):
            data_type.append(type[i].decode('utf-8'))
        data_type = np.array(data_type)

        if len(data_type) > 0:
            if (data_type == data_type[0]).sum() == len(data_type):
                label = int(labels_dict[data_type[0]] - 1)
                count[label] += 1

    print(count)
    print(count.sum())















