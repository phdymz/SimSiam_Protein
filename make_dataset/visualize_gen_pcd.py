import open3d as o3d
import numpy as np
import os
from tqdm import tqdm
#
# if __name__ == "__main__":
#     data = np.load('/media/ymz/11c16692-3687-420b-9787-72a7f9e3bdaf/Protein/site_processed/test/1.npz')
#     src_pcd = data['xyz']
#     tgt_pcd = data['atom']
#
#     pcd0 = o3d.geometry.PointCloud()
#     pcd0.points = o3d.utility.Vector3dVector(src_pcd)
#     pcd0.paint_uniform_color([1, 0, 0])
#
#     pcd1 = o3d.geometry.PointCloud()
#     pcd1.points = o3d.utility.Vector3dVector(tgt_pcd)
#     pcd1.paint_uniform_color([0, 0, 1])
#
#     o3d.visualization.draw_geometries([pcd0, pcd1])


if __name__ == "__main__":
    root = '/media/ymz/11c16692-3687-420b-9787-72a7f9e3bdaf/Protein/site_processed/test'
    for _, _, files in os.walk(root):
        break

    length = []

    for item in tqdm(files):
        data = np.load(os.path.join(root, item))['atom']
        length.append(len(data))

    print(max(length))
