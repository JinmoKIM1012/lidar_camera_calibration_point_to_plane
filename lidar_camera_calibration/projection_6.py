import logging
from typing import Tuple

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from .cloud_handler import CloudHandler
from .data_loader import DataLoader
from .image_handler import ImageHandler
from .point_to_plane_optimization import optimize as optimize_pp
from .types import Array

try:
    from typing import Literal  # type: ignore
except ImportError:
    from typing_extensions import Literal

def _draw_projection_lidar_on_cam(self, image_file: str, pcd, transformation_mat):
    
    image = cv2.imread(image_file)
    assert image is not None, f"failed to load {image_file}"

    rvec, _ = cv2.Rodrigues(transformation_mat[:3, :3])
    tvec = transformation_mat[:3, 3]
    print(transformation_mat)

    plane_lidar = self._cloud_handler.run(cloud_file)
    if plane_lidar is None:
        return None
    height, width = image.shape[:2]
    import open3d as o3d
    pcd = o3d.io.read_point_cloud(cloud_file)
    #file_idx = int(cloud_file[-6:-4])
    # if file_idx >= 8:
    #     R = pcd.get_rotation_matrix_from_xyz((0, 0, np.pi))
    #     pcd.rotate(R, center=(0, 0, 0))
    img_points, _ = cv2.projectPoints(
        np.asarray(pcd.points), rvec, tvec, self._image_handler.camera_info.K, self._image_handler.camera_info.dist_coeffs
    )
    
    valid = np.where(np.asarray(pcd.points)[:,0]>0)[0]
    
    color = np.zeros((len(img_points), 3))
    print(color[len(color) // 3])
    pcd.colors = o3d.utility.Vector3dVector(color/255)
    # o3d.visualization.draw_geometries([pcd])

    points = plane_lidar.projections
    projected_points, _ = cv2.projectPoints(
        points, rvec, tvec, self._image_handler.camera_info.K, self._image_handler.camera_info.dist_coeffs
    )
    projected_points = projected_points.astype(int).squeeze(axis=1)

    img_points = img_points.astype(int).squeeze(axis=1)

    color = np.zeros((len(img_points), 3)) 
    for i, point in enumerate(img_points):
        x, y = point
        # print(width, height)
        if x < 0 or x > width - 1:
            continue
        if y < 0 or y > height - 1:
            continue
        if np.abs(y/x) > 1/np.sqrt(3):
            continue
        if i not in valid:
            continue
        #import pdb; pdb.set_trace()
        color[i] = np.flip(image[y, x])
        # import pdb; pdb.set_trace()
        # image = cv2.circle(image, point, radius=0, color=_GREEN, thickness=10)
    # pcd.colors = o3d.utility.Vector3dVector(color/255)
    # o3d.visualization.draw_geometries([pcd])

    pcd.colors = o3d.utility.Vector3dVector(color/255)
    #o3d.io.write_point_cloud(f'./tmp/{file_idx}.pcd', pcd)

    # image = cv2.imread(image_file)
    # assert image is not None, f"failed to load {image_file}"

    # rvec, _ = cv2.Rodrigues(transformation_mat[:3, :3])
    # tvec = transformation_mat[:3, 3]

    # plane_lidar = self._cloud_handler.run(cloud_file)
    # if plane_lidar is None:
    #     return None

    # points = plane_lidar.projections
    # projected_points, _ = cv2.projectPoints(
    #     points, rvec, tvec, self._image_handler.camera_info.K, self._image_handler.camera_info.dist_coeffs
    # )
    # projected_points = projected_points.astype(int).squeeze(axis=1)

    # height, width = image.shape[:2]
    # for point in projected_points:
    #     x, y = point
    #     if x < 0 or x > width - 1:
    #         continue
    #     if y < 0 or y > height - 1:
    #         continue
    #     image = cv2.circle(image, point, radius=0, color=_GREEN, thickness=10)

    return pcd

def rotate_and_paste(pcd, images, transformation_mat):
    import open3d as o3d
    rot = [-2*np.pi/3, -np.pi/3, 0, np.pi/3, 2*np.pi/3, np.pi]
    new_points = []
    new_colors = []
    for i, img in enumerate(images):
        R = pcd.get_rotation_matrix_from_xyz((0, 0, rot[i]))
        pcd.rotate(R, center=(0, 0, 0))
        pcd = _draw_projection_lidar_on_cam(img, pcd, transformation_mat)
        pcd_points = np.array(pcd.points)
        
        new_pcd = o3d.geometry.PointCloud()
        new_pcd.points = o3d.utility.Vector3dVector(pcd_points)
        new_pcd.colors = o3d.utility.Vector3dVector(pcd_colors)
        R_inv = new_pcd.get_rotation_matrix_from_xyz((0, 0, -rot[i]))
        new_pcd.rotate(R_inv, center=(0,0,0))
    
if __name__ == '__main__':
    transformation_matrix = np.array([[-4.10490434e-03,  2.19886456e-02,  9.99749793e-01, -1.55191622e-01], \
                                    [ 2.09884828e-01, -9.77470394e-01,  2.23604019e-02, -2.46286428e-02], \
                                    [ 9.77717499e-01,  2.09924101e-01, -6.02660644e-04, -8.60130535e-02], \
                                    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
    