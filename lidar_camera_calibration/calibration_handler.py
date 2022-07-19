#!/usr/bin/env python
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

__all__ = ["CalibrationHandler"]

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(name)s %(levelname)s:%(message)s")
_logger = logging.getLogger(__name__)


_GREEN = (0, 255, 0)


class CalibrationHandler:
    def __init__(self, dataset_info_json: str, image_lists_path: str, cloud_lists_path: str, data_path: str):
        self._data_loader = DataLoader(image_lists_path, cloud_lists_path, data_path)
        self._cloud_handler = CloudHandler.from_json(dataset_info_json)
        self._image_handler = ImageHandler.from_json(dataset_info_json)

    def run(self):
        all_planes_lidar = list()
        all_planes_camera = list()

        for (image_file, cloud_file) in zip(tqdm(self._data_loader.all_image_files), self._data_loader.all_cloud_files):
            plane_lidar = self._cloud_handler.run(cloud_file)
            if plane_lidar is None:
                continue

            all_planes_lidar.append(plane_lidar)
        
        all_planes_camera = self._image_handler.run_all()

        if len(all_planes_lidar) == 0:
            _logger.warning("failed to extract planes data")
            return None

        pose_SE3 = optimize_pp(all_planes_lidar, all_planes_camera)
        transformation_mat = pose_SE3.as_matrix()

        rpy = np.deg2rad(cv2.RQDecomp3x3(transformation_mat[:3, :3])[0])
        tvec = transformation_mat[:3, 3]


        _logger.info("drawing projected lidar points on image...")
        self._draw_projections_lidar_on_cam(transformation_mat)

        return rpy, tvec

    def _draw_projections_lidar_on_cam(self, transformation_mat):
        import os

        for (image_file, cloud_file) in zip(tqdm(self._data_loader.all_image_files), self._data_loader.all_cloud_files):
            projected_image = self._draw_projection_lidar_on_cam(image_file, cloud_file, transformation_mat)
            if projected_image is None:
                continue
            image_file_name = os.path.join("./tmp", "projected_" + image_file.split("/")[-1])
            cv2.imwrite(image_file_name, projected_image)

    def _draw_projection_lidar_on_cam(self, image_file: str, cloud_file: str, transformation_mat):

        rvec, _ = cv2.Rodrigues(transformation_mat[:3, :3])
        tvec = transformation_mat[:3, 3]
        print(transformation_mat)

        plane_lidar = self._cloud_handler.run(cloud_file)
        if plane_lidar is None:
            return None
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(cloud_file)
        color_merged = np.empty((len(pcd.points), 3, 0))
        file_idx = int(cloud_file[-6:-4])
        # if file_idx >= 8:
        #     R = pcd.get_rotation_matrix_from_xyz((0, 0, np.pi))
        #     pcd.rotate(R, center=(0, 0, 0))
        rot = [-2*np.pi/3, -np.pi/3, 0, np.pi/3, 2*np.pi/3, np.pi]

        pcd_list = []
        if file_idx >= 8:
            for angle_i in range(6):
                # if angle_i != 2:
                #     continue
                #image_file: ../data/00.jpg
                pcd_temp = o3d.io.read_point_cloud(cloud_file)
                image_i = image_file[:-6] + f'20220718_{angle_i + 1}/' + image_file[-6:]
                image = cv2.imread(image_i)
                assert image is not None, f"failed to load {image_i}"
                height, width = image.shape[:2]


                R = pcd_temp.get_rotation_matrix_from_xyz((0, 0, -rot[angle_i]))
                pcd_temp.rotate(R, center=(0, 0, 0))

                img_points, _ = cv2.fisheye.projectPoints(
                    np.expand_dims(np.asarray(pcd_temp.points), -2), rvec, tvec, self._image_handler.camera_info.K, self._image_handler.camera_info.dist_coeffs[:4]
                )
                
                valid = np.where(np.asarray(pcd_temp.points)[:,0]>0)[0]
                
                color = np.zeros((len(img_points), 3))
                print(color[len(color) // 3])
                pcd_temp.colors = o3d.utility.Vector3dVector(color/255)
                # o3d.visualization.draw_geometries([pcd])


                img_points = img_points.astype(int).squeeze(axis=1)

                color = np.zeros((len(img_points), 3)) 
                for i, point in enumerate(img_points):
                    x, y = point
                    if x < 0 or x > width - 1:
                        continue
                    if y < 0 or y > height - 1:
                        continue
                    if i not in valid:
                        continue
                    color[i] = np.flip(image[y, x])
                
                for i, point in enumerate(np.asarray(pcd_temp.points)):
                    if np.abs(point[1]/point[0]) > 1/np.sqrt(3):
                        color[i] = np.array([0., 0., 0.])

                R = pcd_temp.get_rotation_matrix_from_xyz((0, 0, rot[angle_i]))
                pcd_temp.rotate(R, center=(0, 0, 0))
                pcd_temp.colors = o3d.utility.Vector3dVector(color/255)
                
                o3d.io.write_point_cloud(f'./tmp/{file_idx}/{file_idx}_{angle_i}.pcd', pcd_temp)
                color_merged = np.concatenate((color_merged, np.expand_dims(color, axis=2)), axis=2)
            
            
            valid_map = color_merged != 0 # n*6
            valid_map = valid_map.any(axis=1)
            
            total_color = np.zeros_like(np.array(pcd.points))

            for idx, colors in enumerate(color_merged): 
                #import pdb; pdb.set_trace()    
                valid_colors = colors.T[np.where(valid_map[idx])]
                if len(valid_colors) > 0:
                    total_color[idx] = valid_colors.mean(axis=0)
                else : 
                    total_color[idx] = np.array([0., 0., 0.]) # 3*6 
                #print(valid_map[idx])

            pcd.colors = o3d.utility.Vector3dVector(total_color/255)
            o3d.io.write_point_cloud(f'./tmp/{file_idx}.pcd', pcd)
 

        # image = cv2.imread(image_file)
        # assert image is not None, f"failed to load {image_file}"

        # rvec, _ = cv2.Rodrigues(transformation_mat[:3, :3])
        # tvec = transformation_mat[:3, 3]

        # plane_lidar = self._cloud_handler.run(cloud_file)
        # if plane_lidar is None:
        #     return None

        # points = plane_lidar.projections
        # projected_points, _ = cv2.fisheye.projectPoints(
        #     np.expand_dims(np.asarray(points), -2), rvec, tvec, self._image_handler.camera_info.K, self._image_handler.camera_info.dist_coeffs[:4]
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

        # return image

    @staticmethod
    def get_transformation_matrix(rpy: Array[Tuple[Literal[3]], float], tvec: Array[Tuple[Literal[3]], float]):
        transformation_mat = np.eye(4)
        rotation_mat = Rotation.from_euler("xyz", rpy).as_matrix()
        transformation_mat[:3, :3] = rotation_mat
        transformation_mat[:3, 3] = tvec

        return transformation_mat
