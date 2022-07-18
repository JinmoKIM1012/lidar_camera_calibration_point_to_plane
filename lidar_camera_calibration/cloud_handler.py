#!/usr/bin/env python
import json
import logging
import os
from typing import List, Optional

import numpy as np
import open3d as o3d
import pyransac3d as pyrsc

try:
    from typing import Annotated  # type: ignore
except ImportError:
    from typing_extensions import Annotated

from .types import Plane

__all__ = ["CloudHandler"]


logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(name)s %(levelname)s:%(message)s")
_logger = logging.getLogger(__name__)


class CloudHandler:
    def __init__(
        self,
        min_bound: Annotated[List[float], 3],
        max_bound: Annotated[List[float], 3],
        plane_ransac_thresh: float,
        plane_min_points: int,
        debug: bool = False,
    ):
        # pass through filter
        self._min_bound = min_bound
        self._max_bound = max_bound
        self._plane_ransac_thresh = plane_ransac_thresh
        self._plane_min_points = plane_min_points

        self._debug = debug

    def run(self, cloud_file: str) -> Optional[Plane]:
        pcd = o3d.io.read_point_cloud(cloud_file)
        # import pdb; pdb.set_trace()
        file_idx = int(cloud_file[-6:-4])
        if file_idx >= 8:
            R = pcd.get_rotation_matrix_from_xyz((0, 0, np.pi))
            pcd.rotate(R, center=(0, 0, 0))
        assert not pcd.is_empty(), f"failed to read {cloud_file}"

        #xyz = pcd.point["positions"].numpy()
        pass_through_condition = None
        bounding_box = o3d.geometry.AxisAlignedBoundingBox()
        bounding_box.min_bound = np.array(self._min_bound)
        bounding_box.max_bound = np.array(self._max_bound)
        
        #import pdb; pdb.set_trace()
        pcd = pcd.crop(bounding_box)
        # for i in np.arange(3):
        #     cur_dimension_condition = (xyz[:, i] >= self._min_bound[i]) & (xyz[:, i] <= self._max_bound[i])
        #     pass_through_condition = (
        #         pass_through_condition & cur_dimension_condition
        #         if pass_through_condition is not None
        #         else cur_dimension_condition
        #     )
        original = o3d.geometry.PointCloud()
        original.points = pcd.points
        
        # o3d.visualization.draw_geometries([original])
        # labels = pcd.cluster_dbscan(0.2, 3)
        # values, counts = np.unique(labels, return_counts=True)
        # ind = np.argmax(counts)
        # pcd = pcd.select_by_index(np.where(np.array(labels) == values[ind])[0])
        plane_coeffs, inlier_indices = pcd.segment_plane(distance_threshold=0.01,
                                            ransac_n=100,
                                            num_iterations=1000)
        xyz = np.asarray(pcd.points) 
    # [a, b, c, d] = plane_model

    #     # expect points on checkboard form the cluster with most points
    #     labels = pcd.cluster_dbscan(0.2, 3)
    #     values, counts = np.unique(labels, return_counts=True)
    #     ind = np.argmax(counts)
    #     pcd = pcd.select_by_index(np.where(np.array(labels) == values[ind])[0])
    #     xyz = np.asarray(pcd.points)



    #     # fit plane using ransac
    #     plane_fitter = pyrsc.Plane()
    #     plane_coeffs, inlier_indices = plane_fitter.fit(
    #         xyz, thresh=self._plane_ransac_thresh, minPoints=self._plane_min_points, maxIteration=1000
    #     )

        if len(inlier_indices) < self._plane_min_points:
            _logger.info(f"failed to extract plane of {cloud_file}")
            return None
        pcd = pcd.select_by_index(inlier_indices)
        # xyz = xyz[inlier_indices]
        # pcd.points = o3d.utility.Vector3dVector(xyz.copy())

        # remove noise
        pcd_remove, _ = pcd.remove_statistical_outlier(nb_neighbors=25, std_ratio=1.8)
        xyz = np.asarray(pcd.points)
        # import pdb; pdb.set_trace()
        # vis = o3d.visualization.Visualizer()
        # vis.create_window()
        # original.paint_uniform_color([1.0, 0, 1.0])
        # pcd.paint_uniform_color([0, 1.0, 0])
        # pcd_remove.paint_uniform_color([0, 1.0, 1.0])
        # # vis.add_geometry(original)
        # vis.add_geometry(pcd)
        # vis.add_geometry(pcd_remove)

        # opt = vis.get_render_option()
        # opt.background_color = np.asarray([0, 0, 0])
        # vis.run()
        # vis.destroy_window()
        if self._debug:
            cloud_file_name = os.path.join("/tmp", cloud_file.split("/")[-1])
            o3d.io.write_point_cloud(cloud_file_name, pcd)

        return Plane.from_points(xyz)

    @staticmethod
    def from_json(dataset_info_json: str) -> "CloudHandler":
        with open(dataset_info_json, "r") as _file:
            data = json.load(_file)
            for key in (
                "min_bound",
                "max_bound",
                "plane_ransac_thresh",
                "plane_min_points",
                "debug",
            ):
                assert key in data

        return CloudHandler(
            data["min_bound"],
            data["max_bound"],
            data["plane_ransac_thresh"],
            data["plane_min_points"],
            data["debug"],
        )
