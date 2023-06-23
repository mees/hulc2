import logging

import cv2
import numpy as np

log = logging.getLogger(__name__)


class CamProjections:
    def __init__(
        self,
        intrinsics,
        extrinsic_matrix,
        resolution,
        crop_coords,
        resize_resolution,
        name,
        proj_matrix=None,
    ):
        self.resolution = self.load_res(resolution)
        self.crop_coords = crop_coords
        self.resize_resolution = resize_resolution
        self.name = name
        self.instrinsics = intrinsics
        self.extrinsic_matrix = extrinsic_matrix
        self.intrinsic_matrix = np.array(
            [
                [intrinsics["fx"], 0, intrinsics["cx"]],
                [0, intrinsics["fy"], intrinsics["cy"]],
                [0, 0, 1],
            ]
        )

        self.T_cam_world = np.linalg.inv(self.extrinsic_matrix)
        if proj_matrix is not None:
            self.projection_matrix = proj_matrix
        else:
            self.projection_matrix = self.intrinsic_matrix @ self.T_cam_world[:-1, :]
        self.crop_coords = intrinsics["crop_coords"]
        self.resize_resolution = intrinsics["resize_resolution"]
        self.dist_coeffs = intrinsics["dist_coeffs"]
        self.width = self.instrinsics["width"]
        self.height = self.instrinsics["height"]

    def load_res(self, resolution):
        if "1080" in resolution:
            resolution = (1920, 1080)
        elif "720" in resolution:
            resolution = (1280, 720)
        else:
            return resolution
        return resolution

    def get_intrinsics(self):
        return self.instrinsics

    def get_projection_matrix(self):
        # intr = self.get_intrinsics()
        # cam_mat = np.array([[intr['fx'], 0, intr['cx'], 0],
        #                     [0, intr['fy'], intr['cy'], 0],
        #                     [0, 0, 1, 0]])
        cam_mat = self.projection_matrix
        return cam_mat

    def _crop_and_resize(self, img):
        if self.crop_coords is not None:
            c = self.crop_coords
            img = img[c[0] : c[1], c[2] : c[3]]
        if self.resize_resolution is not None:
            interp = cv2.INTER_NEAREST if len(img.shape) == 2 else cv2.INTER_LINEAR
            img = cv2.resize(img, tuple(self.resize_resolution), interpolation=interp)
        return img

    def revert_crop_and_resize(self, img):
        if self.crop_coords is not None:
            c = self.crop_coords
            res = (c[3] - c[2], c[1] - c[0])
        else:
            res = self.resolution
        if self.resize_resolution is not None:
            interp = cv2.INTER_NEAREST if len(img.shape) == 2 else cv2.INTER_LINEAR
            img = cv2.resize(img, res, interpolation=interp)
        if self.crop_coords is not None:
            if len(img.shape) == 2:
                # case depth image
                new_img = np.zeros(self.resolution[::-1], dtype=img.dtype)
            else:
                # case rgb image
                new_img = np.zeros((*self.resolution[::-1], 3), dtype=img.dtype)
            new_img[c[0] : c[1], c[2] : c[3]] = img
            img = new_img
        return img

    def project(self, X):
        if X.shape[0] == 3:
            if len(X.shape) == 1:
                X = np.append(X, 1)
            else:
                X = np.concatenate([X, np.ones((1, X.shape[1]))], axis=0)

        x = self.get_projection_matrix() @ X
        result = np.round(x[0:2] / x[2]).astype(int)
        # width, height = self.get_intrinsics()["width"], self.get_intrinsics()["height"]
        # if not (0 <= result[0] < width and 0 <= result[1] < height):
        #     log.warning("Projected point outside of image bounds")
        return result[0], result[1]

    def deproject(self, point, depth, homogeneous=False, depth_shape=None):
        """
        Arguments:
            point: (x, y)
            depth: scalar or array, if array index with point
            homogeneous: boolean, return homogenous coordinates
        """
        if depth_shape is None:
            depth_shape = depth.shape

        if depth_shape != self.resolution[::-1]:
            # old_point = point
            # old_depth = depth.copy()
            point_mat = np.zeros(depth_shape)
            point_mat[point[1], point[0]] = 1
            transformed_coords = self.revert_crop_and_resize(point_mat)
            y_candidates, x_candidates = np.where(transformed_coords == 1)
            y_transformed = y_candidates[len(y_candidates) // 2]
            x_transformed = x_candidates[len(x_candidates) // 2]
            point = (x_transformed, y_transformed)

        if not np.isscalar(depth):
            depth = self.revert_crop_and_resize(depth)
        intr = self.get_intrinsics()
        cx = intr["cx"]
        cy = intr["cy"]
        fx = intr["fx"]
        fy = intr["fy"]

        v_crd, u_crd = point

        if np.isscalar(depth):
            Z = depth
        else:
            Z = depth[u_crd, v_crd]

        if Z == 0:
            return None
        X = (v_crd - cx) * Z / fx
        Y = (u_crd - cy) * Z / fy
        if homogeneous:
            return np.array([X, Y, Z, 1])
        else:
            return np.array([X, Y, Z])

    @staticmethod
    def draw_point(img, point, color=(255, 0, 0)):
        img[point[1], point[0]] = color
