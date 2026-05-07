# geometry.py
import numpy as np
from typing import Tuple 
import cv2 

def img3d_to_ptcloud(img3d):
    ptcloud = img3d.copy().reshape(-1, 3)
    return ptcloud[~np.isnan(ptcloud).any(axis=1)]

def transform_pts(pts3d: np.ndarray, RT: np.ndarray) -> np.ndarray:
    """transform points using RT homogeneous matrix

    Args:
        pts3d (np.ndarray): Nx3 array containing 3d point coordinates
        RT (np.ndarray): 4x4 homogeneous transformation matrix

    Returns:
        np.ndarray: Nx3 transformed pts3d points according to RT
    """
    pts3d_h = np.hstack((pts3d, np.ones((pts3d.shape[0], 1))))
    rotated_pts3d_h = (RT @ pts3d_h.T).T
    rotated_pts3d = rotated_pts3d_h[:, :3] / (rotated_pts3d_h[:, 3].reshape(-1, 1))
    return rotated_pts3d

def create_RT(R: np.ndarray = np.eye(3), T: np.ndarray = np.zeros(3)) -> np.ndarray:
    """Create 4x4 homogeneous transformation matrix

    Args:
        R (np.ndarray, optional): 3x3 rotation matrix. Defaults to np.eye(3).
        T (np.ndarray, optional): translation vector. Defaults to np.zeros(3).

    Returns:
        np.ndarray: 4x4 homogeneous transformation matrix
    """
    RT = np.eye(4)
    RT[:3, :3] = R.copy()
    RT[:3, 3] = T.reshape(3).copy()
    return RT

def project_pts(pts3d: np.ndarray, P: np.ndarray) -> np.ndarray:
    """project 3d points to image, according to projection matrix P

    Args:
        pts3d (np.ndarray): Nx3 array containing 3d points
        P (np.ndarray): projection matrix

    Returns:
        np.ndarray: Nx2 array containing pixel coordinates of projected points.
    """
    # convert to homogeneous
    pts3d_h = np.hstack((pts3d, np.ones((pts3d.shape[0], 1))))
    projected_pts = (P @ pts3d_h.T).T
    # convert from homogeneous coordinates.
    projected_pts = projected_pts[:, :2] / projected_pts[:, 2].reshape(-1, 1)
    return projected_pts


def ptcloud_to_img3d(
    ptcloud: np.ndarray, K: np.ndarray, D: np.ndarray, size: Tuple[int, int]
) -> np.ndarray:
    """converts a pointcloud to 3DImage
    
    Converts a pointcloud to a 3 channel 3D image format, similar to what is 
    used to store ground truth information in SCARED. The resulting 3D image is 
    expressed in the same frame of reference with the pointcloud, thus if the
    point cloud is not expressed in the original frame of reference, the output
    of this function can be used to evaluate on the reference data. Each point 
    of the pointcloud is projected to the the image frame based on the calibration
    parameters and distortions are also supported.

    Args:
        ptcloud (np.ndarray): N element pointcloud represented as a Nx3 array
        K (np.ndarray): Camera matrix of the target projection view
        D (np.ndarray): Distortion coefficients of the target projection view
        size (Tuple[int, int]): Height, Width of the resulting 3D image.

    Returns:
        np.ndarray: HxWx3 3D Image, each pixel encodes the projection location 
        of the point it stores as a 3D vector.
    """
    h, w = size
    img3d = np.full((h, w, 3), fill_value=np.nan)

    if np.sum(D) ==0: 
        #in case there is no distortion matrix, save time by just projection the points
        projection_coordinates = project_pts(ptcloud, np.hstack((K, np.zeros(3).reshape(3,1))))

    else:
        try:
            projection_coordinates = cv2.projectPoints(ptcloud, np.eye(3), np.zeros(3), K, D)[
                0
            ].squeeze()
        except:
            breakpoint()
    # get the projection coordinates, round them and check which of the points
    # end up within the image view.
    projection_coordinates = np.round(projection_coordinates)
    valid_projection_indexes = (
        (projection_coordinates[:, 0] >= 0)
        & (projection_coordinates[:, 0] < w)
        & (projection_coordinates[:, 1] >= 0)
        & (projection_coordinates[:, 1] < h)
    )
    projection_coordinates = projection_coordinates[valid_projection_indexes].astype(
        int
    )
    projected_3dpoints = ptcloud[valid_projection_indexes]

    xs, ys = projection_coordinates[:, 0], projection_coordinates[:, 1]

    img3d[ys, xs] = projected_3dpoints
    return img3d



def ptcloud_to_depthmap(
    ptcloud: np.ndarray, K: np.ndarray, D: np.ndarray, size: Tuple[int, int]
) -> np.ndarray:
    """Convert pointcloud to depthmap

    Converts a pointcloud to a depthamp. The function first creates a 3D image
    based on the provided calibration parameters return the last channel. Depthmaps
    are expressed in the same frame of reference with the pointcloud, thus if the
    point cloud is not expressed in the original frame of reference, the output
    of this function can be used to evaluate on the reference data. Each point 
    of the pointcloud is projected to the the image frame based on the calibration
    parameters and distortions are also supported.
    
    
    Args:
        ptcloud (np.ndarray): N element pointcloud represented as a Nx3 array
        K (np.ndarray): Camera matrix of the target projection view
        D (np.ndarray): Distortion coefficients of the target projection view
        size (Tuple[int, int]): Height, Width of the resulting depthmap.

    Returns:
        np.ndarray: HxW float depthmap
    """
    img3d = ptcloud_to_img3d(ptcloud, K, D, size)
    return img3d[..., 2].copy()


def ptcloud_to_disparity(
    pt_cloud: np.ndarray, P1: np.ndarray, P2: np.ndarray, size: Tuple[int, int]
) -> np.ndarray:
    """Converts point clouds to disparity
    
    Convert point cloud to disparity maps with subpixel accuracy. The function
    takes the provided point cloud and project it two stereo rectified views 
    based on the projection matrices P1, P2 which describe the projection to the
    left and right rectified frames of reference respectively. projection
    pixel coordinates in the disparity image are defined as the rounded 
    projection coordinates of the point cloud to the left frame of reference.
    Disparity is defined as the horizontal difference of the projection of a 
    between the left and right rectified frame of defence. 

    Args:
        pt_cloud (np.ndarray): N element pointcloud stored as a Nx3 array 
        P1 (np.ndarray): Projection matrix of the left rectified frame of reference
        P2 (np.ndarray): Projection matrix of the right rectified frame of reference
        size (Tuple[int, int]): height, width of the resulting disparity image.

    Returns:
        np.ndarray: HxW disparity float disparity array.
    """

    h, w = size
    disparity = np.zeros(size)
    projection_l = project_pts(pt_cloud, P1).reshape(-1, 2)
    projection_r = project_pts(pt_cloud, P2).reshape(-1, 2)
    disparities = (projection_l - projection_r)[:, 0]
    # find all points that project inside the image domain.
    projection_l = np.round(projection_l)
    valid_indexes = (
        (projection_l[:, 0] >= 0)
        & (projection_l[:, 0] < w)
        & (projection_l[:, 1] >= 0)
        & (projection_l[:, 1] < h)
    )
    disparity_idxs = projection_l[valid_indexes].astype(int)
    valid_disparities = disparities[valid_indexes]
    xs, ys = disparity_idxs[:, 0], disparity_idxs[:, 1]
    np.maximum.at(disparity, (ys, xs), valid_disparities)
    return disparity