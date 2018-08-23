import numpy as np
from transforms3d.quaternions import quat2mat, mat2quat

__all__ = ['get_world_to_cam', 'cal_quat_angle_error', 'cal_relative_pose', 'cal_essential_matrix']

def get_world_to_cam(cam_to_world):
    '''Extract world-to-camera pose (q, c) from a camera-to-world motion matrix (4x4)'''
    R = cam_to_world[0:3, 0:3]
    q = mat2quat(R.T) 
    c = cam_to_world[0:3, 3] 
    return q, c

def cal_quat_angle_error(label, pred):
    '''Calculate angle between two quaternions'''
    if len(label.shape) == 1:
        label = np.expand_dims(label, axis=0)
    if len(pred.shape) == 1:
        pred = np.expand_dims(pred, axis=0)
    q1 = pred / np.linalg.norm(pred, axis=1, keepdims=True)
    q2 = label / np.linalg.norm(label, axis=1, keepdims=True)
    d = np.around(np.sum(np.multiply(q1,q2), axis=1, keepdims=True), decimals=4)  # around to 0.0001 can assure d <= 1    
    error = 2 * np.degrees(np.arccos(d))
    return error

def cal_relative_pose(c1, c2, q1, q2):
    """Calculate relative pose between two cameras
    Args:
    - c1: absolute position of the first camera
    - c2: absolute position of the second camera
    - q1: orientation quaternion of the first camera
    - q2: orientation quaternion of the second camera
    Return:
    - (t12, q12): relative pose giving the transformation from the 1st camera to the 2nd camera coordinates, 
                  t12 is translation, q12 is relative rotation quaternion 
    """
    r1 = quat2mat(q1)
    r2 = quat2mat(q2)
    r12 = r2.dot(r1.T)
    q12 = mat2quat(r12)
    t12 = r2.dot(c1 - c2)
    return (t12, q12)
          
def cal_essential_matrix(trans_label, quat_label):
    """Calculate essential matrix
    Args:
    - trans: translation vector, size (3,);
    - quat: quaternion vector, size (4,);
    Return:
    - vectorized essential matrix, size (9,)
    """
    t_norm = np.linalg.norm(trans_label)
    if t_norm != 0.0:
        trans_label = trans_label/np.linalg.norm(trans_label)     # force translation to be unit length
    R = quat2mat(quat_label)
    t_skew = hat(trans_label)
    E = t_skew.dot(R)
    return E.flatten().astype(np.float32)

def hat(vec):
    """Skew operator
    Args:
    - vec: vector of size (3,) to be transformed;
    Return: 
    - skew-symmetric matrix of size (3, 3)
    """ 
    [a1, a2, a3] = list(vec)
    skew = np.array([[0, -a3, a2],[a3, 0, -a1],[-a2, a1, 0]])
    return skew
