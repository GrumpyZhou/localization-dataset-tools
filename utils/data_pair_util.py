import matplotlib.pyplot as plt
import numpy as np
from utils.geometry_util import * 
import os

__all__ = ['generate_knn_pairs', 'SampleSchema', 'load_pose', 'sample_indices', 'check_pair_distrb_txt', 
          'check_pair_distrb_npy', 'plot_distrb']

def generate_knn_pairs(base_dir, datasets, K, npy_file, train_txt, test_txt, 
                       pair_type='train', scene_type='indoor', sample_mode=None):
    """Generate relative pose pairs based on the retrieval ranks
    Each generated pair is writen as one line in the target txt file 
    with following format:
    [im1 im2 sim qw qx qy qz x y z e1 ... e9]
    im1 is the retreived database image while im2 is the query image
    (qw, qx, qy, qz), (x, y, z) are the relative rotation in quaternion and translation
    that transform from im1's local coordinates to im2's local coordinates
    (e1, ..., e9) is the vectorized essential matrix calculated by the relative pose
    """
    rank_data = np.load(npy_file).item()  
    rank_key = 'test' if pair_type != 'train' else 'train'    
    sc = SampleSchema(scene_type, sample_mode)
    sampler = sc.sampler
    target_txt_name = '{}_pairs.{}nn.{}.txt'.format(pair_type, K, sc.mode)
    print(sc)
    
    for dataset in datasets:
        np.random.seed(0)  # Make determinisitic 

        # Load absolute pose labels
        train_set, train_list = load_pose(os.path.join(base_dir, dataset, train_txt), with_list=True)
        if pair_type == 'train':
            test_set, test_list = train_set, train_list
        else:
            test_set, test_list = load_pose(os.path.join(base_dir, dataset, test_txt), with_list=True)

        # Load retrieval results
        ranks =  rank_data[dataset][rank_key]['ranks']
        similarity = rank_data[dataset][rank_key]['scores']  
            
        # Generate relative pose pairs
        target_pair_txt = os.path.join(base_dir, dataset, target_txt_name) 
        pair_num = 0
        with open(target_pair_txt, 'w') as pair_txt:
            for i in range(len(test_list)):
                im2 = test_list[i]
                c2, q2 = test_set[im2]
                if sampler is not None:
                    # Perform sampling to improve the diversity in distribution
                    indices = sample_indices(target_num=sampler.num, ratio=sampler.ratio,
                                         total_num=int(len(train_list)*sampler.portion))
                else:
                    # No sampling schema is applied, direct select top-K similar images
                    indices = range(K)
                    if pair_type == 'train':
                        indices = range(K+1) # The self frame will be skipped
                
                count = 0
                for j in indices:
                    score = similarity[j, i]
                    idx = ranks[j, i]
                    im1 = train_list[idx]
                    if im1 == im2:
                        continue # Skip the self frame
                    c1, q1 = train_set[im1]
                    
                    # Calculate relative pose and essential matrix
                    (t12, q12) = cal_relative_pose(c1, c2, q1, q2)
                    ess_vec = cal_essential_matrix(t12, q12)
                    pair = [im1, im2, str(score)] 
                    pair += [str(x) for x in q12] + [str(x) for x in t12] + [str(x) for x in ess_vec] + ['\n']
                    pair_txt.write(' '.join(pair))    
                    count += 1
                    if count >= K:
                        break
                pair_num += count
        pair_txt.close()
        print('Dataset {} write to {}, pair num {}'.format(dataset, target_txt_name, pair_num))
        
class SampleSchema:
    def __init__(self, case='indoor', mode=None):
        self.case = case
        if mode is None:
            self.mode = 'unsampled'
            self.sampler = None
        else:
            self.mode = mode
            if case == 'indoor':  # TODO also increase the num size
                samplers = {'easy': Sampler(portion=0.5, num=4, ratio=[0.6, 0.2, 0.1, 0.1]),
                            'medium' : Sampler(portion=0.5, num=4, ratio=[0.25, 0.35, 0.3, 0.1]),
                            'hard' : Sampler(portion=0.8, num=8, ratio=[0.0, 0.0, 0.2, 0.8])}
            if case == 'outdoor':
                samplers = {'easy': Sampler(portion=0.5, num=40, ratio=[0.7, 0.2, 0.1, 0.0, 0.0, 0.0]),
                            'medium' : Sampler(portion=0.5, num=40, ratio=[0.3, 0.3, 0.3, 0.1, 0.0, 0.0]),
                            'hard' : Sampler(portion=0.8, num=80, ratio=[0.3, 0.2, 0.2, 0.2, 0.1, 0.0])}
            self.sampler = samplers[mode]
            
    def __repr__(self):
        return 'SampleSchema: {}-{}-{}'.format(self.case, self.mode, self.sampler)
    
class Sampler:
    def __init__(self, portion, num, ratio):
        self.portion = portion
        self.num = num
        self.ratio = ratio
        
    def __repr__(self):
        return 'Sampler(portion {}, target number {}, ratios {})'.format(self.portion, self.num, self.ratio)
        
def load_pose(ftxt, with_list=False): 
    '''Load camera pose labels with given ground truth file, e.g., dataset_train.txt'''
    pose_set = {}
    pose_list = []
    with open(ftxt, 'r') as f:
        for line in f:
            if not line.startswith('00') and not line.startswith('seq'):
                continue
            cur = line.split()
            frame = cur[0]
            pos = np.array([float(v) for v in cur[1:4]])
            quat = np.array([float(v) for v in cur[4:8]])
            pose_set[frame] = (pos, quat)
            pose_list.append(frame)
    if not with_list:
        return pose_set
    return pose_set, pose_list

def sampling(target_num, total_num):
    """Sampling target number of indices from total range
    Steps:
        1. split total range equally into n subsets, n is the length of ratio list
        2. for first n-1 subsets randomly pick ration[i]*target_num indices
        3. randomly pick the left indices from the last subset
        4. shuffle all picked indices and return
    """
    split = 4
    base = int(total_num / split)
    indices = []
    ratio = [0.0, 0.0, 0.2, 0.8]
    for i in range(split-1):
        indbase = np.arange(base*i, base*(i+1))
        np.random.shuffle(indbase)
        unit = int(ratio[i]*target_num)
        indices += list(indbase[0: unit])
    indbase = np.arange(base*i, base*(i+1))
    np.random.shuffle(indbase)    
    indices += list(indbase[0: target_num-len(indices)])
    np.random.shuffle(indices)
    return indices

def sample_indices(target_num, ratio, total_num):
    """Sampling target number of indices from total range
    Steps:
        1. split total range equally into n subsets, n is the length of ratio list
        2. for first n-1 subsets randomly pick ration[i]*target_num indices
        3. randomly pick the left indices from the last subset
        4. shuffle all picked indices and return
    """
    split = len(ratio)
    base = int(total_num / split)
    indices = []
    for i in range(split-1):
        indbase = np.arange(base*i, base*(i+1))
        np.random.shuffle(indbase)
        unit = int(ratio[i]*target_num)
        indices += list(indbase[0: unit])
    indbase = np.arange(base*i, base*(i+1))
    np.random.shuffle(indbase)    
    indices += list(indbase[0: target_num-len(indices)])
    np.random.shuffle(indices)
    return indices
    
def check_pair_distrb_txt(pair_txt, abs_pose_set):
    '''Calculate the camera distance and camera angle for each pair in the pair text'''
    cam_dist = []
    quat_ang = []
    with open(pair_txt, 'r') as f:
        for line in f:
            cur = line.split()
            im1, im2 = cur[0], cur[1]
            pose1 = abs_pose_set[im1]
            pose2 = abs_pose_set[im2]
            cam_dist.append(np.linalg.norm(pose1[0] - pose2[0]))
            quat_ang.append(float(cal_quat_angle_error(pose1[1], pose2[1])))
    return cam_dist, quat_ang

def check_pair_distrb_npy(rank_data, K, train_txt, test_txt, sampler):
    train_set, train_list = load_pose(train_txt, with_list=True)
    test_set, test_list = load_pose(test_txt, with_list=True)
    ranks =  rank_data['ranks']
    similarity = rank_data['scores']
    train_num = len(train_set)
    np.random.seed(0)

    # Check pair distribution
    cam_dist = []
    quat_ang = []
    for i in range(len(test_list)):
        im2 = test_list[i]
        if sampler is not None:
            indices = sample_indices(target_num=sampler.num, ratio=sampler.ratio,
                                     total_num=int(train_num*sampler.portion))
        else:
            indices = range(K)
            if train_txt == test_txt:
                indices = range(K+1) # The self frame will be skipped
        count = 0
        for j in indices:
            score = similarity[j, i]
            idx = ranks[j, i]
            im1 = train_list[idx]
            if im1 == im2:
                continue # Skip the self frame
                
            # Calculate distance and angle
            pose1 = train_set[im1]
            pose2 = test_set[im2]
            cam_dist.append(np.linalg.norm(pose1[0] - pose2[0]))
            quat_ang.append(float(cal_quat_angle_error(pose1[1], pose2[1])))  
            count += 1
            if count >= K:
                break  
    return cam_dist, quat_ang

def plot_distrb(cam_dist, quat_ang, ptype='hist', tag='Train'):
    '''Plot histogram/boxplot over the pair distribution results'''
    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(20,4)
    if ptype == 'hist':
        axs[0].hist(cam_dist)
        axs[1].hist(quat_ang)
    else:
        axs[0].boxplot(cam_dist)
        axs[1].boxplot(quat_ang)        
    axs[0 ].set_title('{} - camera center distance'.format(tag))
    axs[1].set_title('{} - camera rotation angle'.format(tag))