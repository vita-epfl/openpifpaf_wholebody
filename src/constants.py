import numpy as np
import copy

body_foot_skeleton = [
    (16, 14), (14, 12), (17, 15), (15, 13), (12, 13), (6, 12), (7, 13),
    (6, 7), (6, 8), (7, 9), (8, 10), (9, 11), (2, 3), (1, 2), (1, 3),
    (2, 4), (3, 5), (4, 6), (5, 7),
    (16, 20), (16, 19), (16, 18),    # left foot
    (17, 23), (17, 21), (17, 22)     # right foot
]

face_skeleton = [
    (25,5), (39,4), # ear to ear body
    (54, 1), #nose to nose body
    (60, 3), (3, 63), (66, 2), (2, 69), # eyes to eyes body 
    ] + [(x,x+1) for x in range(24, 40)] + [ #face outline
    (24,41), (41,42), (42,43), (43,44), (44,45), (45,51), #right eyebrow
    (40,50), (50,49), (49,48), (48,47), (47,46), (46,51), #left eyebrow
    (24,60), (60,61), (61,62), (62,63), (63,51), (63,64), (64,65), (65,60), #right eye
    (40,69), (69,68), (68,67), (67,66), (66,51), (66,71), (71,70), (70,69), #left eye
    ] + [(x,x+1) for x in range(51, 59)] + [ (59, 54), #nose
    (57, 75), (78,36), (72, 28), (72,83)] + [(x,x+1) for x in range(72, 83)] + [ # mouth outline
    (72, 84), (84,85), (85,86), (86,87), (87,88), (88,78), #upper lip
    (72, 91), (91,90), (90,89), (89,78) #lower lip
    ]
                                                                                
lefthand_skeleton = [
    (92, 10), #connect to wrist
    (92,93), (92, 97), (92,101), (92,105), (92, 109) #connect to finger starts
    ] + [(x,x+1) for s in [93,97,101,105,109] for x in range(s, s+3)] #four finger                                                                         

righthand_skeleton = [
    (113, 11), #connect to wrist
    (113,114), (113, 118), (113,122), (113,126), (113, 130) #connect to finger starts
    ] + [(x,x+1) for s in [114,118,122,126,130] for x in range(s, s+3)] #four finger                                                                      

WHOLEBODY_SKELETON = body_foot_skeleton + face_skeleton + lefthand_skeleton + righthand_skeleton

body_kps =[
    'nose',            # 1
    'left_eye',        # 2
    'right_eye',       # 3
    'left_ear',        # 4
    'right_ear',       # 5
    'left_shoulder',   # 6
    'right_shoulder',  # 7
    'left_elbow',      # 8
    'right_elbow',     # 9
    'left_wrist',      # 10
    'right_wrist',     # 11
    'left_hip',        # 12
    'right_hip',       # 13
    'left_knee',       # 14
    'right_knee',      # 15
    'left_ankle',      # 16
    'right_ankle',     # 17
    ]

foot_kps= [
    'left_big_toe',    # 18
    'left_small_toe',  # 19
    'left_heel',       # 20
    'right_big_toe',   # 21
    'right_small_toe', # 22
    'right_heel',      # 23
]

face_kps = ['f_' + str(x) for x in range(24, 92)]
lefth_kps = ['lh_' + str(x) for x in range(92, 113)]
righth_kps = ['rh_' + str(x) for x in range(113, 134)]

WHOLEBODY_KEYPOINTS = body_kps + foot_kps + face_kps + lefth_kps + righth_kps

body_pose = np.array([
    [0.0, 9.3, 2.0],    # 'nose',            # 1
    [-0.35, 9.7, 2.0],  # 'left_eye',        # 2
    [0.35, 9.7, 2.0],   # 'right_eye',       # 3
    [-0.7, 9.5, 2.0],   # 'left_ear',        # 4
    [0.7, 9.5, 2.0],    # 'right_ear',       # 5
    [-1.4, 8.0, 2.0],   # 'left_shoulder',   # 6
    [1.4, 8.0, 2.0],    # 'right_shoulder',  # 7
    [-1.75, 6.0, 2.0],  # 'left_elbow',      # 8
    [1.75, 6.2, 2.0],   # 'right_elbow',     # 9
    [-1.75, 4.0, 2.0],  # 'left_wrist',      # 10
    [1.75, 4.2, 2.0],   # 'right_wrist',     # 11
    [-1.26, 4.0, 2.0],  # 'left_hip',        # 12
    [1.26, 4.0, 2.0],   # 'right_hip',       # 13
    [-1.4, 2.0, 2.0],   # 'left_knee',       # 14
    [1.4, 2.1, 2.0],    # 'right_knee',      # 15
    [-1.4, 0.0, 2.0],   # 'left_ankle',      # 16
    [1.4, 0.1, 2.0],    # 'right_ankle',     # 17
    ])

foot_pose = np.array([
    [-1.3, -0.2, 3.0],   #'left_big_toe',    # 18
    [-1.6, -0.2, 2.9],   #'left_small_toe',  # 19
    [-1.4, -0.1, 1.9],   #'left_heel',       # 20
    [1.3, -0.2, 3.0],    #'right_big_toe',   # 21
    [1.6, -0.2, 2.9],    #'right_small_toe', # 22
    [1.4, -0.1, 1.9],    #'right_heel',      # 23
    ])

face_pose = np.array([
    #face outline
    [0.7, 9.7, 2.0], #24
    [0.7, 9.55, 2.0], #25
    [0.6, 9.3, 2.0], #26
    [0.5, 9.05, 2.0], #27
    [0.4, 8.8, 2.0], #28
    [0.3, 8.65, 2.0], #29
    [0.2, 8.55, 2.0], #30
    [0.1, 8.45, 2.0], #31
    [0.0, 8.4, 2.0], #32
    [-0.1, 8.45, 2.0], #33
    [-0.2, 8.55, 2.0], #34
    [-0.3, 8.65, 2.0], #35
    [-0.4, 8.8, 2.0], #36
    [-0.5, 9.05, 2.0], #37
    [-0.6, 9.3, 2.0], #38
    [-0.7, 9.55, 2.0], #39
    [-0.7, 9.7, 2.0], #40
    #eyebrows
    [0.6, 9.8, 2.0], #41
    [0.5, 9.9, 2.0], #42
    [0.4, 9.95, 2.0], #43
    [0.3, 9.9, 2.0], #44
    [0.2, 9.85, 2.0], #45
    [-0.2, 9.85, 2.0], #46
    [-0.3, 9.9, 2.0], #47
    [-0.4, 9.95, 2.0], #48
    [-0.5, 9.9, 2.0], #49
    [-0.6, 9.8, 2.0], #50
    #nose
    [0.0, 9.7, 2.0], #51
    [0.0, 9.566, 2.0], #52
    [0.0, 9.433, 2.0], #53
    [0.0, 9.3, 2.0], #54 equivalent to nose kp from body
    [0.2, 9.2, 2.0], #55
    [0.1, 9.15, 2.0], #56
    [0.0, 9.1, 2.0], #57
    [-0.1, 9.15, 2.0], #58
    [-0.2, 9.2, 2.0], #59
    #eyes
    [0.45, 9.7, 2.0], #60
    [0.4, 9.75, 2.0], #61
    [0.3, 9.75, 2.0], #62
    [0.2, 9.7, 2.0], #63
    [0.3, 9.65, 2.0], #64
    [0.4, 9.65, 2.0], #65
    [-0.2, 9.7, 2.0], #66
    [-0.3, 9.75, 2.0], #67
    [-0.4, 9.75, 2.0], #68
    [-0.45, 9.7, 2.0], #69
    [-0.4, 9.65, 2.0], #70
    [-0.3, 9.65, 2.0], #71
    #mouth
    [0.3, 8.8, 2.0], #72
    [0.2, 8.85, 2.0], #73
    [0.1, 8.9, 2.0], #74
    [0.0, 8.85, 2.0], #75
    [-0.1, 8.9, 2.0], #76
    [-0.2, 8.85, 2.0], #77
    [-0.3, 8.8, 2.0], #78
    [-0.2, 8.75, 2.0], #79
    [-0.1, 8.7, 2.0], #80
    [0.0, 8.65, 2.0], #81
    [0.1, 8.7, 2.0], #82
    [0.2, 8.75, 2.0], #83
    [0.2, 8.82, 2.0], #84
    [0.1, 8.82, 2.0], #85
    [0.0, 8.82, 2.0], #86
    [-0.1, 8.82, 2.0], #87
    [-0.2, 8.82, 2.0], #88
    [-0.1, 8.79, 2.0], #89
    [0.0, 8.79, 2.0], #90
    [0.1, 8.79, 2.0] #91
    ])

lefthand_pose=np.array([
    [-1.75, 3.9, 2.0], #92
    [-1.65, 3.8, 2.0], #93
    [-1.55, 3.7, 2.0], #94
    [-1.45, 3.6, 2.0], #95
    [-1.35, 3.5, 2.0], #96
    [-1.65, 3.5, 2.0], #97
    [-1.65, 3.4, 2.0], #98
    [-1.65, 3.3, 2.0], #99
    [-1.65, 3.2, 2.0], #100
    [-1.75, 3.5, 2.0], #101
    [-1.75, 3.4, 2.0], #102
    [-1.75, 3.3, 2.0], #103
    [-1.75, 3.2, 2.0], #104
    [-1.85, 3.5, 2.0], #105
    [-1.85, 3.4, 2.0], #106
    [-1.85, 3.3, 2.0], #107
    [-1.85, 3.2, 2.0], #108
    [-1.95, 3.5, 2.0], #109
    [-1.95, 3.433, 2.0], #110
    [-1.95, 3.366, 2.0], #111
    [-1.95, 3.3, 2.0], #112
    ])

righthand_pose = copy.deepcopy(lefthand_pose)
righthand_pose[:,0] = -lefthand_pose[:,0]

# [width, height, depth]
WHOLEBODY_STANDING_POSE = np.vstack((body_pose, foot_pose, face_pose, lefthand_pose, righthand_pose))
  
HFLIP = {
    'left_eye': 'right_eye',
    'right_eye': 'left_eye',
    'left_ear': 'right_ear',
    'right_ear': 'left_ear',
    'left_shoulder': 'right_shoulder',
    'right_shoulder': 'left_shoulder',
    'left_elbow': 'right_elbow',
    'right_elbow': 'left_elbow',
    'left_wrist': 'right_wrist',
    'right_wrist': 'left_wrist',
    'left_hip': 'right_hip',
    'right_hip': 'left_hip',
    'left_knee': 'right_knee',
    'right_knee': 'left_knee',
    'left_ankle': 'right_ankle',
    'right_ankle': 'left_ankle',
    'left_big_toe': 'right_big_toe',
    'right_big_toe': 'left_big_toe',
    'right_small_toe': 'left_small_toe',
    'left_small_toe': 'right_small_toe',
    'left_heel': 'right_heel',
    'right_heel': 'left_heel',
    'f_24': 'f_40',
    'f_40': 'f_24',
    'f_25': 'f_39',
    'f_39': 'f_25',
    'f_26': 'f_38',
    'f_38': 'f_26',
    'f_27': 'f_37',
    'f_37': 'f_27',
    'f_28': 'f_36',
    'f_36': 'f_28',
    'f_29': 'f_35',
    'f_35': 'f_29',
    'f_30': 'f_34',
    'f_34': 'f_30',
    'f_31': 'f_33',
    'f_33': 'f_31',
    'f_41': 'f_50',
    'f_50': 'f_41',
    'f_42': 'f_49',
    'f_49': 'f_42',
    'f_43': 'f_48',
    'f_48': 'f_43',
    'f_44': 'f_47',
    'f_47': 'f_44',
    'f_45': 'f_46',
    'f_46': 'f_45',
    'f_55': 'f_59',
    'f_59': 'f_55',
    'f_56': 'f_58',
    'f_58': 'f_56',
    'f_60': 'f_69',
    'f_69': 'f_60',
    'f_61': 'f_68',
    'f_68': 'f_61',
    'f_62': 'f_67',
    'f_67': 'f_62',
    'f_63': 'f_66',
    'f_66': 'f_63',
    'f_64': 'f_71',
    'f_71': 'f_64',
    'f_65': 'f_70',
    'f_70': 'f_65',
    'f_72': 'f_78',
    'f_78': 'f_72',
    'f_73': 'f_77',
    'f_77': 'f_73',
    'f_74': 'f_76',
    'f_76': 'f_74',
    'f_79': 'f_83',
    'f_83': 'f_79',
    'f_82': 'f_80',
    'f_80': 'f_82',
    'f_84': 'f_88',
    'f_88': 'f_84',
    'f_85': 'f_87',
    'f_87': 'f_85',
    'f_89': 'f_91',
    'f_91': 'f_89',
    'lh_92': 'rh_113',
    'rh_113': 'lh_92',
    'lh_93': 'rh_114',
    'rh_114': 'lh_93',
    'lh_94': 'rh_115',
    'rh_115': 'lh_94',
    'lh_95': 'rh_116',
    'rh_116': 'lh_95',
    'lh_96': 'rh_117',
    'rh_117': 'lh_96',
    'lh_97': 'rh_118',
    'rh_118': 'lh_97',
    'lh_98': 'rh_119',
    'rh_119': 'lh_98',
    'lh_99': 'rh_120',
    'rh_120': 'lh_99',
    'lh_100': 'rh_121',
    'rh_121': 'lh_100',
    'lh_101': 'rh_122',
    'rh_122': 'lh_101',
    'lh_102': 'rh_123',
    'rh_123': 'lh_102',
    'lh_103': 'rh_124',
    'rh_124': 'lh_103',
    'lh_104': 'rh_125',
    'rh_125': 'lh_104',
    'lh_105': 'rh_126',
    'rh_126': 'lh_105',
    'lh_106': 'rh_127',
    'rh_127': 'lh_106',
    'lh_107': 'rh_128',
    'rh_128': 'lh_107',
    'lh_108': 'rh_129',
    'rh_129': 'lh_108',
    'lh_109': 'rh_130',
    'rh_130': 'lh_109',
    'lh_110': 'rh_131',
    'rh_131': 'lh_110',
    'lh_111': 'rh_132',
    'rh_132': 'lh_111',
    'lh_112': 'rh_133',
    'rh_133': 'lh_112'
}

# SIGMAS as in https://github.com/jin-s13/COCO-WholeBody/blob/master/evaluation/myeval_wholebody.py
body = [.026, .025, .025, .035, .035, .079, .079, .072, .072, .062, .062, 0.107, 0.107, .087, .087, .089, .089]
foot = [0.068, 0.066, 0.066, 0.092, 0.094, 0.094]
face = [0.042, 0.043, 0.044, 0.043, 0.040, 0.035, 0.031, 0.025, 0.020, 0.023, 0.029, 0.032, 0.037, 0.038, 0.043,
        0.041, 0.045, 0.013, 0.012, 0.011, 0.011, 0.012, 0.012, 0.011, 0.011, 0.013, 0.015, 0.009, 0.007, 0.007,
        0.007, 0.012, 0.009, 0.008, 0.016, 0.010, 0.017, 0.011, 0.009, 0.011, 0.009, 0.007, 0.013, 0.008, 0.011,
        0.012, 0.010, 0.034, 0.008, 0.008, 0.009, 0.008, 0.008, 0.007, 0.010, 0.008, 0.009, 0.009, 0.009, 0.007,
        0.007, 0.008, 0.011, 0.008, 0.008, 0.008, 0.01, 0.008]
lefthand = [0.029, 0.022, 0.035, 0.037, 0.047, 0.026, 0.025, 0.024, 0.035, 0.018, 0.024, 0.022, 0.026, 0.017,
            0.021, 0.021, 0.032, 0.02, 0.019, 0.022, 0.031]
righthand = [0.029, 0.022, 0.035, 0.037, 0.047, 0.026, 0.025, 0.024, 0.035, 0.018, 0.024, 0.022, 0.026, 0.017,
            0.021, 0.021, 0.032, 0.02, 0.019, 0.022, 0.031]

WHOLEBODY_SIGMAS = np.array(body + foot + face + lefthand + righthand)

WHOLEBODY_SCORE_WEIGHTS = [3.0] * 3 + [1.0] * (len(WHOLEBODY_KEYPOINTS) - 3)


COCO_CATEGORIES = [
    'person',
    'bicycle',
    'car',
    'motorcycle',
    'airplane',
    'bus',
    'train',
    'truck',
    'boat',
    'traffic light',
    'fire hydrant',
    'street sign',
    'stop sign',
    'parking meter',
    'bench',
    'bird',
    'cat',
    'dog',
    'horse',
    'sheep',
    'cow',
    'elephant',
    'bear',
    'zebra',
    'giraffe',
    'hat',
    'backpack',
    'umbrella',
    'shoe',
    'eye glasses',
    'handbag',
    'tie',
    'suitcase',
    'frisbee',
    'skis',
    'snowboard',
    'sports ball',
    'kite',
    'baseball bat',
    'baseball glove',
    'skateboard',
    'surfboard',
    'tennis racket',
    'bottle',
    'plate',
    'wine glass',
    'cup',
    'fork',
    'knife',
    'spoon',
    'bowl',
    'banana',
    'apple',
    'sandwich',
    'orange',
    'broccoli',
    'carrot',
    'hot dog',
    'pizza',
    'donut',
    'cake',
    'chair',
    'couch',
    'potted plant',
    'bed',
    'mirror',
    'dining table',
    'window',
    'desk',
    'toilet',
    'door',
    'tv',
    'laptop',
    'mouse',
    'remote',
    'keyboard',
    'cell phone',
    'microwave',
    'oven',
    'toaster',
    'sink',
    'refrigerator',
    'blender',
    'book',
    'clock',
    'vase',
    'scissors',
    'teddy bear',
    'hair drier',
    'toothbrush',
    'hair brush',
]


def draw_ann(ann, *, keypoint_painter, filename=None, margin=0.5, aspect=None, **kwargs):
    from openpifpaf import show  # pylint: disable=import-outside-toplevel

    bbox = ann.bbox()
    xlim = bbox[0] - margin, bbox[0] + bbox[2] + margin
    ylim = bbox[1] - margin, bbox[1] + bbox[3] + margin
    if aspect == 'equal':
        fig_w = 5.0
    else:
        fig_w = 5.0 / (ylim[1] - ylim[0]) * (xlim[1] - xlim[0])

    with show.canvas(filename, figsize=(fig_w, 5), nomargin=True, **kwargs) as ax:
        ax.set_axis_off()
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

        if aspect is not None:
            ax.set_aspect(aspect)

        keypoint_painter.annotation(ax, ann)


def draw_skeletons(pose, prefix=""):
    from openpifpaf.annotation import Annotation  # pylint: disable=import-outside-toplevel
    from openpifpaf import show  # pylint: disable=import-outside-toplevel

    scale = np.sqrt(
        (np.max(pose[:, 0]) - np.min(pose[:, 0]))
        * (np.max(pose[:, 1]) - np.min(pose[:, 1]))
    )

    show.KeypointPainter.show_joint_scales = True
    keypoint_painter = show.KeypointPainter(line_width=2)

    ann = Annotation(keypoints=WHOLEBODY_KEYPOINTS,
                     skeleton=WHOLEBODY_SKELETON,
                     score_weights=WHOLEBODY_SCORE_WEIGHTS)
    ann.set(pose, np.array(WHOLEBODY_SIGMAS) * scale)
    draw_ann(ann, filename='./docs/'+prefix+'skeleton_wholebody.png', keypoint_painter=keypoint_painter)



def print_associations():
    for j1, j2 in WHOLEBODY_SKELETON:
        print(WHOLEBODY_KEYPOINTS[j1 - 1], '-', WHOLEBODY_KEYPOINTS[j2 - 1])

def rotate(pose, angle=45, axis=2):
    sin = np.sin(np.radians(angle))
    cos = np.cos(np.radians(angle))
    pose_copy = np.copy(pose)
    pose_copy[:, 2] = pose_copy[:, 2]-2 # COOS at human center
    if axis==0:
        rot_mat = np.array([[1, 0, 0],
                            [0, cos, -sin],
                            [0, sin, cos]])
    elif axis==1:
        rot_mat = np.array([[cos, 0, sin],
                            [0, 1, 0],
                            [-sin, 0, cos]])
    elif axis==2:
        rot_mat = np.array([[cos, -sin, 0],
                            [sin, cos, 0],
                            [0, 0, 1]])
    else:
        raise Exception("Axis must be 0,1 or 2 (corresponding to x,y,z).")
    rotated_pose = np.transpose(np.matmul(rot_mat, np.transpose(pose_copy)))
    rotated_pose[:,2] = rotated_pose[:,2]+7 #assure sufficient depth for plotting
    return rotated_pose

if __name__ == '__main__':
    print_associations()
# =============================================================================
#     #Use this to create a gif
#     for deg in range(360):
#         rot = rotate(WHOLEBODY_STANDING_POSE, angle=deg, axis=1)
#         draw_skeletons(rot, prefix="rotate_"+str(deg)+"deg_")
# =============================================================================
    draw_skeletons(WHOLEBODY_STANDING_POSE)
    rot45 = rotate(WHOLEBODY_STANDING_POSE, angle=45, axis=1) 
    draw_skeletons(rot45, prefix="rot45_")
