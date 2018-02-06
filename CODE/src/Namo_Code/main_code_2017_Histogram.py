from configobj import ConfigObj
import glob
import os
import numpy as np
from math import pow, sqrt, sin, cos, radians
from scipy.stats import norm
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, explained_variance_score

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, mean_absolute_error
from sklearn import cross_validation, preprocessing
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import classification_report

from sklearn.neural_network import MLPRegressor

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt



import time
import pickle

from sys import getsizeof

from utility_function_2017 import *


def create_3dim_normalize_score_array(radius):

    normal_dis_array_size = radius + radius + 1
    normal_dis_sigma_width = 3

    array = np.zeros((normal_dis_array_size,normal_dis_array_size,normal_dis_array_size))
    for z in range(0,normal_dis_array_size):
        for y in range(0,normal_dis_array_size):
            for x in range(0, normal_dis_array_size):
                distance = sqrt(pow(x-radius,2) + pow(y-radius,2) + pow(z-radius,2))
                distance = distance*normal_dis_sigma_width/radius
                array[x, y, z] = round(norm.pdf(distance), 3)

    #print(array)
    return array

def create_4dim_normalize_score_array(radius):

    normal_dis_array_size = radius + radius + 1
    normal_dis_sigma_width = 3

    array = np.zeros((normal_dis_array_size,normal_dis_array_size,normal_dis_array_size,normal_dis_array_size))
    for z in range(0,normal_dis_array_size):
        for y in range(0,normal_dis_array_size):
            for x in range(0, normal_dis_array_size):
                for w in range(0, normal_dis_array_size):
                    distance = sqrt(pow(w-radius,2) + pow(x-radius,2) + pow(y-radius,2) + pow(z-radius,2))
                    distance = distance*normal_dis_sigma_width/radius
                    array[w, x, y, z] = round(norm.pdf(distance), 3)

    #print(array)
    return array

def collect_data(postureName):

    posture_dataset = []
    path = './Postures/posture_set_' + str(postureName)
    for i, filename in enumerate(glob.glob(os.path.join(path, '*.ini'))):
        config = ConfigObj(filename)
        main_index = [index for index, x in enumerate(config['Keyframe_Posture_Type']) if x == 'main']
        for index in main_index:
            posture_dataset.append(list(map(int, config['Keyframe_Value']['Keyframe_' + str(index)])))

    data_set = convert_motorValue_to_cartesianSpace(posture_dataset)
    return data_set

def convert_motorValue_to_cartesianSpace(posture_dataSet):
    int_motorDirection_and_ratio = [0.5, -1, 1, 1, 1, -1, 1,-0.5, -1, 1, -1, 1, -1, -1, 1, -1, 1]

    ### read motor center value ###
    file_center = open("./Postures/motor_center.txt", 'r')
    int_motorCenterValue = file_center.read()
    file_center.close()
    int_motorCenterValue = int_motorCenterValue.split('\n')
    for x in range(17):
        int_motorCenterValue[x] = int(int_motorCenterValue[x])

    ### cal diff ###
    diff_set = []
    for i, item in enumerate(posture_dataSet):
        diff = []
        for j,value in enumerate(item):
            diff.append(round((value - int_motorCenterValue[j])*int_motorDirection_and_ratio[j]*359/4095,0))

        diff_set.append(diff)

    #print("diff_set=",diff_set)

    ### convert to degree ###
    motorDegree_set = []
    for i, item in enumerate(diff_set):
        motorDegree = []
        for j, value in enumerate(item):
            #motorDegree.append(round(value * 359 / 4095.0,0))
            motorDegree.append(value)

        motorDegree_set.append(motorDegree)

    #print("motor_degree=", motorDegree_set)

    return motorDegree_set




##################################################################################################
if __name__ == "__main__":

    #base_score_3dim = create_3dim_normalize_score_array(20) ### @param(sphere_radius)
    #print(base_score_3dim.shape)

    #base_score_4dim = create_4dim_normalize_score_array(5)
    #print(base_score_4dim.shape)
    #print(base_score_4dim[3][3][3][3])

     ### load data ###
    jointAngle_degree_bye_set = collect_data('bye') ### @param(postureName)
    jointAngle_degree_salute_set = collect_data('salute')  ### @param(postureName)
    jointAngle_degree_sinvite_set = collect_data('side_invite')  ### @param(postureName)
    jointAngle_degree_wai_set = collect_data('wai')  ### @param(postureName)

    ### extract right arm data ###
    right_side_bye_set = extract_arm_data( jointAngle_degree_bye_set,'right')
    right_side_salute_set = extract_arm_data(jointAngle_degree_salute_set, 'right')
    right_side_sinvite_set = extract_arm_data(jointAngle_degree_sinvite_set, 'right')
    right_side_wai_set = extract_arm_data(jointAngle_degree_wai_set, 'right')

    ### calculate each posture stat ###
    right_side_bye_stat = calculate_stat_all_joint(right_side_bye_set)
    right_side_salute_stat = calculate_stat_all_joint(right_side_salute_set)
    right_side_sinvite_stat = calculate_stat_all_joint(right_side_sinvite_set)
    right_side_wai_stat = calculate_stat_all_joint(right_side_wai_set)

    ### calculate average join angle from all posture ###
    all_posture_stat_list = [right_side_bye_stat, right_side_salute_stat, right_side_sinvite_stat, right_side_wai_stat]
    ### type :: 'std' = standard_deviation, 'equl' = all weight equal
    avg_joint_angle_std = find_avg_joint_angle(all_posture_stat_list, 'std')
    avg_joint_angle_equl = find_avg_joint_angle(all_posture_stat_list, 'equl')

    ### calculate kinematics ###
    bye_kinematics_set = collect_kinematics_data(right_side_bye_set)
    salute_kinematics_set = collect_kinematics_data(right_side_salute_set)
    sinvite_kinematics_set = collect_kinematics_data(right_side_sinvite_set)
    wai_kinematics_set = collect_kinematics_data(right_side_wai_set)

    print(bye_kinematics_set[0][0])
    #print(avg_joint_angle_equl)
