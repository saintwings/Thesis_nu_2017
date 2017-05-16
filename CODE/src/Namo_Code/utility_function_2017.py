from math import pow, sqrt, sin, cos, radians
import numpy as np


def collect_kinematics_data(joint7dof_dataSet):
    kinematics_dataSet = []

    for i, jointSet in enumerate(joint7dof_dataSet):
        kinematics_dataSet.append(cal_kinematics_namo_numpy(jointSet,'right'))

    return kinematics_dataSet

def cal_kinematics_namo_numpy(degree7Joint,armSide):
    """ calculate Namo robot kinematic 7DOF Arm
    :param degree7Joint: input [degree0,1,2,3,4,5,6]
    :param armSide: input arm side 'L' for Left side, 'R' for Right side
    :return: Transformation Matrix List [T01,T02,T03,T04,T05,T06,T07,T0E]
    """
    ## DH parameter >> alpha[0,1,...6,end effector]
    alpha = [radians(90),radians(90),radians(-90),radians(-90),radians(90),radians(90),
             radians(-90),radians(0)]
    ## DH parameter >> a[0,1,...6,end effector]
    a = [0,0,0,0,0,0,0,-140]
    ## DH parameter >> d[1,2,...7,end effector]
    d = [182,0,206.5,0,206,0,0,0]
    if armSide == 'left':
        d[0] = d[0]*(-1)
    elif armSide == 'right':
        d[0] = d[0]*1
    #print("7dof ="+str(degree7Joint))
    ## DH parameter >> theta[1,2,...7,end effector]
    theta = [radians(degree7Joint[0] + 90),radians(degree7Joint[1] + 90),radians(degree7Joint[2] - 90),
             radians(degree7Joint[3]),radians(degree7Joint[4] + 90),radians(degree7Joint[5] - 90),
             radians(degree7Joint[6]),radians(0)]
    T = {}
    for i in range(0,8):
        #print i

        T[i] = np.array([[(cos(theta[i])), (-sin(theta[i])), 0, a[i]],
                       [(sin(theta[i]) * (cos(alpha[i]))), (cos(theta[i])) * (cos(alpha[i])), (-sin(alpha[i])),
                        (-sin(alpha[i])) * d[i]],
                       [(sin(theta[i]) * (sin(alpha[i]))), (cos(theta[i])) * (sin(alpha[i])), (cos(alpha[i])),
                        (cos(alpha[i])) * d[i]],
                       [0, 0, 0, 1]])


    T01 = T[0]
    T02 = np.dot(T01,T[1])
    T03 = np.dot(T02,T[2])
    T04 = np.dot(T03,T[3])
    T05 = np.dot(T04,T[4])
    T06 = np.dot(T05,T[5])
    T07 = np.dot(T06,T[6])
    T0E = np.dot(T07,T[7])
    return [T01,T02,T03,T04,T05,T06,T07,T0E]

def collect_quaternion_data(kinematics_dataSet):
    quaternion_dataSet = []

    for i, kinetics in enumerate(kinematics_dataSet):
        quaternion_dataSet.append(cal_quaternion(kinetics[7]))

    return quaternion_dataSet

def cal_quaternion(Tmatrix):
    #print Tmatrix
    tr = Tmatrix[0,0] + Tmatrix[1,1] + Tmatrix[2,2]
    if(tr>0):
        S = sqrt(tr+1.0)*2 ## S=4*qw
        qw = 0.25*S
        qx = (Tmatrix[2,1]-Tmatrix[1,2])/S
        qy = (Tmatrix[0,2]-Tmatrix[2,0])/S
        qz = (Tmatrix[1,0]-Tmatrix[0,1])/S
        #print "case1"
    elif((Tmatrix[0,0]>Tmatrix[1,1]) and (Tmatrix[0,0]>Tmatrix[2,2])):
        S = sqrt(1.0 + Tmatrix[0,0] - Tmatrix[1,1] - Tmatrix[2,2])*2 ## S=4*qx
        qw = (Tmatrix[2,1]-Tmatrix[1,2])/S
        qx = 0.25*S
        qy = (Tmatrix[0,1]+Tmatrix[1,0])/S
        qz = (Tmatrix[0,2]+Tmatrix[2,0])/S
        #print "case2"
    elif(Tmatrix[1,1]>Tmatrix[2,2]):
        S = sqrt(1.0 + Tmatrix[1,1] - Tmatrix[0,0] - Tmatrix[2,2])*2 ## S=4*qy
        qw = (Tmatrix[0,2]-Tmatrix[2,0])/S
        qx = (Tmatrix[0,1]+Tmatrix[1,0])/S
        qy = 0.25*S
        qz = (Tmatrix[1,2]+Tmatrix[2,1])/S
        #print "case3"
    else:
        S = sqrt(1.0 + Tmatrix[2,2] - Tmatrix[0,0] - Tmatrix[1,1])*2 ## S=4*qz
        qw = (Tmatrix[1,0]-Tmatrix[0,1])/S
        qx = (Tmatrix[0,2]+Tmatrix[2,0])/S
        qy = (Tmatrix[1,2]+Tmatrix[2,1])/S
        qz = 0.25*S
        #print "case4"

    qw = round(qw,2)
    qx = round(qx, 2)
    qy = round(qy, 2)
    qz = round(qz, 2)

    norm = round(sqrt((qw*qw) + (qx*qx) + (qy*qy) + (qz*qz)),1)
    return [qw,qx,qy,qz,norm]

def cal_Single_Posture_Score(T_matrix_ref, Q_ref, T_matrix, Q, score_weight):
    score = []

    Q_diff = [Q_ref[0] - Q[0], Q_ref[1] - Q[1], Q_ref[2] - Q[2], Q_ref[3] - Q[3]]

    norm_Q_diff = sqrt(
        (Q_diff[0] * Q_diff[0]) + (Q_diff[1] * Q_diff[1]) + (Q_diff[2] * Q_diff[2]) + (Q_diff[3] * Q_diff[3]))
    # elbow_diff =[TMatrix_Target_Posture[3][0,3] - T_Matrix[3][0,3], TMatrix_Target_Posture[3][1,3] - T_Matrix[3][1,3], TMatrix_Target_Posture[3][2,3] - T_Matrix[3][2,3]]
    norm_elbow_diff = sqrt(pow(T_matrix_ref[3][0, 3] - T_matrix[3][0, 3], 2) + pow(
        T_matrix_ref[3][1, 3] - T_matrix[3][1, 3], 2) + pow(
        T_matrix_ref[3][2, 3] - T_matrix[3][2, 3], 2))
    # wrist_diff =[TMatrix_Target_Posture[4][0,3] - T_Matrix[4][0,3], TMatrix_Target_Posture[4][1,3] - T_Matrix[4][1,3], TMatrix_Target_Posture[4][2,3] - T_Matrix[4][2,3]]
    norm_wrist_diff = sqrt(pow(T_matrix_ref[4][0, 3] - T_matrix[4][0, 3], 2) + pow(
        T_matrix_ref[4][1, 3] - T_matrix[4][1, 3], 2) + pow(
        T_matrix_ref[4][2, 3] - T_matrix[4][2, 3], 2))
    sum_diff = norm_Q_diff * score_weight[0] + norm_elbow_diff * score_weight[1] + norm_wrist_diff * score_weight[2]

    #score.append(Q_diff)
    score.append(norm_Q_diff)
    score.append(norm_elbow_diff)
    score.append(norm_wrist_diff)
    score.append(sum_diff)

    return score