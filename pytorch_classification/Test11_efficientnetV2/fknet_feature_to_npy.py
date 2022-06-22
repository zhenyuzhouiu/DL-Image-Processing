import os

import argparse
import torch
from PIL import Image
import numpy as np
from torch.autograd import Variable
from model import efficientnetv2_s as create_model
import scipy.io as io
import math


def feature_vector_single_matrix(data, subject_no, set_no, save_path, num_classes):

    feature_matrix = io.loadmat(data)['feature_matrix_test']

    matching_matrix = np.ones((subject_no * set_no, subject_no * set_no)) * 1000000
    for i in range(1, feature_matrix.shape[0]):
        feat1 = feature_matrix[:-i, :]
        feat2 = feature_matrix[i:, :]
        distance = np.sum((feat1 - feat2)**2, axis=1) / feat1.shape[1]
        matching_matrix[:-i, i] = distance
        print("[*] Pre-processing matching dict for {} / {} \r".format(i, feature_matrix.shape[0]))

    matt = np.ones_like(matching_matrix) * 1000000
    matt[0, :] = matching_matrix[0, :]
    for i in range(1, feature_matrix.shape[0]):
        # matching_matrix每行的数值向后移动一位
        matt[i, i:] = matching_matrix[i, :-i]
        for j in range(i):
            # matt[i, j] = matt[j, i]
            matt[i, j] = matching_matrix[j, i - j]

    print("\n [*] Done")

    g_scores = []
    i_scores = []
    for i in range(subject_no * set_no):
        start_idx = int(math.floor(i / set_no))
        start_remainder = int(i % set_no)

        argmin_idx = np.argmin(matt[i, start_idx * set_no: start_idx * set_no + set_no])
        g_scores.append(float(matt[i, start_idx * set_no + argmin_idx]))
        select = list(range(subject_no * set_no))
        # remove genuine matching score
        for j in range(set_no):
            select.remove(start_idx * set_no + j)
        # remove imposter matching scores of same index sample on other subjects
        for j in range(subject_no):
            if j == start_idx:
                continue
            select.remove(j * set_no + start_remainder)
        i_scores += list(np.min(np.reshape(matt[i, select], (-1, set_no - 1)), axis=1))
        print("[*] Processing genuine imposter for {} / {} \r".format(i, subject_no * set_no))
    print("\n [*] Done")
    protocol_path = os.path.join(save_path, 'protocol.npy')
    np.save(protocol_path, {"g_scores": np.array(g_scores), "i_scores": np.array(i_scores), "mmat": matt})


def feature_vector_two_matrix(data, session2, subject_no, set_no, save_path, num_classes):

    feature_matrix = io.loadmat(data)['s1_feature_matrix_test']
    feature_matrix_session2 = io.loadmat(session2)['s2_feature_matrix_test']
    feats_gallery = np.concatenate((feature_matrix_session2, feature_matrix_session2), 0)

    nl = subject_no * set_no
    matching_matrix = np.ones((nl, nl)) * 1000000
    for i in range(nl):
        distance = np.sum((feature_matrix - feats_gallery[i:i+nl, :]) ** 2, axis=1) / feature_matrix.shape[1]
        matching_matrix[:, i] = distance
        print("[*] Pre-processing matching dict for {} / {} \r".format(i, nl))

    for i in range(1, nl):
        tmp = matching_matrix[i, -i:].copy()
        matching_matrix[i, i:] = matching_matrix[i, :-i]
        matching_matrix[i, :i] = tmp
    print("\n [*] Done")

    g_scores = []
    i_scores = []
    for i in range(nl):
        start_idx = int(math.floor(i / set_no))
        start_remainder = int(i % set_no)
        g_scores.append(float(np.min(matching_matrix[i, start_idx * set_no: start_idx * set_no + set_no])))
        select = list(range(nl))
        for j in range(set_no):
            select.remove(start_idx * set_no + j)
        i_scores += list(np.min(np.reshape(matching_matrix[i, select], (-1, set_no)), axis=1))
        print("[*] Processing genuine imposter for {} / {} \r".format(i, subject_no * set_no))

    print("\n [*] Done")
    protocol_path = os.path.join(save_path, 'protocol.npy')
    np.save(protocol_path, {"g_scores": np.array(g_scores), "i_scores": np.array(i_scores), "mmat": matching_matrix})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_classes", type=int, dest="num_classes", default=190)
    parser.add_argument("--data_path", type=str, dest="data_path", default="/home/zhenyuzhou/Desktop/finger-knuckle/deep-learning/codekevin/fknet-3d1s/feature_matrix_test.mat")
    parser.add_argument("--session2", type=str, dest="session2", default="/home/zhenyuzhou/Desktop/finger-knuckle/deep-learning/codekevin/fknet-yolov5_fkv3_session1_105_221-two-session/s2_feature_matrix_test.mat")
    parser.add_argument("--save_path", type=str, dest="save_path", default="/home/zhenyuzhou/Desktop/finger-knuckle/deep-learning/codekevin/fknet-3d1s/")
    parser.add_argument("--subject_no", type=int, dest="subject_no", default=190)
    parser.add_argument("--set_no", type=int, dest="set_no", default=6)
    args = parser.parse_args()

    feature_vector_single_matrix(args.data_path, args.subject_no, args.set_no, args.save_path, args.num_classes)
    # feature_vector_two_matrix(args.data_path, args.session2, args.subject_no, args.set_no, args.save_path, args.num_classes)




