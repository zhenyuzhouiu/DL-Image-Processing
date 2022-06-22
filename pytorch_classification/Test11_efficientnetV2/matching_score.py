import os

import argparse
import torch
from PIL import Image
import numpy as np
from torch.autograd import Variable
from model import efficientnetv2_s as create_model
import scipy.io as io
import math


def load_data(data_path, subject_no, set_no, default_size):
    data = torch.zeros([subject_no * set_no, 3, default_size, default_size])
    for subjectID in range(subject_no):
        for setID in range(set_no):
            path = os.path.join(data_path, "subject"+str(subjectID+1), 'stack'+str(setID)+'.bmp')
            im = np.array(
                Image.open(path).convert("RGB").resize((default_size, default_size)),
                dtype=np.float32)
            # change h,w,c = c,h,w
            im = np.transpose(im, (2, 0, 1))
            im /= 255.
            im = torch.from_numpy(im.astype(np.float32))
            im = im.cuda()
            im = Variable(im, requires_grad=False)
            data[subjectID*set_no+setID, :, :, :] = im.unsqueeze(0)

    return data


def class_matrix(data, subject_no, set_no, model, device, save_path):
    score_matrix = np.zeros([subject_no, subject_no*set_no])
    for i in range(subject_no*set_no):
        print(i)
        I = data[i, :, :, :]
        I = I.unsqueeze(0)
        pred = model(I.to(device))
        pred_class = torch.nn.functional.softmax(pred)
        score_matrix[:, i] = pred_class.cpu().numpy()

    score_matrix_path = os.path.join(save_path, "score_matrix.mat")
    io.savemat(score_matrix_path, {'score_matrix':score_matrix})
    score_matrix = -score_matrix

    D_genuine = np.zeros([1, subject_no*set_no])
    D_imposter = np.zeros([1, subject_no*(subject_no-1)*set_no])
    counter_genuine = 0;
    counter_imposter = 0;
    for i in range(score_matrix.shape[0]):
        for j in range(score_matrix.shape[1]):
            subjectID = i
            subjectID_2 = math.floor(j/set_no)
            if(subjectID == subjectID_2):
                D_genuine[:, counter_genuine] = score_matrix[i, j]
                counter_genuine = counter_genuine + 1
            else:
                D_imposter[:, counter_imposter] = score_matrix[i, j]
                counter_imposter = counter_imposter + 1
    D_genuine_path = os.path.join(save_path, "D_genuine.mat")
    io.savemat(D_genuine_path, {'D_genuine':D_genuine})
    D_imposter_path = os.path.join(save_path, "D_imposter.mat")
    io.savemat(D_imposter_path, {'D_imposter': D_imposter})


def feature_vector_single_matrix(data, subject_no, set_no, model, device, save_path, num_classes):
    feature_matrix = np.zeros([subject_no*set_no, num_classes])
    for i in range(subject_no*set_no):
        print(i)
        I = data[i, :, :, :]
        I = I.unsqueeze(0)
        pred = model(I.to(device))
        feature_matrix[i, :] = pred.cpu().numpy()
    feature_matrix_path = os.path.join(save_path, "feature_matrix.mat")
    io.savemat(feature_matrix_path, {'feature_matrix': feature_matrix})

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


def feature_vector_two_matrix(data, session2, subject_no, set_no, model, device, save_path, num_classes):
    feature_matrix = np.zeros([subject_no * set_no, num_classes])
    feature_matrix_session2 = np.zeros([subject_no * set_no, num_classes])
    for i in range(subject_no * set_no):
        print(i)
        I = data[i, :, :, :]
        I = I.unsqueeze(0)
        pred = model(I.to(device))
        feature_matrix[i, :] = pred.cpu().numpy()
    feature_matrix_path = os.path.join(save_path, "feature_matrix.mat")
    io.savemat(feature_matrix_path, {'feature_matrix': feature_matrix})

    for i in range(subject_no * set_no):
        print(i)
        I = session2[i, :, :, :]
        I = I.unsqueeze(0)
        pred = model(I.to(device))
        feature_matrix_session2[i, :] = pred.cpu().numpy()
    feature_matrix_path = os.path.join(save_path, "feature_matrix_session2.mat")
    io.savemat(feature_matrix_path, {'feature_matrix_session2': feature_matrix_session2})
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
    parser.add_argument("--model_path", type=str, dest="model_path", default="/home/zhenyuzhou/Desktop/CV/deep-learning-for-image-processing/pytorch_classification/Test11_efficientnetV2/3d1s-weights/model-159.pth")
    parser.add_argument("--num_classes", type=int, dest="num_classes", default=190)
    parser.add_argument("--data_path", type=str, dest="data_path", default="/home/zhenyuzhou/Desktop/finger-knuckle/deep-learning/Finger-Knuckle-Recognition/dataset/3DFingerKnuckle/finger/caffe/forefinger/train/session2/80_48/")
    parser.add_argument("--session2", type=str, dest="session2", default="/home/zhenyuzhou/Desktop/finger-knuckle/deep-learning/Finger-Knuckle-Recognition/dataset/PolyUHD/yolov5/caffe/test-set-80-48/")
    parser.add_argument("--save_path", type=str, dest="save_path", default="/home/zhenyuzhou/Desktop/CV/deep-learning-for-image-processing/pytorch_classification/Test11_efficientnetV2/3d1s-weights/feature_vector/")
    parser.add_argument("--subject_no", type=int, dest="subject_no", default=190)
    parser.add_argument("--set_no", type=int, dest="set_no", default=6)
    parser.add_argument("--default_size", type=int, dest="default_size", default=300)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    args = parser.parse_args()

    data = load_data(args.data_path, args.subject_no, args.set_no, args.default_size)

    # session2 = load_data(args.session2, args.subject_no, args.set_no, args.default_size)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        # 如果存在预训练权重则载入
        model = create_model(num_classes=args.num_classes).to(device)
        if args.model_path != "":
            if os.path.exists(args.model_path):
                weights_dict = torch.load(args.model_path, map_location=device)
                load_weights_dict = {k: v for k, v in weights_dict.items()
                                     if model.state_dict()[k].numel() == v.numel()}
                print(model.load_state_dict(load_weights_dict, strict=False))
            else:
                raise FileNotFoundError("not found weights file: {}".format(args.weights))
        model.eval()
        feature_vector_single_matrix(data, args.subject_no, args.set_no, model, device, args.save_path, args.num_classes)
        # feature_vector_two_matrix(data, session2, args.subject_no, args.set_no, model, device, args.save_path, args.num_classes)
        # class_matrix(data, args.subject_no, args.set_no, model, device, args.save_path)




