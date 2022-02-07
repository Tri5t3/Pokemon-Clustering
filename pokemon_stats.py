import csv
import math
import numpy as np
import random
from numpy.lib.function_base import append
from scipy.cluster.hierarchy import dendrogram, linkage
from itertools import islice
import matplotlib.pyplot as plt


def load_data(filepath):
    columns = ['#', 'Name', 'Type 1', 'Type 2', 'Total', 'HP',
               'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
    dataset = []
    ret = []
    with open(filepath, newline='') as csvfile:
        reader = csv.DictReader(csvfile, columns)
        for elem in islice(reader, 21):
            temp = {}
            for column in columns:
                temp[column] = elem[column]
            dataset.append(temp)
    dataset.pop(0)
    return dataset


def calculate_x_y(stats):
    x = int(stats['Attack']) + int(stats['Sp. Atk']) + int(stats['Speed'])
    y = int(stats['Defense']) + int(stats['Sp. Def']) + int(stats['HP'])
    return (x, y)


def hac(dataset):
    existedTuple = []
    existCluster = []
    clu_index = []
    ret = []
    leng = len(dataset)
    for time in range(leng - 1):
        # searchForSmallest(dataset, leng, existCluster)
        min_dis = 1000
        min_ind_1 = -1
        min_ind_2 = -1
        min_len = -1
        for i in range(leng - 1):
            for j in range(i+1, leng):
                # print(dataset)
                test = calc_dist(dataset[i], dataset[j])
                if test < min_dis:
                    if (i, j) in existedTuple:
                        continue
                    tmpIndic = 0
                    for tmp in range(len(existCluster)):
                        if i in existCluster[tmp] and j in existCluster[tmp]:
                            tmpIndic = 1
                    if tmpIndic == 1:
                        continue
                    min_dis = calc_dist(dataset[i], dataset[j])
                    min_ind_1 = i
                    min_ind_2 = j
        # got nums and dist

        existedTuple.append((min_ind_1, min_ind_2))
        clu_index1 = -1
        clu_index2 = -1
        for i in range(len(existCluster)):
            if min_ind_1 in existCluster[i]:
                clu_index1 = i
            if min_ind_2 in existCluster[i]:
                clu_index2 = i
        # test:
        # print("Index 1: ", min_ind_1, "Index 2: ", min_ind_2)
        # print("Index 1 in: ", clu_index1, "Index 2 in: ", clu_index2)

        # both single
        if clu_index1 == -1 and clu_index2 == -1:
            temp = []
            temp.append(int(min_ind_1))
            temp.append(int(min_ind_2))
            existCluster.append(temp)
            clu_index.append(time)
            min_len = 2

            # add row
            loopRow = []
            loopRow.append(min_ind_1)
            loopRow.append(min_ind_2)
            loopRow.append(min_dis)
            loopRow.append(min_len)
            # print("Loop ", time, " :", loopRow)
            ret.append(loopRow)

        # 1 single 2 in cluster
        if clu_index1 == -1 and clu_index2 != -1:
            existCluster[clu_index2].append(min_ind_1)
            min_ind_2 = clu_index[clu_index2] + leng
            clu_index[clu_index2] = time
            min_len = len(existCluster[clu_index2])

            # add row
            loopRow = []
            loopRow.append(min_ind_1)
            loopRow.append(min_ind_2)
            loopRow.append(min_dis)
            loopRow.append(min_len)
            # print("Loop ", time, " :", loopRow)
            ret.append(loopRow)

        # 1 in cluster 2 single
        if clu_index1 != -1 and clu_index2 == -1:
            existCluster[clu_index1].append(min_ind_2)
            min_ind_1 = clu_index[clu_index1] + leng
            clu_index[clu_index1] = time
            min_len = len(existCluster[clu_index1])

            # add row
            loopRow = []
            loopRow.append(min_ind_1)
            loopRow.append(min_ind_2)
            loopRow.append(min_dis)
            loopRow.append(min_len)
            # print("Loop ", time, " :", loopRow)
            ret.append(loopRow)

        # both in cluster
        if clu_index1 != -1 and clu_index2 != -1:
            min_ind_1 = clu_index[clu_index1] + leng
            min_ind_2 = clu_index[clu_index2] + leng
            for elem in existCluster[clu_index2]:
                existCluster[clu_index1].append(elem)
            clu_index[clu_index1] = time
            min_len = len(existCluster[clu_index1])

            # pop 2
            existCluster.pop(clu_index2)
            clu_index.pop(clu_index2)

            # add row
            loopRow = []
            loopRow.append(min_ind_1)
            loopRow.append(min_ind_2)
            loopRow.append(min_dis)
            loopRow.append(min_len)
            # print("Loop ", time, " :", loopRow)
            ret.append(loopRow)
    for i in range(len(ret)):
        if ret[i][0] > ret[i][1]:
            swap = ret[i][0]
            ret[i][0] = ret[i][1]
            ret[i][1] = swap
    # for i in range(len(ret)):
    #     ret[i][0] = int(ret[i][0])
    #     ret[i][1] = int(ret[i][1])
    #     ret[i][3] = int(ret[i][3])
    npArr = np.array(ret)
    for i in range(len(npArr)):
        npArr[i][0] = math.trunc(npArr[i][0])
        npArr[i][1] = math.trunc(npArr[i][1])
        npArr[i][3] = math.trunc(npArr[i][3])
    return npArr


def random_x_y(m):
    dataset = []
    for i in range(m):
        x = random.randint(1, 359)
        y = random.randint(1, 359)
        dataset.append((x, y))
    return dataset


def imshow_hac(dataset):
    existedTuple = []
    existCluster = []
    clu_index = []
    ret = []
    leng = len(dataset)
    # plot
    x, y = zip(*dataset)
    fig = plt.figure()
    plt.scatter(x, y)
    plt.pause(0.4)
    for time in range(leng - 1):
        # searchForSmallest(dataset, leng, existCluster)
        min_dis = 1000
        min_ind_1 = -1
        min_ind_2 = -1
        min_len = -1
        for i in range(leng - 1):
            for j in range(i+1, leng):
                # print(dataset)
                test = calc_dist(dataset[i], dataset[j])
                if test < min_dis:
                    if (i, j) in existedTuple:
                        continue
                    tmpIndic = 0
                    for tmp in range(len(existCluster)):
                        if i in existCluster[tmp] and j in existCluster[tmp]:
                            tmpIndic = 1
                    if tmpIndic == 1:
                        continue
                    min_dis = calc_dist(dataset[i], dataset[j])
                    min_ind_1 = i
                    min_ind_2 = j
        # got nums and dist

        draw_Index_1 = min_ind_1
        draw_Index_2 = min_ind_2

        existedTuple.append((min_ind_1, min_ind_2))
        clu_index1 = -1
        clu_index2 = -1
        for i in range(len(existCluster)):
            if min_ind_1 in existCluster[i]:
                clu_index1 = i
            if min_ind_2 in existCluster[i]:
                clu_index2 = i
        # test:
        # print("Index 1: ", min_ind_1, "Index 2: ", min_ind_2)
        # print("Index 1 in: ", clu_index1, "Index 2 in: ", clu_index2)

        # both single
        if clu_index1 == -1 and clu_index2 == -1:
            temp = []
            temp.append(int(min_ind_1))
            temp.append(int(min_ind_2))
            existCluster.append(temp)
            clu_index.append(time)
            min_len = 2

            # add row
            loopRow = []
            loopRow.append(min_ind_1)
            loopRow.append(min_ind_2)
            loopRow.append(min_dis)
            loopRow.append(min_len)
            # print("Loop ", time, " :", loopRow)
            ret.append(loopRow)
            point1 = np.array(
                [dataset[draw_Index_1][0], dataset[draw_Index_2][0]])
            point2 = np.array(
                [dataset[draw_Index_1][1], dataset[draw_Index_2][1]])
            plt.plot(point1, point2)
            plt.draw()
            plt.pause(0.4)

        # 1 single 2 in cluster
        if clu_index1 == -1 and clu_index2 != -1:
            existCluster[clu_index2].append(min_ind_1)
            min_ind_2 = clu_index[clu_index2] + leng
            clu_index[clu_index2] = time
            min_len = len(existCluster[clu_index2])

            # add row
            loopRow = []
            loopRow.append(min_ind_1)
            loopRow.append(min_ind_2)
            loopRow.append(min_dis)
            loopRow.append(min_len)
            # print("Loop ", time, " :", loopRow)
            ret.append(loopRow)
            point1 = np.array(
                [dataset[draw_Index_1][0], dataset[draw_Index_2][0]])
            point2 = np.array(
                [dataset[draw_Index_1][1], dataset[draw_Index_2][1]])
            plt.plot(point1, point2)
            plt.draw()
            plt.pause(0.4)

        # 1 in cluster 2 single
        if clu_index1 != -1 and clu_index2 == -1:
            existCluster[clu_index1].append(min_ind_2)
            min_ind_1 = clu_index[clu_index1] + leng
            clu_index[clu_index1] = time
            min_len = len(existCluster[clu_index1])

            # add row
            loopRow = []
            loopRow.append(min_ind_1)
            loopRow.append(min_ind_2)
            loopRow.append(min_dis)
            loopRow.append(min_len)
            # print("Loop ", time, " :", loopRow)
            ret.append(loopRow)
            point1 = np.array(
                [dataset[draw_Index_1][0], dataset[draw_Index_2][0]])
            point2 = np.array(
                [dataset[draw_Index_1][1], dataset[draw_Index_2][1]])
            plt.plot(point1, point2)
            plt.draw()
            plt.pause(0.4)

        # both in cluster
        if clu_index1 != -1 and clu_index2 != -1:
            min_ind_1 = clu_index[clu_index1] + leng
            min_ind_2 = clu_index[clu_index2] + leng
            for elem in existCluster[clu_index2]:
                existCluster[clu_index1].append(elem)
            clu_index[clu_index1] = time
            min_len = len(existCluster[clu_index1])

            # pop 2
            existCluster.pop(clu_index2)
            clu_index.pop(clu_index2)

            # add row
            loopRow = []
            loopRow.append(min_ind_1)
            loopRow.append(min_ind_2)
            loopRow.append(min_dis)
            loopRow.append(min_len)
            # print("Loop ", time, " :", loopRow)
            ret.append(loopRow)
            point1 = np.array(
                [dataset[draw_Index_1][0], dataset[draw_Index_2][0]])
            point2 = np.array(
                [dataset[draw_Index_1][1], dataset[draw_Index_2][1]])
            plt.plot(point1, point2)
            plt.draw()
            plt.pause(0.4)

    for i in range(len(ret)):
        if ret[i][0] > ret[i][1]:
            swap = ret[i][0]
            ret[i][0] = ret[i][1]
            ret[i][1] = swap
    npArr = np.array(ret)

    plt.show()
    return npArr


def calc_dist(point1, point2):
    return math.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1]) ** 2)
