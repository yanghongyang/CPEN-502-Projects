import os
import re
import matplotlib.pyplot as plt

if __name__ == '__main__':
    path = '/Users/hannah/IdeaProjects/CPEN-502-Projects/Assignment-1/Error'   # training result  documents path
    files = os.listdir(path)  # acquire all document name under a folder
    number_binary = []
    number_bipolar = []
    number_momentum_binary = []
    number_momentum_bipolar = []
    X_binary, Y_binary = [], []
    #  X_binary_min stands for the epoch number of the trail
    #  with minimum epoch number of training using binary dataset
    #  Y_binary_min stands for the total error of the trail
    #  with minimun epoch number of training using binary dataset
    # X_binary_max, Y_binary_max = [], []
    # #  X_binary_max stands for the epoch number of the trail
    # with maximum epoch number of training using binary dataset
    # #  Y_binary_max stands for the total error of the trail
    # with maximun epoch number of training using binary dataset X_binary_momentum_min, Y_binary_momentum_min = [], []
    X_binary_momentum, Y_binary_momentum = [], []
    # X_binary_momentum_max, Y_binary_momentum_max = [], []
    X_bipolar, Y_bipolar = [], []
    # X_bipolar_max, Y_bipolar_max = [], []
    X_bipolar_momentum, Y_bipolar_momentum = [], []
    # X_bipolar_momentum_max, Y_bipolar_momentum_max = [], []
    for file in files:  # 遍历文件夹
        if re.search(r'TrainTotalError', file, re.M):
            if file.split('-', 2)[1] == 'true':
                number_binary.append(int(file.split('-', 2)[-1].split('.', 1)[0]))
            else:
                number_bipolar.append(int(file.split('-', 2)[-1].split('.', 1)[0]))
    epoch_num_binary = number_binary[number_binary.index(min(number_binary))]
    epoch_num_bipolar = number_bipolar[number_bipolar.index(min(number_bipolar))]
    # epoch_max_num_binary = number_binary[number_binary.index(max(number_binary))]
    # epoch_max_num_bipolar = number_bipolar[number_bipolar.index(max(number_bipolar))]
    path_binary = os.path.join(path, "trainTotalError-true-%d" % epoch_num_binary + ".txt")
    # absolute path with binary dataset
    path_bipolar = os.path.join(path, "trainTotalError-false-%d" % epoch_num_bipolar + ".txt")
    # absolute path with bipolar dataset
    # path_binary_max = os.path.join(path, "trainTotalError-true-%d" % epoch_max_num_binary + ".txt")
    # path_bipolar_max = os.path.join(path, "trainTotalError-false-%d" % epoch_max_num_bipolar + ".txt")
    with open(path_binary, "r", encoding='utf-8') as f1:  # open the file
        lines_binary = f1.readlines()
        for count, line in enumerate(lines_binary):
            X_binary.append(count)
            Y_binary.append(float(line))

    plt.xlim(0, max(X_binary)+500)
    plt.ylim(0, max(Y_binary))
    plt.xlabel("Number of epochs")
    plt.ylabel("Error")
    plt.plot(X_binary, Y_binary, color = 'blue', label = 'binary representation')
    # plt.plot(X_binary_max, Y_binary_max, color = 'red', label = 'maximum_epoch_binary')
    plt.legend(loc = 1)
    plt.title("(1) - a): \n standard backpropagation of XOR problem using a binary representation")
    plt.figure(figsize=(6, 8))
    plt.show()

    with open(path_bipolar, "r", encoding='utf-8') as f3:  # open the file
        lines_bipolar = f3.readlines()
        for count, line in enumerate(lines_bipolar):
            X_bipolar.append(count)
            Y_bipolar.append(float(line))
    plt.xlim(0, max(X_bipolar)+50)
    plt.ylim(0, max(Y_bipolar))
    plt.xlabel("Number of epochs")
    plt.ylabel("Error")
    plt.plot(X_bipolar, Y_bipolar, color = 'red', label = 'bipolar representation')
    plt.legend(loc = 1)
    plt.title("(1) - b): \n standard backpropagation of XOR problem using a bipolar representation")
    plt.figure(figsize=(6, 8))
    plt.show()

    for file in files:  # traverse the files in the folder
        if re.search(r'TrainMomentumTotalError', file, re.M):
            if file.split('-', 2)[1] == 'true':
                number_momentum_binary.append(int(file.split('-', 2)[-1].split('.', 1)[0]))
            else:
                number_momentum_bipolar.append(int(file.split('-', 2)[-1].split('.', 1)[0]))
    epoch_num_momentum_binary = number_momentum_binary[number_momentum_binary.index(min(number_momentum_binary))]
    epoch_num_momentum_bipolar = number_momentum_bipolar[number_momentum_bipolar.index(min(number_momentum_bipolar))]
    path_momentum_binary = os.path.join(path, "TrainMomentumTotalError-true-%d" % epoch_num_momentum_binary + ".txt")  # absolute path with binary dataset
    path_momentum_bipolar = os.path.join(path, "TrainMomentumTotalError-false-%d" % epoch_num_momentum_bipolar + ".txt") # absolute path with bipolar dataset

    with open(path_momentum_binary, "r", encoding='utf-8') as f4:  # open the file
        lines_momentum_binary = f4.readlines()
        for count, line in enumerate(lines_momentum_binary):
            X_binary_momentum.append(count)
            Y_binary_momentum.append(float(line))
    with open(path_momentum_bipolar, "r", encoding='utf-8') as f5:  # open the file
        lines_momentum_bipolar = f5.readlines()
        for count, line in enumerate(lines_momentum_bipolar):
            X_bipolar_momentum.append(count)
            Y_bipolar_momentum.append(float(line))
    plt.ylim(0, max(Y_binary_momentum))
    plt.xlim(0, max(X_binary_momentum) + 30)
    plt.xlabel("Number of epochs")
    plt.ylabel("Error")
    plt.plot(X_binary_momentum, Y_binary_momentum, color='blue', label='binary representation(momentum)')
    plt.legend(loc=1)
    plt.title("(1) - c): backpropagation with momentum = 0.9 of \n XOR problem using binary representation")
    plt.figure(figsize=(6, 8))
    plt.show()

    plt.ylim(0, max(Y_bipolar_momentum))
    plt.xlim(0, max(X_bipolar_momentum)+10)
    plt.xlabel("Number of epochs")
    plt.ylabel("Error")
    plt.plot(X_bipolar_momentum, Y_bipolar_momentum, color='red', label='bipolar representation(momentum)')
    plt.legend(loc = 1)
    plt.title("(1) - c): backpropagation with momentum = 0.9 of \n XOR problem using bipolar representation")
    plt.figure(figsize=(6, 8))
    plt.show()
