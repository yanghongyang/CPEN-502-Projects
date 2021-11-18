import matplotlib.pyplot as plt
import os
import re
def plot(res, str, path):
    global label3_c, label1_c, label2_c, label4_c, label1_a, label2_a, label3_a, label4_a
    path_save = r"/Users/hannah/Desktop/log_f2/"  # training result  documents path
    filename = []
    for file in res:
        if re.search(r'log', file):
            filename.append(file)
    if len(filename) == 1:
        logname = filename[0]
        round = []
        winningrate = []
        f = open(path+"/"+logname)
        line = f.readline()
        while line:
            round.append(float(line.split(' ')[0]))
            winningrate.append(float(line.split(' ')[1].replace('\n', '').replace('\r', '')))
            line = f.readline()
        f.close()
        plt.plot(round, winningrate)
        plt.xlabel('# of 100 rounds')
        plt.ylabel('Winning rate')
        plt.title("2-a)-progress of learning with winning rate")
        plt.savefig(path_save+"result/"+"2a"+".png")
        plt.show()
    elif (len(filename) == 4) & (str == "2-b)"):
        logname1_a = ""
        logname2_a = ""
        logname3_a = ""
        logname4_a = ""
        if re.search("b", str):
            for name in filename:
                if r'immediate' in name and 'onPolicy' in name:
                    logname1_a = name
                    label1_a = "immediate&onPolicy"
                elif r'immediate' in name and 'offPolicy' in name:
                    logname2_a = name
                    label2_a = "immediate&offPolicy"
                elif r'terminal' in name and 'onPolicy' in name:
                    logname3_a = name
                    label3_a = "terminal&onPolicy"
                elif r'terminal' in name and 'offPolicy' in name:
                    logname4_a = name
                    label4_a = "terminal&offPolicy"

        round1_a = []
        winningrate1_a = []
        f1_a = open(path+"/"+logname1_a)
        line1_a = f1_a.readline()
        round2_a = []
        winningrate2_a = []
        f2_a = open(path+"/"+logname2_a)
        line2_a = f2_a.readline()
        round3_a = []
        winningrate3_a = []
        f3_a = open(path+"/"+logname3_a)
        line3_a = f3_a.readline()
        round4_a = []
        winningrate4_a = []
        f4_a = open(path+"/"+logname4_a)
        line4_a = f4_a.readline()
        while line1_a:
            round1_a.append(float(line1_a.split(' ')[0]))
            winningrate1_a.append(float(line1_a.split(' ')[1].replace('\n', '').replace('\r', '')))
            line1_a = f1_a.readline()
        f1_a.close()
        while line2_a:
            round2_a.append(float(line2_a.split(' ')[0]))
            winningrate2_a.append(float(line2_a.split(' ')[1].replace('\n', '').replace('\r', '')))
            line2_a = f2_a.readline()
        f2_a.close()
        while line3_a:
            round3_a.append(float(line3_a.split(' ')[0]))
            winningrate3_a.append(float(line3_a.split(' ')[1].replace('\n', '').replace('\r', '')))
            line3_a = f3_a.readline()
        f3_a.close()
        while line4_a:
            round4_a.append(float(line4_a.split(' ')[0]))
            winningrate4_a.append(float(line4_a.split(' ')[1].replace('\n', '').replace('\r', '')))
            line4_a = f4_a.readline()
        f4_a.close()
        graph1 = plt.plot(round1_a, winningrate1_a, round2_a, winningrate2_a)
        plt.setp(graph1[0], color='blue', label=label1_a)
        plt.setp(graph1[1], color='red', label=label2_a)
        plt.legend(loc=4)
        plt.xlabel('# of 100 rounds')
        plt.ylabel('Winning rate')
        plt.title("2-b)- on-policy learning VS off-policy performance(intermediate rewards)")
        plt.savefig(path_save+"result/2-b-immediate.png")
        plt.show()
        graph2 = plt.plot(round3_a, winningrate3_a, round4_a, winningrate4_a)
        plt.setp(graph2[0], color='orange', label=label3_a)
        plt.setp(graph2[1], color='purple', label=label4_a)
        plt.legend(loc=4)
        plt.xlabel('# of 100 rounds')
        plt.ylabel('Winning rate')
        plt.title("2-b)- on-policy learning VS off-policy performance(terminal rewards)")
        plt.savefig(path_save+"result/2-b-terminal.png")
        plt.show()
    elif (len(filename) == 4) & (str == "2-c)"):
        logname1_c = ""
        logname2_c = ""
        logname3_c = ""
        logname4_c = ""
        if re.search("c", str):
            for name in filename:
                if r'immediate' in name and 'onPolicy' in name:
                    logname1_c = name
                    label1_c = "immediate&onPolicy"
                elif r'immediate' in name and 'offPolicy' in name:
                    logname2_c = name
                    label2_c = "immediate&offPolicy"
                elif r'terminal' in name and 'onPolicy' in name:
                    logname3_c = name
                    label3_c = "terminal&onPolicy"
                elif r'terminal' in name and 'offPolicy' in name:
                    logname4_c = name
                    label4_c = "terminal&offPolicy"
        round1_c = []
        winningrate1_c = []
        f1_c = open(path + "/" + logname1_c)
        line1_c = f1_c.readline()
        round2_c = []
        winningrate2_c = []
        f2_c = open(path + "/" + logname2_c)
        line2_c = f2_c.readline()
        round3_c = []
        winningrate3_c = []
        f3_c = open(path+"/"+logname3_c)
        line3_c = f3_c.readline()
        round4_c = []
        winningrate4_c = []
        f4_c = open(path+"/"+logname4_c)
        line4_c = f4_c.readline()
        while line1_c:
            round1_c.append(float(line1_c.split(' ')[0]))
            winningrate1_c.append(float(line1_c.split(' ')[1].replace('\n', '').replace('\r', '')))
            line1_c = f1_c.readline()
        f1_c.close()
        while line2_c:
            round2_c.append(float(line2_c.split(' ')[0]))
            winningrate2_c.append(float(line2_c.split(' ')[1].replace('\n', '').replace('\r', '')))
            line2_c = f2_c.readline()
        f2_c.close()
        while line3_c:
            round3_c.append(float(line3_c.split(' ')[0]))
            winningrate3_c.append(float(line3_c.split(' ')[1].replace('\n', '').replace('\r', '')))
            line3_c = f3_c.readline()
        f3_c.close()
        while line4_c:
            round4_c.append(float(line4_c.split(' ')[0]))
            winningrate4_c.append(float(line4_c.split(' ')[1].replace('\n', '').replace('\r', '')))
            line4_c = f4_c.readline()
        f4_c.close()
        graph_c1 = plt.plot(round1_c, winningrate1_c, round3_c, winningrate3_c)

        plt.setp(graph_c1[0], color='blue', label=label1_c)
        plt.setp(graph_c1[1], color='red', label=label3_c)

        plt.legend(loc=4)
        plt.xlabel('# of 100 rounds')
        plt.ylabel('Winning rate')
        plt.title("behaviour with terminal rewards VS intermediate rewards(on-policy)")
        plt.savefig(path_save + "result/" + "2c-onPolicy" + ".png")
        plt.show()
        graph_c2 = plt.plot(round2_c, winningrate2_c, round4_c, winningrate4_c)
        plt.setp(graph_c2[0], color='orange', label=label2_c)
        plt.setp(graph_c2[1], color='purple', label=label4_c)
        plt.xlabel('# of 100 rounds')
        plt.ylabel('Winning rate')
        plt.title("behaviour with terminal rewards VS intermediate rewards(off-policy)")
        plt.savefig(path_save + "result/" + "2c-offPolicy" + ".png")
        plt.show()
    elif (len(filename) == 4) & (str == "3-a)"):
        logname1_3a = filename[0]
        label1_3a = "e: " + logname1_3a.split("-")[-2]
        logname2_3a = filename[1]
        label2_3a = "e: " + logname2_3a.split("-")[-2]
        logname3_3a = filename[2]
        label3_3a = "e: " + logname3_3a.split("-")[-2]
        logname4_3a = filename[3]
        label4_3a = "e: " + logname4_3a.split("-")[-2]
        if label1_3a == "e: 0.1":
            label1_3a = "myRobot(e=0.1)"
        if label2_3a == "e: 0.1":
            label2_3a = "myRobot(e=0.1)"
        if label3_3a == "e: 0.1":
            label3_3a = "myRobot(e=0.1)"
        if label4_3a == "e: 0.1":
            label4_3a = "myRobot(e=0.1)"
        # logname5 = filename[4]
        # label5 = logname5.split("-")[-2]
        round1_3a = []
        winningrate1_3a = []
        f1_3a = open(path+"/"+logname1_3a)
        line1_3a = f1_3a.readline()
        round2_3a = []
        winningrate2_3a = []
        f2_3a = open(path+"/"+logname2_3a)
        line2_3a = f2_3a.readline()
        round3_3a = []
        winningrate3_3a = []
        f3_3a = open(path+"/"+logname3_3a)
        line3_3a = f3_3a.readline()
        round4_3a = []
        winningrate4_3a = []
        f4_3a = open(path+"/"+logname4_3a)
        line4_3a = f4_3a.readline()
        # round5 = []
        # winningrate5 = []
        # f5 = open(path+"/"+logname5)
        # line5 = f5.readline()
        while line1_3a:
            round1_3a.append(float(line1_3a.split(' ')[0]))
            winningrate1_3a.append(float(line1_3a.split(' ')[1].replace('\n', '').replace('\r', '')))
            line1_3a = f1_3a.readline()
        f1_3a.close()
        while line2_3a:
            round2_3a.append(float(line2_3a.split(' ')[0]))
            winningrate2_3a.append(float(line2_3a.split(' ')[1].replace('\n', '').replace('\r', '')))
            line2_3a = f2_3a.readline()
        f2_3a.close()
        while line3_3a:
            round3_3a.append(float(line3_3a.split(' ')[0]))
            winningrate3_3a.append(float(line3_3a.split(' ')[1].replace('\n', '').replace('\r', '')))
            line3_3a = f3_3a.readline()
        f3_3a.close()
        while line4_3a:
            round4_3a.append(float(line4_3a.split(' ')[0]))
            winningrate4_3a.append(float(line4_3a.split(' ')[1].replace('\n', '').replace('\r', '')))
            line4_3a = f4_3a.readline()
        f4_3a.close()
        # while line5:
        #     round5.append(float(line5.split(' ')[0]))
        #     winningrate5.append(float(line5.split(' ')[1].replace('\n', '').replace('\r', '')))
        #     line5 = f5.readline()
        # f5.close()

        graph = plt.plot(round1_3a, winningrate1_3a, round2_3a, winningrate2_3a, round3_3a, winningrate3_3a, round4_3a, winningrate4_3a)
        plt.setp(graph[0], color='blue', label=label1_3a)
        plt.setp(graph[1], color='red', label=label2_3a)
        plt.setp(graph[2], color='orange', label=label3_3a)
        plt.setp(graph[3], color='purple', label=label4_3a)
        # plt.setp(graph[4], color='gray', label=label5)
        plt.legend(loc=4)
        plt.xlabel('# of 100 rounds')
        plt.ylabel('Winning rate')
        plt.title("Comparison of training performance using different values of e")
        plt.savefig(path_save+"result/"+"3a"+".png")
        plt.show()
if __name__ == '__main__':
    path = r"/Users/hannah/Desktop/log_f2/" # training result  documents path
    files = os.listdir(path) # acquire all document name under a folder
    # file = "IRobot-winningRate.log"
    #
    for file in files:  # 遍历文件夹
        folder = path + file
        if re.search(r'2a', file):
            plot(os.listdir(folder), "2-a)", folder)
        if re.search(r'2b', file):
            plot(os.listdir(folder), "2-b)", folder)
        if re.search(r'2c', file):
            plot(os.listdir(folder), "2-c)", folder)
        if re.search(r'3a', file):
            plot(os.listdir(folder), "3-a)", folder)




