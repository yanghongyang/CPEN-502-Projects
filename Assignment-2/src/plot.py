import matplotlib.pyplot as plt

if __name__ == '__main__':
    filename = "E:\\UBC_Graduate\\Term_1\\CPEN%20502\\作业\\Code\\CPEN-502-Projects\\Assignment-2\\out\\production\\Assignment-2\\RLRobot\\IRobot.data\\IRobot-winningRate.log"
    round = []
    winningrate = []
    f = open(filename)
    line = f.readline()
    while line:
        round.append(float(line.split(' ')[0]))
        winningrate.append(float(line.split(' ')[1].replace('\n', '').replace('\r', '')))
        line = f.readline()
    f.close()
    print(round)
    print(winningrate)
    plt.plot(round, winningrate)
    plt.xlabel('# of 20 rounds')
    plt.ylabel('Winning rate')
    plt.title('2-a)')
    plt.show()
    plt.savefig("E:/2-a.png")


