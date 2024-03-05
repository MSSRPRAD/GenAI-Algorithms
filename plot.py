import matplotlib.pyplot as plt
import json


def plot(logs, name):
    X = []
    Y = []

    for log in logs:
        X.append(log[0])
        Y.append(log[1])

    plt.plot(X, Y)
    plt.savefig(name + ".png")
    plt.close()


logsPCA = json.load(open("./PCA/logs.json"))
logsPPCA = json.load(open("./PPCA/logs.json"))
# logsVAE = json.load(open("./VAE/logs.json"))

plot(logsPCA, "PCA")
plot(logsPPCA, "PPCA")
# plot(logsVAE, "VAE")
