import matplotlib.pyplot as plt
import json

plt.style.use("ggplot")

colors = ["red", "green", "blue"]


def plot(logss, names, save=True):
    idx = 0
    for logs in logss:
        name = names[idx]
        color = colors[idx]
        X = [log[0] for log in logs]
        Y = [log[1] for log in logs]

        plt.plot(X, Y, label=name, color=color)
        plt.title("MSE vs Latent Variable Dimension")
        plt.legend(names)
        plt.xlabel("Latent Variable Dimension")
        plt.ylabel("MSE")
        if save:
            plt.savefig(name + ".png")
            plt.close()
        idx += 1


logsPCA = json.load(open("./PCA/logs.json"))
logsPPCA = json.load(open("./PPCA/logs.json"))
logsVAE = json.load(open("./VAE/logs.json"))

# Plotting individual plots
plot([logsPCA], ["PCA"])
plot([logsPPCA], ["PPCA"])
plot([logsVAE], ["VAE"])
plot([logsPCA, logsPPCA, logsVAE], ["PCA", "PPCA", "VAE"], save=False)
plt.savefig("comparison.png")
