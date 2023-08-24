import json
import numpy as np
import matplotlib.pyplot as plt

def plot_signal(traces_path):
    files = glob.glob(traces_path + "*.json")

    for file in tqdm(files, total=len(files)):
        file = file.replace('\\', '/')

        with open(file, 'r') as f:
            data = json.load(f)

        r = np.array(data["R"], dtype=float)
        g = np.array(data["G"], dtype=float)
        b = np.array(data["B"], dtype=float)
        t = np.array(data["Times"], dtype=float)

        r_norm = (r - r.min()) / (r.max() - r.min())
        g_norm = (g - g.min()) / (g.max() - g.min())
        b_norm = (b - b.min()) / (b.max() - b.min())

        plt.plot(t, r_norm, 'r')
        plt.plot(t, g_norm, 'g')
        plt.plot(t, b_norm, 'b')
        plt.show()
