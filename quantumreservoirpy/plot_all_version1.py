# standard libs
import numpy as np
import matplotlib.pyplot as plt
import joblib
import tikzplotlib


def calculate_mean_std(arrays):
    stacked_arrays = np.stack(arrays, axis=0)
    mean_values = np.mean(stacked_arrays, axis=0)
    std_values = np.std(stacked_arrays, axis=0)

    return mean_values, std_values


def plot_EV(scores, col, color, style, text):
    mean, std = calculate_mean_std(scores)
    x = range(len(scores[0]))
    plt.plot(x, mean, style + "-" + col, label=r"mean " + text)
    plt.fill_between(
        x,
        mean - std,
        mean + std,
        color=color,
        alpha=0.35,
        label=r"$1\sigma$",
    )


if __name__ == "__main__":
    num_qubits = 4
    num_meas = 3
    num_reservoirs = 3
    degree = 3
    noise_model = None
    num_samples = 10
    num_shots = 10**3

    fig = plt.figure()

    for typ in ["standard", "stabilizer"]:
        filename = (
            "simulation_"
            + str(num_qubits)
            + "_"
            + str(num_meas)
            + "_"
            + str(num_reservoirs)
            + "_"
            + str(degree)
            + "_"
            + str(typ)
            + "_"
            + str(noise_model)
            + "_"
            + str(num_samples)
            + "_"
            + str(num_shots)
            + "_"
            + str()
            + ".sav"
        )
        [scores, predictions, X, y, pipe, states, resmodel] = joblib.load(filename)
        if typ == "standard":
            plot_EV(scores, "r", "red", "x", typ)
        else:
            plot_EV(scores, "b", "blue", "x", typ)
    plt.legend()
    plt.ylim([0, 1.01])
    plt.hlines(1, 0, 21, colors="k", linestyles="dashed")
    plt.xlim([0, 21])
    # tikzplotlib.save("test.tex")
    plt.savefig("test.png")
