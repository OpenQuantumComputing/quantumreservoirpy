# standard libs
import numpy as np
import matplotlib.pyplot as plt
import joblib
import tikzplotlib

def tikzplotlib_fix_ncols(obj):
    """
    workaround for matplotlib 3.6 renamed legend's _ncol to _ncols, which breaks tikzplotlib
    """
    if hasattr(obj, "_ncols"):
        obj._ncol = obj._ncols
    for child in obj.get_children():
        tikzplotlib_fix_ncols(child)

# sklearn
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge


def fit_model(model, res_states, series, WARMUP):
    warmup = int(len(series) * WARMUP)

    X = res_states
    y = series

    X = X[warmup:]
    y = y[warmup:]

    model.fit(X, y)

    return model, X, y

def fit(states, X_train, pipe, WARMUP):
    scores = []
    predictions = []
    for j in range(1, states.shape[1] + 1):
        pipe, X, y = fit_model(pipe, states[:, :j], X_train, WARMUP)
        score = pipe.score(X, y)
        prediction = pipe.predict(X)
        scores.append(score)
        predictions.append(prediction)

    return scores, predictions, X, y, pipe


def calculate_mean_std(arrays):
    stacked_arrays = np.stack(arrays, axis=0)
    mean_values = np.mean(stacked_arrays, axis=0)
    std_values = np.std(stacked_arrays, axis=0)

    return mean_values, std_values

def plot_EV(scores, col, color, style, text):
    mean, std = calculate_mean_std(scores)
    x=range(len(scores[0]))
    plt.plot(x, mean, style+"-"+col, label=r"mean "+text)
    plt.fill_between(
        x,
        mean - std,
        mean + std,
        color=color,
        alpha=0.35,
        label=r"$1\sigma$",
    )

if __name__ == "__main__":

    WARMUP = 0.3

    num_qubits = 4
    num_meas = 3
    num_reservoirs = 3
    noise_model = None
    num_samples = 2 
    num_shots = 10**3

    degree = num_meas

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
        [ts, states, resmodelparams] = joblib.load(filename)

        assert(len(states)==num_samples)

        scores = []

        for i in range(num_samples):

            pipe = Pipeline(
                steps=[
                    # ('scale', StandardScaler()),
                    # ('preprocessor', PolynomialFeatures(degree=1, include_bias=False)),
                    ("estimator", Ridge(alpha=1e-8))
                ]
            )

            
            sc, _, _, _, _ = fit(states[i], ts, pipe, WARMUP)
            scores.append(sc)

        if typ=="standard":
            plot_EV(scores, 'r', 'red', 'x', typ)
        else:
            plot_EV(scores, 'b', 'blue', 'x', typ)
    plt.legend()
    plt.ylim([0,1.01])
    plt.hlines(1, 0, 21, colors='k', linestyles="dashed")
    plt.xlim([0,21])
    plt.savefig("test.png")
    tikzplotlib_fix_ncols(fig)
    tikzplotlib.save("test.tex")
