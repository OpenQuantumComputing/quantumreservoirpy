# standard libs
import numpy as np
import matplotlib.pyplot as plt
import joblib
import tikzplotlib
from scipy.special import comb


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

# from sklearn.svm import SVR
from sklearn.linear_model import Ridge


def fit_model(model, res_states, series, WARMUP):
    warmup = int(len(series) * WARMUP)

    X = res_states
    y = series

    X = X[warmup:]
    y = y[warmup:]

    model.fit(X, y)

    return model, X, y


def fit(states, X_train, pipe, WARMUP, degree=None, maxdegree=None):
    scores = []
    predictions = []

    if maxdegree:
        shift = 2**maxdegree - 1
        num_reservoirs = states.shape[1] / shift
        assert num_reservoirs.is_integer()
        num_reservoirs = int(num_reservoirs)
        # print(states.shape[1], shift, num_reservoirs)

        num_obs = sum_binomial_coefficients(maxdegree, degree)

        indices = []
        for ti in range(num_reservoirs):
            # print(shift, ti, num_obs)
            indices += list(range(shift * ti, shift * ti + num_obs))
            pipe, X, y = fit_model(pipe, states[:, indices], X_train, WARMUP)
            score = pipe.score(X, y)
            prediction = pipe.predict(X)
            scores.append(score)
            predictions.append(prediction)
        # print(indices)

    else:
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


def plot_file(filename, col, color, style, text, WARMUP, degree=None, maxdegree=None):
    # degree defines up to which degree we include the observables
    # max degree, equals maximum degree available, which should be equal to 2^(number of measured states) -1
    [ts, states, resmodelparams] = joblib.load(filename)
    scores = get_scores(states, ts, WARMUP, degree=degree, maxdegree=maxdegree)
    plot_EV(scores, col, color, style, text)


def plot_EV(scores, col, color, style, text):
    mean, std = calculate_mean_std(scores)
    x = range(1, len(scores[0]) + 1)
    plt.plot(x, mean, style + "-" + col, label=r"mean " + text)
    plt.fill_between(
        x,
        mean - std,
        mean + std,
        color=color,
        alpha=0.35,
        label=r"$1\sigma$",
    )


def sum_binomial_coefficients(n, l):
    ret = 0
    for k in range(1, l + 1):
        ret += comb(n, k)
    return int(ret)


def get_scores(states, ts, WARMUP, degree, maxdegree):
    scores = []

    num_samples = len(states)
    for i in range(num_samples):  # number of different samples
        # pipe = Pipeline(
        #    steps=[
        #        # ('scale', StandardScaler()),
        #        # ('preprocessor', PolynomialFeatures(degree=1, include_bias=False)),
        #        ("estimator", Ridge(alpha=1e-6))
        #    ]
        # )
        pipe = Pipeline(steps=[("estimator", LinearRegression(fit_intercept=True))])
        # pipe = Pipeline(steps=[('estimator', SVR())])

        # print(states[i].shape, ts.shape)

        sc, _, _, _, _ = fit(
            states[i], ts, pipe, WARMUP, degree=degree, maxdegree=maxdegree
        )
        scores.append(sc)
    return scores


if __name__ == "__main__":
    WARMUP = 0.3

    if False:
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
                + str(precision)
                + ".sav"
            )
            [ts, states, variances, shots_taken, resmodelparams] = joblib.load(filename)

            assert len(states) == num_samples

            scores = get_scores(states, ts, WARMUP)

            if typ == "standard":
                plot_EV(scores, "r", "red", "x", typ)
            else:
                plot_EV(scores, "b", "blue", "x", typ)
        plt.legend()
        plt.ylim([0, 1.01])
        plt.hlines(1, 0, 21, colors="k", linestyles="dashed")
        plt.xlim([0, 21])
        plt.savefig("test.png")
        tikzplotlib_fix_ncols(fig)
        tikzplotlib.save("test.tex")

    else:
        ## fns = []
        ## fns.append("simulation_6_4_5_4_stabilizer_None_10_10000_.sav")
        # fig = plt.figure()
        ## plot_file(filename, col, color, style, text, WARMUP):
        # plot_file(
        #    "simulation_6_4_5_4_stabilizer_None_10_10000_.sav",
        #    "r",
        #    "red",
        #    "x",
        #    "stabilizer, 10^4 shots",
        #    WARMUP
        # )
        # plot_file(
        #    "simulation_6_4_5_4_stabilizer_None_10_1000_.sav",
        #    "b",
        #    "blue",
        #    "o",
        #    "stabilizer, 10^3 shots",
        #    WARMUP
        # )
        # plot_file(
        #    "simulation_6_4_5_4_standard_None_10_10000_.sav",
        #    "g",
        #    "green",
        #    "x",
        #    "standard, 10^4 shots",
        #    WARMUP
        # )
        # plot_file(
        #    "simulation_6_4_5_4_standard_None_10_1000_.sav",
        #    "k",
        #    "black",
        #    "o",
        #    "standard, 10^3 shots",
        #    WARMUP
        # )

        ## fns.append("simulation_6_4_5_4_standard_None_10_10000_.sav")

        ## fns.append("simulation_6_4_5_4_standard_None_10_1000_.sav")
        ## fns.append("simulation_8_6_5_6_stabilizer_None_10_1000_.sav")
        ## fns.append("simulation_8_6_5_6_standard_None_10_1000_.sav")

        ## for filename in fns:
        ##    [ts, states, resmodelparams] = joblib.load(filename)

        ##    scores = get_scores(states, ts, WARMUP)

        ##    plot_EV(scores, "r", "red", "x", "one of the two")
        # plt.legend()
        ## plt.ylim([0,1.01])
        # plt.hlines(1, 0, 71, colors='k', linestyles="dashed")
        ## plt.xlim([0,21])
        # plt.ylabel("score")
        # plt.xlabel("number of observables")
        # plt.savefig("test.png")
        # tikzplotlib_fix_ncols(fig)
        # tikzplotlib.save("test.tex")

        ## fns = []
        ## fns.append("simulation_6_4_5_4_stabilizer_None_10_10000_.sav")
        # fig = plt.figure()
        ## plot_file(filename, col, color, style, text, WARMUP):
        # plot_file(
        #    "simulation_6_4_5_4_stabilizer_None_10_1000_.sav",
        #    "b",
        #    "blue",
        #    "o",
        #    "stabilizer, 6,4, 10^3 shots",
        #    WARMUP
        # )
        # plot_file(
        #    "simulation_6_4_5_4_standard_None_10_1000_.sav",
        #    "k",
        #    "black",
        #    "o",
        #    "standard,  6,4,10^3 shots",
        #    WARMUP
        # )

        # plot_file(
        #    "simulation_8_6_5_6_stabilizer_None_10_1000_.sav",
        #    "r",
        #    "red",
        #    "o",
        #    "stabilizer, 8,6, 10^3 shots",
        #    WARMUP
        # )
        # plot_file(
        #    "simulation_8_6_5_6_standard_None_10_1000_.sav",
        #    "g",
        #    "green",
        #    "o",
        #    "standard, 8,6, 0^3 shots",
        #    WARMUP
        # )

        ## fns.append("simulation_6_4_5_4_standard_None_10_10000_.sav")

        ## fns.append("simulation_6_4_5_4_standard_None_10_1000_.sav")
        ## fns.append("simulation_8_6_5_6_stabilizer_None_10_1000_.sav")
        ## fns.append("simulation_8_6_5_6_standard_None_10_1000_.sav")

        ## for filename in fns:
        ##    [ts, states, resmodelparams] = joblib.load(filename)

        ##    scores = get_scores(states, ts, WARMUP)

        ##    plot_EV(scores, "r", "red", "x", "one of the two")
        # plt.legend()
        ## plt.ylim([0,1.01])
        # plt.hlines(1, 0, 71, colors='k', linestyles="dashed")
        ## plt.xlim([0,21])
        # plt.ylabel("score")
        # plt.xlabel("number of observables")
        # plt.savefig("test2.png")
        # tikzplotlib_fix_ncols(fig)
        # tikzplotlib.save("test2.tex")

        # fns = []
        # fns.append("simulation_6_4_5_4_stabilizer_None_10_10000_.sav")
        fig = plt.figure()
        # plot_file(filename, col, color, style, text, WARMUP):
        # plot_file(
        #    "simulation_7_6_5_6_stabilizer_None_10_10000_.sav",
        #    "c",
        #    "cyan",
        #    "o",
        #    "stabilizer, 7,6,5, 10^4 shots",
        #    WARMUP
        # )
        plot_file(
            "simulation_5_3_5_3_standard_None_10_10000_.sav",
            "k",
            "black",
            "o",
            "standard,  5,3,5, 10^4 shots",
            WARMUP,
            degree=3,
            maxdegree=3,
        )
        plot_file(
            "simulation_4_3_5_3_standard_None_10_10000_.sav",
            "c",
            "cyan",
            "o",
            "standard, 4,3,5, 10^4 shots",
            WARMUP,
            degree=3,
            maxdegree=3,
        )

        plot_file(
            "simulation_4_3_5_3_stabilizer_None_10_10000_.sav",
            "b",
            "blue",
            "o",
            "stabilizer, 4,3,5, 10^4 shots",
            WARMUP,
            degree=3,
            maxdegree=3,
        )
        plot_file(
            "simulation_5_3_5_3_stabilizer_None_10_10000_.sav",
            "r",
            "red",
            "o",
            "stabilizer, 5,3,5, 10^4 shots",
            WARMUP,
            degree=3,
            maxdegree=3,
        )
        # plot_file(
        #    "simulation_8_6_5_6_standard_None_10_1000_.sav",
        #    "k",
        #    "black",
        #    "o",
        #    "standard,  8,6,5, 10^3 shots",
        #    WARMUP,
        #    degree=6,
        #    maxdegree=6
        # )

        # plot_file(
        #    "simulation_7_6_5_6_stabilizer_None_10_1000_.sav",
        #    "b",
        #    "blue",
        #    "o",
        #    "stabilizer, 7,6,5, 10^3 shots",
        #    WARMUP,
        #    degree=6,
        #    maxdegree=6

        # )
        # plot_file(
        #    "simulation_6_5_5_5_standard_None_10_10000_.sav",
        #    "g",
        #    "green",
        #    "o",
        #    "standard,  6,5,5, 10^4 shots",
        #    WARMUP
        # )

        # plot_file(
        #    "simulation_5_3_5_3_stabilizer_None_10_10000_.sav",
        #    "y",
        #    "yellow",
        #    "o",
        #    "stabilizer, 5,3, 10^4 shots",
        #    WARMUP
        # )
        # plot_file(
        #    "simulation_5_3_5_3_standard_None_10_10000_.sav",
        #    "m",
        #    "magenta",
        #    "o",
        #    "standard, 5,3, 10^4 shots",
        #    WARMUP
        # )

        # fns.append("simulation_6_4_5_4_standard_None_10_10000_.sav")

        # fns.append("simulation_6_4_5_4_standard_None_10_1000_.sav")
        # fns.append("simulation_8_6_5_6_stabilizer_None_10_1000_.sav")
        # fns.append("simulation_8_6_5_6_standard_None_10_1000_.sav")

        # for filename in fns:
        #    [ts, states, resmodelparams] = joblib.load(filename)

        #    scores = get_scores(states, ts, WARMUP)

        #    plot_EV(scores, "r", "red", "x", "one of the two")
        plt.legend()
        # plt.ylim([0.5,1.01])
        # plt.hlines(1, 0, 71, colors='k', linestyles="dashed")
        # plt.xlim([2,100])
        plt.ylabel("score")
        plt.xlabel("number of observables")
        plt.savefig("test2.png")
        tikzplotlib_fix_ncols(fig)
        tikzplotlib.save("test2.tex")
