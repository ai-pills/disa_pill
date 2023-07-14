import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp
from sklearn import metrics
from scipy import stats
from ortools.linear_solver import pywraplp
import os
import json

# Configuration
anomaly_color = 'sandybrown'
prediction_color = 'yellowgreen'
training_color = 'yellowgreen'
validation_color = 'gold'
test_color = 'coral'

def click_through_rate(avg_ratings, num_reviews, dollar_ratings):
    dollar_rating_baseline = {"D": 3, "DD": 2, "DDD": 4, "DDDD": 4.5}
    return 1 / (1 + np.exp(
        np.array([dollar_rating_baseline[d] for d in dollar_ratings]) -
        avg_ratings * np.log1p(num_reviews) / 4))

def load_restaurant_data():
    def sample_restaurants(n):
        avg_ratings = np.random.uniform(1.0, 5.0, n)
        num_reviews = np.round(np.exp(np.random.uniform(0.0, np.log(200), n)))
        dollar_ratings = np.random.choice(["D", "DD", "DDD", "DDDD"], n)
        ctr_labels = click_through_rate(avg_ratings, num_reviews, dollar_ratings)
        return avg_ratings, num_reviews, dollar_ratings, ctr_labels


    def sample_dataset(n, testing_set):
        (avg_ratings, num_reviews, dollar_ratings, ctr_labels) = sample_restaurants(n)
        if testing_set:
            # Testing has a more uniform distribution over all restaurants.
            num_views = np.random.poisson(lam=3, size=n)
        else:
            # Training/validation datasets have more views on popular restaurants.
            num_views = np.random.poisson(lam=ctr_labels * num_reviews / 40.0, size=n)

        return pd.DataFrame({
                "avg_rating": np.repeat(avg_ratings, num_views),
                "num_reviews": np.repeat(num_reviews, num_views),
                "dollar_rating": np.repeat(dollar_ratings, num_views),
                "clicked": np.random.binomial(n=1, p=np.repeat(ctr_labels, num_views))
            })

    # Generate
    np.random.seed(42)
    data_train = sample_dataset(2000, testing_set=False)
    data_val = sample_dataset(1000, testing_set=False)
    data_test = sample_dataset(1000, testing_set=True)
    return data_train, data_val, data_test


def plot_ctr_truth(figsize=None):
    plt.figure(figsize=figsize)
    res = 100
    nticks = 3
    avgr = np.repeat(np.linspace(0, 5, res), res)
    nrev = np.tile(np.linspace(0, 200, res), res)
    avgr_ticks = np.linspace(0, 5, nticks)
    nrev_ticks = np.linspace(0, 200, nticks)
    rticks = np.linspace(0, res, nticks)
    for i, drating in enumerate(['D', 'DD', 'DDD', 'DDDD']):
        drt = [drating] * (res*res)
        ctr = click_through_rate(avgr, nrev, drt)
        plt.subplot(1, 4, i+1)
        plt.pcolor(ctr.reshape((res, res)), vmin=0, vmax=1)
        plt.xlabel('average rating')
        if i == 0:
            plt.ylabel('num. reviews')
        plt.title(drating)
        plt.xticks(rticks, avgr_ticks, fontsize=7)
        if i == 0:
            plt.yticks(rticks, nrev_ticks, fontsize=7)
        else:
            plt.yticks([], [])
    plt.tight_layout()


def plot_ctr_distribution(data, figsize=None):
    plt.figure(figsize=figsize)
    nbins = 15
    plt.subplot(131)
    plt.hist(data['avg_rating'], density=True, bins=nbins)
    plt.xlabel('average rating')
    plt.subplot(132)
    plt.hist(data['num_reviews'], density=True, bins=nbins)
    plt.xlabel('num. reviews')
    plt.subplot(133)
    vcnt = data['dollar_rating'].value_counts()
    vcnt /= vcnt.sum()
    plt.bar([0.5, 1.5, 2.5, 3.5],
            [vcnt['D'], vcnt['DD'], vcnt['DDD'], vcnt['DDDD']])
    plt.xlabel('dollar rating')
    plt.tight_layout()


def build_nn_model(input_shape, output_shape, hidden,
        output_activation='linear', kernel_regularizers=[],
                   scale=None, name=None):
    model_in = keras.Input(shape=input_shape, dtype='float32')
    x = model_in
    for i, h in enumerate(hidden):
        kr = kernel_reguralizers[i] if i < len(kernel_regularizers)-1 else None
        x = layers.Dense(h, activation='relu', kernel_regularizer=kr)(x)
    kr_out = kernel_regularizers[-1] if len(kernel_regularizers) > len(hidden) else None
    model_out = layers.Dense(output_shape, activation=output_activation,
            kernel_regularizer=kr_out)(x)
    if scale is not None:
        model_out *= scale
    model = keras.Model(model_in, model_out, name=name)
    return model


def plot_nn_model(model, show_layer_names=True, show_layer_activations=True, show_shapes=True):
    return keras.utils.plot_model(model, show_shapes=show_shapes,
            show_layer_names=show_layer_names, rankdir='LR',
            show_layer_activations=show_layer_activations)


def train_nn_model(model, X, y, loss,
        verbose=0, patience=10,
        validation_split=0.0, **fit_params):
    # Compile the model
    model.compile(optimizer='Adam', loss=loss)
    # Build the early stop callback
    cb = []
    if validation_split > 0:
        cb += [callbacks.EarlyStopping(patience=patience,
            restore_best_weights=True)]
    # Train the model
    history = model.fit(X, y, callbacks=cb,
            validation_split=validation_split,
            verbose=verbose, **fit_params)
    return history


# def plot_training_history(history=None,
#         figsize=None, print_final_scores=True):
#     plt.figure(figsize=figsize)
#     for metric in history.history.keys():
#         plt.plot(history.history[metric], label=metric)
#     # if 'val_loss' in history.history.keys():
#     #     plt.plot(history.history['val_loss'], label='val. loss')
#     if len(history.history.keys()) > 0:
#         plt.legend()
#     plt.xlabel('epochs')
#     plt.grid(linestyle=':')
#     plt.tight_layout()
#     plt.show()
#     if print_final_scores:
#         trl = history.history["loss"][-1]
#         s = f'Final loss: {trl:.4f} (training)'
#         if 'val_loss' in history.history:
#             vll = history.history["val_loss"][-1]
#             s += f', {vll:.4f} (validation)'
#         print(s)

def plot_training_history(history=None,
        figsize=None, print_final_scores=True, excluded_metrics=[]):
    plt.figure(figsize=figsize)
    for metric in history.history.keys():
        if metric not in excluded_metrics:
            plt.plot(history.history[metric], label=metric)
    # if 'val_loss' in history.history.keys():
    #     plt.plot(history.history['val_loss'], label='val. loss')
    if len(history.history.keys()) > 0:
        plt.legend()
    plt.xlabel('epochs')
    plt.grid(linestyle=':')
    plt.tight_layout()
    plt.show()
    if print_final_scores:
        trl = history.history["loss"][-1]
        s = f'Final loss: {trl:.4f} (training)'
        if 'val_loss' in history.history:
            vll = history.history["val_loss"][-1]
            s += f', {vll:.4f} (validation)'
        print(s)


def plot_ctr_estimation(estimator, scale,
        split_input=False, one_hot_categorical=True,
        figsize=None):
    plt.figure(figsize=figsize)
    res = 100
    nticks = 3
    avgr = np.repeat(np.linspace(0, 5, res), res).reshape(-1, 1)
    avgr = avgr / scale['avg_rating']
    nrev = np.tile(np.linspace(0, 200, res), res).reshape(-1, 1)
    nrev = nrev / scale['num_reviews']
    avgr_ticks = np.linspace(0, 5, nticks)
    nrev_ticks = np.linspace(0, 200, nticks)
    rticks = np.linspace(0, res, nticks)
    for i, drating in enumerate(['D', 'DD', 'DDD', 'DDDD']):
        if one_hot_categorical:
            # Categorical encoding for the dollar rating
            dr_cat = np.zeros((1, 4))
            dr_cat[0, i] = 1
            dr_cat = np.repeat((dr_cat), res*res, axis=0)
            # Concatenate all inputs
            x = np.hstack((avgr, nrev, dr_cat))
        else:
            # Integer encoding for the categorical attribute
            dr_cat = np.full((res*res, 1), i)
            x = np.hstack((avgr, nrev, dr_cat))
        # Split input, if requested
        if split_input:
            x = [x[:, i].reshape(-1, 1) for i in range(x.shape[1])]
        # Obtain the predictions
        ctr = estimator.predict(x, verbose=0)
        plt.subplot(1, 4, i+1)
        plt.pcolor(ctr.reshape((res, res)), vmin=0, vmax=1)
        plt.xlabel('average rating')
        if i == 0:
            plt.ylabel('num. reviews')
        plt.title(drating)
        plt.xticks(rticks, avgr_ticks, fontsize=7)
        if i == 0:
            plt.yticks(rticks, nrev_ticks, fontsize=7)
        else:
            plt.yticks([], [])
    plt.tight_layout()


def plot_ctr_calibration(calibrators, scale, figsize=None):
    plt.figure(figsize=figsize)
    res = 100
    nticks = 3

    # Average rating calibration
    avgr = np.linspace(0, 5, res).reshape(-1, 1)
    avgr = avgr / scale['avg_rating']
    avgr_cal = calibrators[0].predict(avgr, verbose=0)
    plt.subplot(131)
    plt.plot(avgr, avgr_cal)
    plt.xlabel('avg_rating')
    plt.ylabel('cal. output')
    plt.grid(linestyle=':')
    # Num. review calibration
    nrev = np.linspace(0, 200, res).reshape(-1, 1)
    nrev = nrev / scale['num_reviews']
    nrev_cal = calibrators[1].predict(nrev, verbose=0)
    plt.subplot(132)
    plt.plot(nrev, nrev_cal)
    plt.xlabel('num_reviews')
    plt.grid(linestyle=':')
    # Dollar rating calibration
    drating = np.arange(0, 4).reshape(-1, 1)
    drating_cal = calibrators[2].predict(drating, verbose=0).ravel()
    plt.subplot(133)
    xticks = np.linspace(0.5, 3.5, 4)
    plt.bar(xticks, drating_cal)
    plt.xticks(xticks, ['D', 'DD', 'DDD', 'DDDD'])
    plt.grid(linestyle=':')

    plt.tight_layout()


def load_ed_data(data_file):
    # Read the CSV file
    data = pd.read_csv(data_file, sep=';', parse_dates=[2, 3])
    # Remove the "Flow" column
    f_cols = [c for c in data.columns if c != 'Flow']
    data = data[f_cols]
    # Convert a few fields to categorical format
    data['Code'] = data['Code'].astype('category')
    data['Outcome'] = data['Outcome'].astype('category')
    # Sort by triage time
    data.sort_values(by='Triage', inplace=True)
    # Discard the firl
    return data


def plot_bars(data, figsize=None, tick_gap=1, series=None):
    plt.figure(figsize=figsize)
    # x = np.arange(len(data))
    # x = 0.5 + np.arange(len(data))
    # plt.bar(x, data, width=0.7)
    # x = data.index-0.5
    x = data.index
    plt.bar(x, data, width=0.7)
    # plt.bar(x, data, width=0.7)
    if series is not None:
        # plt.plot(series.index-0.5, series, color='tab:orange')
        plt.plot(series.index, series, color='tab:orange')
    if tick_gap > 0:
        plt.xticks(x[::tick_gap], data.index[::tick_gap], rotation=45)
    plt.grid(linestyle=':')
    plt.tight_layout()


def plot_series(data, labels=None,
                    windows=None,
                    predictions=None,
                    highlights=None,
                    val_start=None,
                    test_start=None,
                    figsize=None,
                    show_sampling_points=False,
                    show_markers=False,
                    filled_version=None,
                    std=None,
                    ci=None,
                    title=None,
                    ylim=None,
                    threshold=None):
    # Open a new figure
    plt.figure(figsize=figsize)
    # Plot data
    if not show_markers:
        plt.plot(data.index, data.values, zorder=0)
    else:
        plt.plot(data.index, data.values, zorder=0,
                marker='.', markersize=3)
    if filled_version is not None:
        filled = filled_version.copy()
        filled[~data['value'].isnull()] = np.nan
        plt.scatter(filled.index, filled,
                marker='.', c='tab:orange', s=5);
    if show_sampling_points:
        vmin = data.min()
        lvl = np.full(len(data.index), vmin)
        plt.scatter(data.index, lvl, marker='.',
                c='tab:red', s=5)
    # Plot standard deviations
    if std is not None:
        lb = data.values.ravel() - std.values.ravel()
        ub = data.values.ravel() + std.values.ravel()
        plt.fill_between(data.index, lb, ub, alpha=0.3, label='+/- std')
    # Plot confidence intervals
    if ci is not None:
        lb = ci[0].values.ravel()
        ub = ci[1].values.ravel()
        plt.fill_between(data.index, lb, ub, alpha=0.3, label='C.I.')
    # Rotated x ticks
    plt.xticks(rotation=45)
    # Plot labels
    if labels is not None:
        plt.scatter(labels.values, data.loc[labels],
                    color='tab:orange', zorder=2, s=2*np.max(figsize) if figsize else None)
    # Plot windows
    if windows is not None:
        for _, wdw in windows.iterrows():
            plt.axvspan(wdw['begin'], wdw['end'],
                        color='tab:orange', alpha=0.3, zorder=1)
    # Plot training data
    if val_start is not None:
        plt.axvspan(data.index[0], val_start,
                    color=training_color, alpha=0.1, zorder=-1)
    if val_start is None and test_start is not None:
        plt.axvspan(data.index[0], test_start,
                    color=training_color, alpha=0.1, zorder=-1)
    if val_start is not None:
        plt.axvspan(val_start, test_start,
                    color=validation_color, alpha=0.1, zorder=-1)
    if test_start is not None:
        plt.axvspan(test_start, data.index[-1],
                    color=test_color, alpha=0.3, zorder=0)
    # Predictions
    if predictions is not None:
        plt.scatter(predictions.values, data.loc[predictions],
                    color=prediction_color, alpha=.4, zorder=3,
                    s=5)
    # Plot threshold
    if threshold is not None:
        plt.plot([data.index[0], data.index[-1]], [threshold, threshold], linestyle=':', color='tab:red')
    # Force y limits
    if ylim is not None:
        plt.ylim(*ylim)
    plt.grid(linestyle=':')
    plt.title(title)
    plt.tight_layout()


def build_nn_poisson_model(input_shape, hidden, rate_guess=1):
    model_in = keras.Input(shape=input_shape, dtype='float32')
    x = model_in
    for h in hidden:
        x = layers.Dense(h, activation='relu')(x)
    log_rate = layers.Dense(1, activation='linear')(x)
    lf = lambda t: tfp.distributions.Poisson(rate=rate_guess * tf.math.exp(t))
    model_out = tfp.layers.DistributionLambda(lf)(log_rate)
    model = keras.Model(model_in, model_out)
    return model


def plot_pred_scatter(y_true, y_pred, figsize=None, print_metrics=True, alpha=0.1):
    plt.figure(figsize=figsize)
    plt.scatter(y_pred, y_true, marker='.', alpha=alpha)
    xl, xu = plt.xlim()
    yl, yu = plt.ylim()
    l, u = min(xl, yl), max(xu, yu)
    plt.plot([l, u], [l, u], ':', c='0.3')
    plt.grid(linestyle=':')
    plt.xlim(l, u)
    plt.ylim(l, u)
    plt.xlabel('prediction')
    plt.ylabel('target')
    plt.tight_layout()

    if print_metrics:
        print(f'R2: {metrics.r2_score(y_true, y_pred):.2f}')
        print(f'MAE: {metrics.mean_absolute_error(y_true, y_pred):.2f}')


def generate_costs(nsamples, nitems, noise_scale=0,
                   seed=None, sampling_seed=None,
                   nsamples_per_point=1,
                   noise_type='normal', noise_scale_type='absolute'):
    assert(noise_scale >= 0)
    assert(nsamples_per_point > 0)
    assert(noise_type in ('normal', 'rayleigh'))
    assert(noise_scale_type in ('absolute', 'relative'))
    # Seed the RNG
    np.random.seed(seed)
    # Generate costs
    speed = np.random.choice([-14, -11, 11, 14], size=nitems)
    scale = 1.3 + 1.3 * np.random.rand(nitems)
    base = 1 * np.random.rand(nitems) / scale
    offset = -0.75 + 0.5 * np.random.rand(nitems)

    # Generate input
    if sampling_seed is not None:
        np.random.seed(sampling_seed)
    x = np.random.rand(nsamples)
    x = np.repeat(x, nsamples_per_point)

    # Prepare a result dataset
    res = pd.DataFrame(data=x, columns=['x'])

    # scale = np.sort(scale)[::-1]
    for i in range(nitems):
        # Compute base cost
        cost = scale[i] / (1 + np.exp(-speed[i] * (x + offset[i])))
        # Rebase
        cost = cost - np.min(cost) + base[i]
        # sx = direction[i]*speed[i]*(x+offset[i])
        # cost = base[i] + scale[i] / (1 + np.exp(sx))
        res[f'C{i}'] = cost
    # Add noise
    if noise_scale > 0:
        for i in range(nitems):
            # Define the noise scale
            if noise_scale_type == 'absolute':
                noise_scale_vals = noise_scale * res[f'C{i}']**0
            elif noise_scale_type == 'relative':
                noise_scale_vals = noise_scale * res[f'C{i}']
            # Define the noise distribution
            if noise_type == 'normal':
                noise_dist = stats.norm(scale=noise_scale_vals)
            elif noise_type == 'rayleigh':
                # noise_dist = stats.expon(scale=noise_scale_vals)
                noise_dist = stats.rayleigh(scale=noise_scale_vals)
            r_mean = noise_dist.mean()
            # pnoise = noise * np.random.randn(nsamples)
            # res[f'C{i}'] = res[f'C{i}'] + pnoise
            pnoise = noise_dist.rvs()
            res[f'C{i}'] = res[f'C{i}'] + pnoise - r_mean
    # Reindex
    res.set_index('x', inplace=True)
    # Sort by index
    res.sort_index(inplace=True)
    # Normalize
    vmin, vmax = res.min().min(), res.max().max()
    res = (res - vmin) / (vmax - vmin)

    # Return results
    return res


# def generate_costs(nsamples, nitems, noise_scale=0,
#                    seed=None, sampling_seed=None,
#                    nsamples_per_point=1,
#                    noise_type='normal', noise_scale_type='absolute'):
#     assert(noise_scale >= 0)
#     assert(nsamples_per_point > 0)
#     assert(noise_type in ('normal', 'rayleigh'))
#     assert(noise_scale_type in ('absolute', 'relative'))
#     # Seed the RNG
#     np.random.seed(seed)
#     # Generate costs
#     speed = np.random.choice([-15, -12, 12, 15], size=nitems)
#     scale = 1.3 + 1.3 * np.random.rand(nitems)
#     base = 1 * np.random.rand(nitems) / scale
#     offset = -0.7 + 0.6 * np.random.rand(nitems)

#     # Generate input
#     if sampling_seed is not None:
#         np.random.seed(sampling_seed)
#     x = np.random.rand(nsamples)
#     x = np.repeat(x, nsamples_per_point)

#     # Prepare a result dataset
#     res = pd.DataFrame(data=x, columns=['x'])

#     # scale = np.sort(scale)[::-1]
#     for i in range(nitems):
#         # Compute base cost
#         cost = scale[i] / (1 + np.exp(-speed[i] * (x + offset[i])))
#         # Rebase
#         cost = cost - np.min(cost) + base[i]
#         # sx = direction[i]*speed[i]*(x+offset[i])
#         # cost = base[i] + scale[i] / (1 + np.exp(sx))
#         res[f'C{i}'] = cost
#     # Add noise
#     if noise_scale > 0:
#         for i in range(nitems):
#             # Define the noise scale
#             if noise_scale_type == 'absolute':
#                 noise_scale_vals = noise_scale * res[f'C{i}']**0
#             elif noise_scale_type == 'relative':
#                 noise_scale_vals = noise_scale * res[f'C{i}']
#             # Define the noise distribution
#             if noise_type == 'normal':
#                 noise_dist = stats.norm(scale=noise_scale_vals)
#             elif noise_type == 'rayleigh':
#                 # noise_dist = stats.expon(scale=noise_scale_vals)
#                 noise_dist = stats.rayleigh(scale=noise_scale_vals)
#             r_mean = noise_dist.mean()
#             # pnoise = noise * np.random.randn(nsamples)
#             # res[f'C{i}'] = res[f'C{i}'] + pnoise
#             pnoise = noise_dist.rvs()
#             res[f'C{i}'] = res[f'C{i}'] + pnoise - r_mean
#     # Reindex
#     res.set_index('x', inplace=True)
#     # Sort by index
#     res.sort_index(inplace=True)
#     # Normalize
#     vmin, vmax = res.min().min(), res.max().max()
#     res = (res - vmin) / (vmax - vmin)

#     # Return results
#     return res


def generate_problem(nitems, rel_req, seed=None, surrogate=False):
    # Seed the RNG
    np.random.seed(seed)
    # Generate the item values
    values = 1 + 0.4*np.random.rand(nitems)
    # Generate the requirement
    req = rel_req * np.sum(values)
    # Return the results
    if not surrogate:
        return ProductionProblem(values, req)
    else:
        return ProductionProblemSurrogate(values, req)


# def generate_problem(nitems, rel_req, seed=None):
#     # Seed the RNG
#     np.random.seed(seed)
#     # Generate the item values
#     # values = 0.4 + 0.3*np.random.rand(nitems)
#     values = 1.0 + 0.6*np.random.rand(nitems)
#     # Generate the requirement
#     req = rel_req * np.sum(values)
#     # Return the results
#     return KnapsackProblem(values, req)


class ProductionProblem(object):
    def __init__(self, values, requirement):
        """TODO: to be defined. """
        # Store the problem configuration
        self.values = values
        self.requirement = requirement

    def solve(self, costs, tlim=None, print_solution=False):
        # Quick access to some useful fields
        values = self.values
        req = self.requirement
        nv = len(values)
        # Build the solver
        slv = pywraplp.Solver.CreateSolver('CBC')
        # Build the variables
        x = [slv.IntVar(0, 1, f'x_{i}') for i in range(nv)]
        # Build the requirement constraint
        rcst = slv.Add(sum(values[i] * x[i] for i in range(nv)) >= req)
        # Build the objective
        slv.Minimize(sum(costs[i] * x[i] for i in range(nv)))

        # Set a time limit, if requested
        if tlim is not None:
            slv.SetTimeLimit(1000 * tlim)
        # Solve the problem
        status = slv.Solve()
        # Prepare the results
        if status in (slv.OPTIMAL, slv.FEASIBLE):
            res = []
            # Extract the solution
            sol = [x[i].solution_value() for i in range(nv)]
            res.append(sol)
            # Determine whether the problem was closed
            if status == slv.OPTIMAL:
                res.append(True)
            else:
                res.append(False)
            # Attach the computed cost
            res.append(slv.Objective().Value())
            # Attach the solution time
            res.append(slv.wall_time()/1000.0)
        else:
            # TODO I am not handling the unbounded case
            # It should never arise in the first place
            if status == slv.INFEASIBLE:
                res = [None, True, None, slv.wall_time()/1000.0]
            else:
                res = [None, False, None, slv.wall_time()/1000.0]

        # Print the solution, if requested
        if print_solution:
            self._print_sol(res[0], res[1], costs)
        return res

    def _print_sol(self, sol, closed, costs):
        # Obtain indexes of selected items
        idx = [i for i in range(len(sol)) if sol[i] > 0]
        # Print selected items with values and costs
        s = ', '.join(f'{i}' for i in idx)
        print('Selected items:', s)
        s = f'Cost: {sum(costs):.2f}, '
        s += f'Value: {sum(self.values):.2f}, '
        s += f'Requirement: {self.requirement:.2f}, '
        s += f'Closed: {closed}'
        print(s)

    def __repr__(self):
        return f'ProductionProblem(values={self.values}, requirement={self.requirement})'


# class KnapsackProblem(object):
#     def __init__(self, weights, capacity):
#         """TODO: to be defined. """
#         # Store the problem configuration
#         self.weights = weights
#         self.capacity = capacity

#     def solve(self, costs, tlim=None, print_solution=False):
#         # Quick access to some useful fields
#         weights = self.weights
#         cap = self.capacity
#         nv = len(weights)
#         # Build the solver
#         slv = pywraplp.Solver.CreateSolver('CBC')
#         # Build the variables
#         x = [slv.IntVar(0, 1, f'x_{i}') for i in range(nv)]
#         # Build the capacity constraint
#         rcst = slv.Add(sum(weights[i] * x[i] for i in range(nv)) <= cap)
#         # Build the objective
#         slv.Minimize(sum(costs[i] * x[i] for i in range(nv)))

#         # Set a time limit, if requested
#         if tlim is not None:
#             slv.SetTimeLimit(1000 * tlim)
#         # Solve the problem
#         status = slv.Solve()
#         # Prepare the results
#         if status in (slv.OPTIMAL, slv.FEASIBLE):
#             res = []
#             # Extract the solution
#             sol = [x[i].solution_value() for i in range(nv)]
#             res.append(sol)
#             # Determine whether the problem was closed
#             if status == slv.OPTIMAL:
#                 res.append(True)
#             else:
#                 res.append(False)
#             # Attach the computed cost
#             res.append(slv.Objective().Value())
#             # Attach the solution time
#             res.append(slv.wall_time()/1000.0)
#         else:
#             # TODO I am not handling the unbounded case
#             # It should never arise in the first place
#             if status == slv.INFEASIBLE:
#                 res = [None, True, None, slv.wall_time()/1000.0]
#             else:
#                 res = [None, False, None, slv.wall_time()/1000.0]

#         # Print the solution, if requested
#         if print_solution:
#             self._print_sol(res[0], res[1], costs)
#         return res

#     def _print_sol(self, sol, closed, costs):
#         # Obtain indexes of selected items
#         idx = [i for i in range(len(sol)) if sol[i] > 0]
#         # Print selected items with weights and costs
#         s = ', '.join(f'{i}' for i in idx)
#         print('Selected items:', s)
#         s = f'Cost: {sum(costs):.2f}, '
#         s += f'Value: {sum(self.weights):.2f}, '
#         s += f'Requirement: {self.capacity:.2f}, '
#         s += f'Closed: {closed}'
#         print(s)

#     def __repr__(self):
#         wgt = [f'{w:.2f}' for w in self.weights]
#         return f'KnapsackProblem(weights={[", ".join(wgt)]}, capacity={self.capacity:.2f})'


def plot_df_cols(data, scatter=False, figsize=None, legend=True, title=None):
    # Build figure
    fig = plt.figure(figsize=figsize)
    # Setup x axis
    x = data.index
    plt.xlabel(data.index.name)
    # Plot all columns
    for cname in data.columns:
        y = data[cname]
        plt.plot(x, y, label=cname,
                 linestyle='-' if not scatter else '',
                 marker=None if not scatter else '.',
                 alpha=1 if not scatter else 0.3)
    # Add legend
    if legend and len(data.columns) <= 10:
        plt.legend(loc='best')
    plt.grid(':')
    # Add a title
    plt.title(title)
    # Make it compact
    plt.tight_layout()
    # Show
    plt.show()


def get_ml_metrics(model, X, y):
    # Obtain the predictions
    pred = model.predict(X, verbose=0)
    # Compute the root MSE
    rmse = np.sqrt(metrics.mean_squared_error(y, pred))
    # Compute the MAE
    mae = metrics.mean_absolute_error(y, pred)
    # Compute the coefficient of determination
    r2 = metrics.r2_score(y, pred)
    return r2, mae, rmse


def print_ml_metrics(model, X, y, label=''):
    r2, mae, rmse = get_ml_metrics(model, X, y)
    if len(label) > 0:
        label = f' ({label})'
    print(f'R2: {r2:.2f}, MAE: {mae:.2}, RMSE: {rmse:.2f}{label}')


def compute_regret(problem, predictor, pred_in, true_costs, tlim=None):
    # Obtain all predictions
    costs = predictor.predict(pred_in, verbose=0)
    # Compute all solutions
    sols = []
    for c in costs:
        sol, _, _, _  = problem.solve(c, tlim=tlim)
        sols.append(sol)
    sols = np.array(sols)
    # Compute the true solutions
    tsols = []
    for c in true_costs:
        sol, _, _, _ = problem.solve(c, tlim=tlim)
        tsols.append(sol)
    tsols = np.array(tsols)
    # Compute true costs
    costs_with_predictions = np.sum(true_costs * sols, axis=1)
    costs_with_true_solutions = np.sum(true_costs * tsols, axis=1)
    # Compute regret
    regret = (costs_with_predictions - costs_with_true_solutions) / np.abs(costs_with_true_solutions)
    # Return true costs
    return regret


def plot_histogram(data, label=None, bins=20, figsize=None,
        data2=None, label2=None, print_mean=False, title=None):
    # Build figure
    fig = plt.figure(figsize=figsize)
    # Setup x axis
    if data2 is None:
        plt.xlabel(label)
    # Define bins
    rmin, rmax = data.min(), data.max()
    if data2 is not None:
        rmin = min(rmin, data2.min())
        rmax = max(rmax, data2.max())
    bins = np.linspace(rmin, rmax, bins)
    # Histogram
    hist, edges = np.histogram(data, bins=bins)
    hist = hist / np.sum(hist)
    plt.step(edges[:-1], hist, where='post', label=label)
    if data2 is not None:
        hist2, edges2 = np.histogram(data2, bins=bins)
        hist2 = hist2 / np.sum(hist2)
        plt.step(edges2[:-1], hist2, where='post', label=label2)
    plt.grid(':')
    # Add a title
    plt.title(title)
    # Make it compact
    plt.tight_layout()
    # Legend
    plt.legend()
    # Show
    plt.show()
    # Print mean, if requested
    if print_mean:
        s = f'Mean: {np.mean(data):.3f}'
        if label is not None:
            s += f' ({label})'
        if data2 is not None:
            s += f', {np.mean(data2):.3f}'
            if label2 is not None:
                s += f' ({label2})'
        print(s)


def build_dfl_ml_model(input_size, output_size,
        problem, tlim=None, hidden=[], recompute_chance=1,
        output_activation='linear', loss_type='scr',
        sfge=False, sfge_sigma_init=1, sfge_sigma_trainable=False,
        surrogate=False, standardize_loss=False, name=None):
    assert(not sfge or recompute_chance==1)
    assert(not sfge or loss_type in ('cost', 'regret', 'scr'))
    assert(sfge or loss_type in ('scr', 'spo', 'sc'))
    assert(sfge_sigma_init > 0)
    assert(not surrogate or (sfge and loss_type == 'cost'))
    # Build all layers
    nnin = keras.Input(input_size)
    nnout = nnin
    for h in hidden:
        nnout = layers.Dense(h, activation='relu')(nnout)
    if output_activation != 'linear_normalized' or output_size == 1:
        nnout = layers.Dense(output_size, activation=output_activation)(nnout)
    else:
        h1_out = layers.Dense(output_size-1, activation='linear')(nnout)
        h2_out = tf.reshape(1 - tf.reduce_sum(h1_out, axis=1), (-1, 1))
        nnout = tf.concat([h1_out, h2_out], axis=1)
    # Build the model
    if not sfge:
        model = DFLModel(problem, tlim=tlim, recompute_chance=recompute_chance,
                    inputs=nnin, outputs=nnout, loss_type=loss_type,
                    name=name)
    else:
        model = SFGEModel(problem, tlim=tlim,
                    inputs=nnin, outputs=nnout, loss_type=loss_type,
                    sigma_init=sfge_sigma_init, sigma_trainable=sfge_sigma_trainable,
                    surrogate=surrogate, standardize_loss=standardize_loss, name=name)
    return model


class DFLModel(keras.Model):
    def __init__(self, prb, tlim=None, recompute_chance=1,
                 loss_type='spo',
                 **params):
        super(DFLModel, self).__init__(**params)
        assert(loss_type in ('scr', 'spo', 'sc'))
        # Store configuration parameters
        self.prb = prb
        self.tlim = tlim
        self.recompute_chance = recompute_chance
        self.loss_type = loss_type

        # Build metrics
        self.metric_loss = keras.metrics.Mean(name="loss")
        # self.metric_regret = keras.metrics.Mean(name="regret")
        # self.metric_mae = keras.metrics.MeanAbsoluteError(name="mae")
        # Prepare a field for the solutions
        self.sol_store = None
        self.tsol_index = None


    def fit(self, X, y, **kwargs):
        # Precompute all solutions for the true costs
        self.sol_store = []
        self.tsol_index = {}
        for x, c in zip(X.astype('float32'), y):
            sol, closed, obj, _ = self.prb.solve(c, tlim=self.tlim)
            self.sol_store.append(sol)
            self.tsol_index[x] = len(self.sol_store)-1
        self.sol_store = np.array(self.sol_store)
        # Call the normal fit method
        return super(DFLModel, self).fit(X, y, **kwargs)

    def _find_best(self, costs):
        tc = np.dot(self.sol_store, costs)
        best_idx = np.argmin(tc)
        best = self.sol_store[best_idx]
        return best

    def train_step(self, data):
        # Unpack the data
        x, costs_true = data
        # Quick access to some useful fields
        prb = self.prb
        tlim = self.tlim

        # Loss computation
        with tf.GradientTape() as tape:
            # Obtain the predictions
            costs = self(x, training=True)
            # Define the costs to be used for computing the solutions
            costs_iter = costs
            # Compute SPO costs, if needed
            if self.loss_type == 'spo':
                spo_costs = 2*costs - costs_true
                costs_iter = spo_costs
            # Solve all optimization problems
            sols, tsols = [], []
            for xv, c, tc in zip(x.numpy(), costs_iter.numpy(), costs_true.numpy()):
                # Decide whether to recompute the solution
                if np.random.rand() < self.recompute_chance:
                    sol, closed, _, _ = prb.solve(c, tlim=self.tlim)
                    # Store the solution, if needed
                    if self.recompute_chance < 1:
                        # Check if the solutions is already stored
                        if not (self.sol_store == sol).all(axis=1).any():
                            self.sol_store = np.vstack((self.sol_store, sol))
                else:
                    sol = self._find_best(c)
                # Find the best solution with the predicted costs
                sols.append(sol)
                # Find the best solution with the true costs
                # tsol = self._find_best(tc)
                tsol = self.sol_store[self.tsol_index[xv]]
                tsols.append(tsol)
            # Convert solutions to numpy arrays
            sols = np.array(sols)
            tsols = np.array(tsols)
            # Compute the loss
            if self.loss_type == 'scr':
                # Compute the cost difference
                cdiff = costs - costs_true
                # Compute the solution difference
                sdiff = tsols - sols
                # Compute the loss terms
                loss_terms = tf.reduce_sum(cdiff * sdiff, axis=1)
            elif self.loss_type == 'spo':
                loss_terms = tf.reduce_sum(spo_costs * (tsols - sols), axis=1)
            elif self.loss_type == 'sc':
                loss_terms = tf.reduce_sum(costs * (tsols - sols), axis=1)
            # Compute the mean loss
            loss = tf.reduce_mean(loss_terms)

        # Perform a gradient descent step
        tr_vars = self.trainable_variables
        gradients = tape.gradient(loss, tr_vars)
        self.optimizer.apply_gradients(zip(gradients, tr_vars))

        # Update main metrics
        self.metric_loss.update_state(loss)
        # Update compiled metrics
        self.compiled_metrics.update_state(costs_true, costs)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    @property
    def metrics(self):
        return [self.metric_loss]


def train_dfl_model(model, X, y, tlim=None,
        epochs=20, verbose=0, patience=10, batch_size=32,
        validation_split=0.2, optimizer='Adam',
        save_weights=False,
        load_weights=False,
        warm_start_pfl=None,
        **params):
    # Try to load the weights, if requested
    if load_weights:
        history = load_ml_model_weights(model, model.name)
        return history
    # Attempt a warm start
    if warm_start_pfl is not None:
        transfer_weights_to_dfl_model(warm_start_pfl, model)
    # Compile and train
    model.compile(optimizer=optimizer, run_eagerly=True)
    if validation_split > 0:
        cb = [callbacks.EarlyStopping(patience=patience,
            restore_best_weights=True)]
    else:
        cb = None
    # Start training
    history = model.fit(X, y, validation_split=validation_split,
                     callbacks=cb, batch_size=batch_size,
                     epochs=epochs, verbose=verbose, **params)
    # Save the model, if requested
    if save_weights:
        save_ml_model_weights_and_history(model, model.name, history)
    return history


def transfer_weights_to_dfl_model(source, dest):
    source_weights = source.get_weights()
    dest_weights = dest.get_weights()
    transfer_weights = source_weights[:len(source_weights)] + dest_weights[len(source_weights):]
    dest.set_weights(transfer_weights)


def load_taxi_series(file_name, data_folder):
    # Load the input data
    data_path = os.path.join(data_folder, 'data', file_name)
    data = pd.read_csv(data_path)
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data.set_index('timestamp', inplace=True)
    # Load the labels
    label_path = os.path.join(data_folder, 'labels', 'combined_labels.json')
    with open(label_path) as fp:
        labels = pd.Series(json.load(fp)[file_name])
    labels = pd.to_datetime(labels)
    # Load the windows
    window_path = os.path.join(data_folder, 'labels', 'combined_windows.json')
    window_cols = ['begin', 'end']
    with open(window_path) as fp:
        windows = pd.DataFrame(columns=window_cols,
                data=json.load(fp)[file_name])
    windows['begin'] = pd.to_datetime(windows['begin'])
    windows['end'] = pd.to_datetime(windows['end'])
    # Return data
    return data, labels, windows


def plot_histogram2d(xdata, ydata, bins=10, figsize=None):
    # Build a new figure
    plt.close('all')
    plt.figure(figsize=figsize)
    # Plot a histogram
    plt.hist2d(xdata, ydata, density=True, bins=bins)
    plt.tight_layout()


def get_anomaly_metrics(pred, labels, windows):
    tp = [] # True positives
    fp = [] # False positives
    fn = [] # False negatives
    advance = [] # Time advance, for true positives
    # Loop over all windows
    used_pred = set()
    for idx, w in windows.iterrows():
        # Search for the earliest prediction
        pmin = None
        for p in pred:
            if p >= w['begin'] and p < w['end']:
                used_pred.add(p)
                if pmin is None or p < pmin:
                    pmin = p
        # Compute true pos. (incl. advance) and false neg.
        l = labels[idx]
        if pmin is None:
            fn.append(l)
        else:
            tp.append(l)
            advance.append(l-pmin)
    # Compute false positives
    for p in pred:
        if p not in used_pred:
            fp.append(p)
    # Return all metrics as pandas series
    return pd.Series(tp, dtype='datetime64[ns]'), \
            pd.Series(fp, dtype='datetime64[ns]'), \
            pd.Series(fn, dtype='datetime64[ns]'), \
            pd.Series(advance)


class ADSimpleCostModel:
    def __init__(self, c_alrm, c_missed, c_late):
        self.c_alrm = c_alrm
        self.c_missed = c_missed
        self.c_late = c_late

    def cost(self, signal, labels, windows, thr):
        # Obtain predictions
        pred = get_anomaly_pred(signal, thr)
        # Obtain metrics
        tp, fp, fn, adv = get_anomaly_metrics(pred, labels, windows)
        # Compute the cost
        adv_det = [a for a in adv if a.total_seconds() <= 0]
        cost = self.c_alrm * len(fp) + \
           self.c_missed * len(fn) + \
           self.c_late * (len(adv_det))
        return cost


def get_anomaly_pred(signal, thr):
    return pd.Series(signal.index[signal >= thr])


def opt_thr(signal, labels, windows, cmodel, thr_range):
    costs = [cmodel.cost(signal, labels, windows, thr)
            for thr in thr_range]
    costs = np.array(costs)
    best_idx = np.argmin(costs)
    return thr_range[best_idx], costs[best_idx]
