#!/usr/bin/env python
import argparse
import glob
import itertools
import seaborn as sns
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re

FONTSIZE = 35
# font = {'size': FONTSIZE}
# matplotlib.rc('font', **font)
sns.set(font_scale = 1.5); sns.set_palette(sns.color_palette("Set2")); sns.set_style("whitegrid")

def parse_results(file_content, sp_idx, method_name):
    results = []
    lines = file_content.split('\n')
    timing = float(lines[0])
    all_bounds = lines[1].split('\t')
    all_bounds = list(map(float, all_bounds))

    for neur_idx, bound in enumerate(all_bounds):
        results.append({
            "Neuron index": neur_idx,
            "Sample": sp_idx,
            "Value": bound,
            "Method": method_name,
            "Time": timing
        })
    return results

def do_bound_versus_time_graph(datapoints, filename):
    datapoints = datapoints.reset_index(level=1)

    fig = plt.figure(figsize=(10,10))
    ax_value = plt.subplot(1, 1, 1)
    ax_value.set_xscale("log")
    sns.scatterplot(x="Time", y="Gap to best bound", data=datapoints,
                    hue="Method", legend='brief', alpha=0.2)
    leg = ax_value.legend()
    for lh in leg.legendHandles:
        lh.set_alpha(1)
        lh.set_linewidth(5)

    target_format = "eps" if filename.endswith('.eps') else "png"
    plt.tight_layout()
    plt.savefig(filename, format=target_format, dpi=50)


def do_violin_plots(datapoints, filename, labels):
    datapoints = datapoints.reset_index(level=1)

    fig, (time_ax, bound_ax) = plt.subplots(2, 1, sharex='col',
                                            figsize=(20, 7))
    sns.boxplot(x="Method", y="Time", data=datapoints, ax=time_ax,
                order=labels)
    time_ax.set(yscale="log")

    sns.violinplot(x="Method", y="Gap to best bound", data=datapoints,
                   inner=None, width=1, scale='area', ax=bound_ax,
                   order=labels, cut=0)
    bound_ax.set(yscale="log")
    bound_ax.xaxis.set_label_text("")

    target_format = "eps" if filename.endswith('.eps') else "png"
    plt.tight_layout()
    plt.savefig(filename, format=target_format, dpi=50)
    plt.savefig(filename.replace(target_format, 'eps'), pad_inches=0)



def main():
    parser = argparse.ArgumentParser(description="Compare the resuts between differents bounds computation")
    parser.add_argument('results_folder', type=str,
                        help="Folder where you have the results files")
    parser.add_argument('output_image_prefix', type=str,
                        help="Where to dump the resulting image.")
    parser.add_argument('to_load', type=str,
                        help="Comma separated list of the bounds to load for each samples.")
    parser.add_argument('names', type=str,
                        help="Comma separated list of legends for each bound")
    args = parser.parse_args()

    files_to_load = args.to_load.split(',')
    legend_names = args.names.replace('\\n', '\n').split(',')

    all_results = []
    for sp_idx in sorted(os.listdir(args.results_folder)):
        sp_results = []
        for method_name, filename in zip(legend_names, files_to_load):
            file_to_load = os.path.join(args.results_folder, sp_idx, filename)
            if os.path.exists(file_to_load):
                with open(file_to_load, 'r') as res_file:
                    sp_results.extend(parse_results(res_file.read(),
                                                    int(sp_idx),
                                                    method_name))
            else:
                break
        else:
            all_results.extend(sp_results)

    all_results = pd.DataFrame(all_results)
    all_results["Optimization Problem"] = all_results["Sample"].astype(str).str.cat(
        all_results["Neuron index"].astype(str), sep=' - ')
    all_results.drop(columns=["Neuron index", "Sample"], inplace=True)
    all_results.set_index(["Optimization Problem", "Method"], inplace=True)

    best_bound = all_results["Value"].groupby(["Optimization Problem"]).min()
    all_results["Gap to best bound"] = all_results["Value"] - best_bound

    # Let's check that Gurobi is always the best bound
    gurobi_worst = all_results.query("Method == 'Gurobi'")["Gap to best bound"].max()
    print(f"Gurobi worst gap is: {gurobi_worst}")

    do_bound_versus_time_graph(all_results,
                               args.output_image_prefix + "_bound_vs_time.png")
    do_violin_plots(all_results,
                    args.output_image_prefix + "_violins.png",
                    labels=legend_names)

    # do_pairwise_plot(all_results, args.output_image_prefix)

    do_selected_pairwise_hex(all_results, args.output_image_prefix)


def do_pairwise_plot(datapoints, filename_prefix):

    sns.set(font_scale=2.5)
    sns.set_palette(sns.color_palette("Set2"))
    sns.set_style("whitegrid")

    datapoints = datapoints.reset_index(level=1)

    all_methods = datapoints.Method.unique()
    target_dir = filename_prefix + "_pairwise_comparisons"
    os.makedirs(target_dir, exist_ok=True)
    all_pairs = itertools.permutations(all_methods, 2)
    for meth1, meth2 in all_pairs:

        filename = "{meth1}_vs_{meth2}.png".format(meth1=meth1, meth2=meth2)
        filename = filename.replace("\n", '_')
        filename = os.path.join(target_dir, filename)
        target_format = "pdf" if filename.endswith(".pdf") else "png"
        meth1_data = datapoints[datapoints.Method == meth1]
        meth2_data = datapoints[datapoints.Method == meth2]

        fig, (time_ax, bound_ax) = plt.subplots(1, 2, figsize=(20, 10))
        sns.scatterplot(x=meth1_data.Time, y=meth2_data.Time, ax=time_ax, s=64)
        time_ax.set_xscale("log")
        time_ax.set_yscale("log")
        time_ax.xaxis.set_label_text(meth1.replace('\n', ' '))
        time_ax.yaxis.set_label_text(meth2.replace('\n', ' '))
        time_ax.set_title("Timing (in s)")
        xlim = time_ax.get_xlim()
        ylim = time_ax.get_ylim()
        rng = (min(xlim[0], ylim[0]), max(xlim[1], ylim[1]))
        time_ax.plot(rng, rng, ls="--", c=".3")


        to_plot = "Gap to best bound"
        # to_plot = "Value"
        sns.scatterplot(x=meth1_data[to_plot], y=meth2_data[to_plot], ax=bound_ax, s=64)
        bound_ax.set_xscale("log")
        bound_ax.set_yscale("log")
        bound_ax.set_xlim(1e-3, 1e0)
        bound_ax.set_ylim(1e-3, 1e0)
        bound_ax.xaxis.set_label_text(meth1.replace('\n', ' '))
        bound_ax.yaxis.set_label_text(meth2.replace('\n', ' '))
        bound_ax.set_title(to_plot)
        xlim = bound_ax.get_xlim()
        ylim = bound_ax.get_ylim()
        rng = (min(xlim[0], ylim[0]), max(xlim[1], ylim[1]))
        bound_ax.plot(rng, rng, ls="--", c=".3")

        plt.tight_layout()
        plt.savefig(filename, format=target_format, dpi=100)

        plt.close(fig)


def do_selected_pairwise_hex(datapoints, filename_prefix):

    sns.set(font_scale=2)
    sns.set_palette(sns.color_palette("Set2"))
    import matplotlib.ticker as ticker
    sns.set_style("ticks")

    datapoints = datapoints.reset_index(level=1)

    all_methods = ["DSG+\n1040 steps", "Supergradient\n640 steps", "Proximal\n400 steps"]

    bnd_extent = (-2.3, -0.7, -2.3, -0.7) if filename_prefix == "../results/temp/madry8" else (-1.8, -0.4, -1.8, -0.4)

    target_dir = filename_prefix + "_pairwise_comparisons"
    os.makedirs(target_dir, exist_ok=True)
    all_pairs = itertools.permutations(all_methods, 2)
    for meth1, meth2 in all_pairs:

        filename = "{meth1}_vs_{meth2}_hex.pdf".format(meth1=meth1, meth2=meth2)
        filename = filename.replace("\n", '_')
        filename = os.path.join(target_dir, filename)
        target_format = "pdf" if filename.endswith(".pdf") else "png"
        meth1_data = datapoints[datapoints.Method == meth1]
        meth2_data = datapoints[datapoints.Method == meth2]

        fig, (time_ax, bound_ax) = plt.subplots(1, 2, figsize=(20, 10))
        time_ax.hexbin(x=meth1_data.Time, y=meth2_data.Time, cmap="Greens", gridsize=30, mincnt=1, xscale='log',
                       yscale='log', bins='log', extent=(0.25, 0.7, 0.25, 0.7))
        time_ax.xaxis.set_label_text(meth1.replace('\n', ' '))
        time_ax.yaxis.set_label_text(meth2.replace('\n', ' '))
        time_ax.set_title("Timing (in s)")
        xlim = time_ax.get_xlim()
        ylim = time_ax.get_ylim()
        rng = (min(xlim[0], ylim[0]), max(xlim[1], ylim[1]))
        time_ax.plot(rng, rng, ls="--", c=".3")
        time_ax.grid(True, which='major', axis='both')
        for axis in [time_ax.xaxis, time_ax.yaxis]:
            axis.set_major_formatter(ticker.LogFormatter())

        to_plot = "Gap to best bound"
        bound_ax.hexbin(x=meth1_data[to_plot], y=meth2_data[to_plot], gridsize=30, mincnt=1, cmap="Greens",
                        xscale='log', yscale='log', bins='log', extent=bnd_extent)
        bound_ax.xaxis.set_label_text(meth1.replace('\n', ' '))
        bound_ax.yaxis.set_label_text(meth2.replace('\n', ' '))
        bound_ax.set_title(to_plot)
        xlim = bound_ax.get_xlim()
        ylim = bound_ax.get_ylim()
        rng = (min(xlim[0], ylim[0]), max(xlim[1], ylim[1]))
        bound_ax.plot(rng, rng, ls="--", c=".3")

        plt.tight_layout()
        plt.savefig(filename, format=target_format, dpi=100)

        plt.close(fig)


if __name__ == '__main__':
    main()
