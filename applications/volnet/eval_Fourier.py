"""
Script to evaluate the network quality if it is trained for densities or for colors.
They are tested with world-space training and the best configuration from eval_network_configs
"""

import sys
import os
sys.path.insert(0, os.getcwd())

import numpy as np
import sys
import os
import shutil
import subprocess
import itertools
import imageio
import json
import torch
import io
import matplotlib.pyplot as plt
import matplotlib.ticker
from collections import defaultdict

BEST_ACTIVATION = "SnakeAlt:1"

BASE_PATH = 'volnet/results/eval_Fourier'

configX = [
    ("plume100", "config-files/plume100-v2-dvr.json"),
    ("ejecta70", "config-files/ejecta70-v6-dvr.json"),
    ("RM20", "config-files/RichtmyerMeshkov-t20-v1-dvr.json"),
    ("RM60", "config-files/RichtmyerMeshkov-t60-v1-dvr.json"),
]
networkX = [
    ("l48x10", 48, 10),
    ("l64x6", 64, 6)
]
fourierX = [
    ("f%03d"%int(f*10), f) for f in [0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
] + [("fNeRF", -1)]
#fourierX = [("fNeRF", -1)]

def main():
    configs = collect_configurations()
    #train(configs)
    statistics_file = eval(configs)
    make_plots(statistics_file)

def collect_configurations():
    cfgs = []
    for config, network, fourier in itertools.product(configX, networkX, fourierX):
        filename = "fourier-world-%s-%s-%s" % (
            config[0], network[0], fourier[0])
        cfgs.append((config[1], network[1:], fourier[1], filename))
    return cfgs

def get_args_and_hdf5_file(cfg):
    """
    Assembles the command line arguments for training and the filename for the hdf5-file
    with the results
    :return: args, filename
    """

    common_parameters = [
        "--train:mode", "world",
        "--train:samples", "256**3",
        "--train:batchsize", "64*64*128",
        "--train:sampler_importance", "0.01",
        "--val:copy_and_split",
        "--outputmode", "density:direct",
        "--lossmode", "density",
        "--activation", BEST_ACTIVATION,
        "-l1", "1",
        "--lr_step", "50",
        "-i", "200",
        "--logdir", 'volnet/results/eval_Fourier/log',
        "--modeldir", 'volnet/results/eval_Fourier/model',
        "--hdf5dir", 'volnet/results/eval_Fourier/hdf5',
    ]
    def getNetworkParameters(network):
        channels, layers = network
        return ["--layers", ':'.join([str(channels)] * (layers - 1))]

    def getFourierParameters(network, fourier):
        channels, layers = network
        std = fourier
        return ['--fouriercount', str((channels - 4) // 2), '--fourierstd', str(std)]

    config, network, fourier, filename = cfg

    launcher = [sys.executable, "volnet/train_volnet.py"]
    args = launcher + [config] + \
           common_parameters + \
           getNetworkParameters(network) + \
           getFourierParameters(network, fourier) + \
           ['--name', filename]

    hdf5_file = os.path.join(BASE_PATH, 'hdf5', filename + ".hdf5")
    return args, hdf5_file, filename

def train(configs):
    print("Configurations:", len(configs))
    for cfg in configs:
        args, filename, outputname = get_args_and_hdf5_file(cfg)
        if os.path.exists(filename):
            print("Skipping test", filename)
        else:
            print("\n=====================================\nRun", filename)
            subprocess.run(args, check=True)
    print("\n===========================================\nDONE!")

def eval(configs):
    print("Evaluate")
    statistics_file = os.path.join(BASE_PATH, 'stats.json')
    if os.path.exists(statistics_file):
        print("Statistics file already exists!")
        return statistics_file

    import common.utils as utils
    import pyrenderer
    from volnet.inference import LoadedModel
    from losses.lossbuilder import LossBuilder

    num_cameras = 64
    width = 512
    height = 512
    STEPSIZE = 1/512
    timer = pyrenderer.GPUTimer()
    rendering_mode = LoadedModel.EvaluationMode.TENSORCORES_MIXED
    #rendering_mode = LoadedModel.EvaluationMode.PYTORCH16
    enable_preintegration = rendering_mode==LoadedModel.EvaluationMode.TENSORCORES_MIXED

    output_stats = []
    device = torch.device('cuda')
    ssim_loss = LossBuilder(device).ssim_loss(4)
    lpips_loss = LossBuilder(device).lpips_loss(4, 0.0, 1.0)

    def compute_stats(ln, mode, reference_images, stepsize, filename_template=None, do_ssim=False, do_lpips=False):
        timingsX = []
        ssimX = []
        lpipsX = []
        for i in range(num_cameras):
            current_image = ln.render_network(
                cameras[i], width, height, mode,
                stepsize, timer=timer)
            if i>0:
                timingsX.append(timer.elapsed_milliseconds())
            if filename_template is not None:
                imageio.imwrite(
                    filename_template%i,
                    LoadedModel.convert_image(current_image))
            if do_ssim:
                ssimX.append(ssim_loss(current_image, reference_images[i]).item())
            if do_lpips:
                lpipsX.append(lpips_loss(current_image, reference_images[i]).item())
        return \
            (np.mean(timingsX), np.std(timingsX)), \
            (np.mean(ssimX), np.std(ssimX)) if do_ssim else (np.NaN, np.NaN), \
            (np.mean(lpipsX), np.std(lpipsX)) if do_lpips else (np.NaN, np.NaN)

    # load networks
    def load_and_save(cfg):
        _, filename, output_name = get_args_and_hdf5_file(cfg)
        ln = LoadedModel(filename)
        ln.enable_preintegration(enable_preintegration)
        ln.save_compiled_network(filename.replace('.hdf5', '.volnet'))
        return ln, output_name

    """
    for config, network, fourier in itertools.product(configX, networkX, fourierX):
        filename = "fourier-world-%s-%s-%s" % (
            config[0], network[0], fourier[0])
        cfgs.append((config[1], network[1:], fourier[1], filename))
    """

    for cfg_index, config in enumerate(configX):
        image_folder = os.path.join(BASE_PATH, "images_"+config[0])
        local_stats = {
            'cfg_index': cfg_index,
            'cfg': config[1]}

        reference_images = None
        # collect models
        lns = dict()
        base_ln = None
        for network, fourier in itertools.product(networkX, fourierX):
            filename = "fourier-world-%s-%s-%s" % (
                config[0], network[0], fourier[0])
            ln, name = load_and_save((config[1], network[1:], fourier[1], filename))
            lns[(network[0], fourier[0])] = (ln, name)
            if base_ln is None: base_ln = ln

        # render reference
        if reference_images is None:
            image_folder_reference = os.path.join(image_folder, "reference")
            os.makedirs(image_folder_reference, exist_ok=True)
            print("\n===================================== Render reference", cfg_index)
            cameras = base_ln.get_rotation_cameras(num_cameras)
            reference_images = [None] * num_cameras
            for i in range(num_cameras):
                reference_images[i] = base_ln.render_reference(cameras[i], width, height)
                imageio.imwrite(
                    os.path.join(image_folder_reference, 'reference%03d.png' % i),
                    LoadedModel.convert_image(reference_images[i]))

        # render networks
        for network, fourier in itertools.product(networkX, fourierX):
            ln, name = lns[(network[0], fourier[0])]
            image_folder_screen = os.path.join(image_folder, "%s" % name)
            os.makedirs(image_folder_screen, exist_ok=True)
            time, ssim, lpips = compute_stats(
                ln, rendering_mode, reference_images, STEPSIZE,
                os.path.join(image_folder_screen, 'img%03d.png'),
                True, True)
            local_stats[name] = {
                'time': time,
                'ssim': ssim,
                'lpips': lpips,
            }

        output_stats.append(local_stats)

    # save statistics
    print("\n===================================== Done, save statistics")
    with open(statistics_file, "w") as f:
        json.dump(output_stats, f)
    return statistics_file

def make_plots(statistics_file):
    print("\n===================================== Make Plots")
    with open(statistics_file, "r") as f:
        stats = json.load(f)
    output_folder = os.path.split(statistics_file)[0]
    FILETYPE = "eps"

    numRows = len(configX)
    statNames = ['SSIM $\\uparrow$', 'LPIPS $\\downarrow$']
    statTags = ["ssim", "lpips"]
    numCols = len(statTags)

    fig, axs = plt.subplots(numRows, numCols, squeeze=False, sharex=True, figsize=(7, 1 + 2 * numRows))
    legend_handles = []
    legend_names = []
    for row in range(numRows):
        local_stat = stats[row]
        axs[row, 0].set_ylabel(configX[row][0])
        for col, (name, tag) in enumerate(zip(statNames, statTags)):
            ax = axs[row,col]
            if row==0:
                ax.set_title(name)
            X, Xlabel = None, None
            for network_label, network_channels, network_layers in networkX:
                X = []
                Xlabel = []
                Y = []
                err = []
                for i,(fn,f) in enumerate(fourierX[:-1]):
                    filename = "fourier-world-%s-%s-%s" % (
                        configX[row][0], network_label, fn)
                    y,e = local_stat[filename][tag]
                    X.append(i)
                    Xlabel.append("%.1f"%f)
                    Y.append(y)
                    err.append(e)
                h = ax.errorbar(X, Y, yerr=err)
                # extra NeRF-fourier
                filename = "fourier-world-%s-%s-%s" % (
                    configX[row][0], network_label, fourierX[-1][0])
                y, e = local_stat[filename][tag]
                ax.errorbar([X[-1]+1.5], [y], yerr=[e], color=h[0].get_color(), fmt='.')
                X.append(X[-1]+1.5)
                Xlabel.append("NeRF")
                # legend
                if row==0 and col==0:
                    legend_handles.append(h)
                    legend_names.append(f"{network_channels} channels, {network_layers} layers")
            ax.set_xticks(X)
            ax.set_xticklabels(Xlabel)
            ax.set_xlabel("Fourier std $\sigma^2$")

        # determine and copy best and worst images
        tag = "lpips"
        worst_lpips = 0
        worst_filename = None
        best_lpips = 1
        best_filename = None
        for network_label, network_channels, network_layers in networkX:
              for i, (fn, f) in enumerate(fourierX[:-1]):
                filename = "fourier-world-%s-%s-%s" % (
                    configX[row][0], network_label, fn)
                y, _ = local_stat[filename][tag]
                if y < best_lpips:
                    best_lpips = y
                    best_filename = filename
                if y > worst_lpips:
                    worst_lpips = y
                    worst_filename = filename

        shutil.copyfile(
            os.path.join(output_folder, "images_%s/reference/reference000.png" % (configX[row][0])),
            os.path.join(output_folder, "%s_reference.png" % configX[row][0]))
        shutil.copyfile(
            os.path.join(output_folder, "images_%s/%s/img000.png"%(configX[row][0], best_filename)),
            os.path.join(output_folder, "%s_best.png"%configX[row][0]))
        shutil.copyfile(
            os.path.join(output_folder, "images_%s/%s/img000.png" % (configX[row][0], worst_filename)),
            os.path.join(output_folder, "%s_worst.png" % configX[row][0]))

    lgd = fig.legend(
        legend_handles, legend_names,
        #bbox_to_anchor=(0.75, 0.7), loc='lower center', borderaxespad=0.
        loc='upper center', bbox_to_anchor=(0.5, 0.05),
        ncol=len(legend_handles))
    fig.savefig(os.path.join(output_folder, 'Fourier-SSIM.%s'%FILETYPE),
                bbox_inches='tight', bbox_extra_artists=(lgd,))

    print("Done")
    plt.show()


if __name__ == '__main__':
    main()