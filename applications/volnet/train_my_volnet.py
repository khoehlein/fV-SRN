"""
Neural network for 3D scene representation.
"""

import sys
import os

from volnet.storage_manager import StorageManager
from volnet.visualizer import Visualizer

sys.path.insert(0, os.getcwd())

import numpy as np
import torch
import torch.nn.functional as F
import os
import tqdm
import time
import h5py
import argparse
from contextlib import ExitStack
from collections import defaultdict

import common.utils as utils

from volnet.network import SceneRepresentationNetwork
from volnet.lossnet import LossFactory
from volnet.input_data import TrainingInputData
from volnet.training_data import TrainingData
from volnet.optimizer import Optimizer
from volnet.evaluation import EvaluateWorld, EvaluateScreen


def build_parser():
    this_folder = os.path.split(__file__)[0]
    parser = argparse.ArgumentParser(
        description='Scene representation networks',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    TrainingInputData.init_parser(parser)
    TrainingData.init_parser(parser)
    SceneRepresentationNetwork.init_parser(parser)
    LossFactory.init_parser(parser)
    Optimizer.init_parser(parser)
    StorageManager.init_parser(parser, this_folder)
    parser.add_argument('--seed', type=int, default=124, help='random seed to use. Default=124')
    return parser


def apply_seeding(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


def exit_upon_direction_data_error():
    print(
        "ERROR: The network requires the direction as input, but world-space training or validation was requested.")
    print(" Directions are only available for pure screen-space training")
    exit(-1)


def build_evaluation_helpers(device, dtype, image_evaluator, loss_screen, loss_world, network, training_data):
    if training_data.training_mode() == 'world':
        evaluator_train = EvaluateWorld(
            network, image_evaluator, loss_world, dtype, device)
    else:
        evaluator_train = EvaluateScreen(
            network, image_evaluator, loss_screen,
            training_data.training_image_size(), training_data.training_image_size(),
            True, training_data.train_disable_inversion_trick(), dtype, device)
    if training_data.validation_mode() == 'world':
        evaluator_val = EvaluateWorld(
            network, image_evaluator, loss_world, dtype, device)
    else:
        evaluator_val = EvaluateScreen(
            network, image_evaluator, loss_screen,
            training_data.validation_image_size(), training_data.validation_image_size(),
            False, False, dtype, device)
    evaluator_vis = EvaluateScreen(
        network, image_evaluator, loss_screen,
        training_data.visualization_image_size(), training_data.visualization_image_size(),
        False, False, dtype, device)
    return evaluator_train, evaluator_val, evaluator_vis


def trace_handler(prof):
    print(prof.key_averages().table(
        sort_by="self_cuda_time_total", row_limit=-1))
    prof.export_chrome_trace("test_trace_" + str(prof.step_num) + ".json")


def build_profiler(stack):
    profiler = stack.enter_context(torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ], schedule=torch.profiler.schedule(
            wait=1,
            warmup=1,
            active=2),
        on_trace_ready=trace_handler))
    return profiler


def main():
    # Settings
    parser = build_parser()

    opt = vars(parser.parse_args())

    apply_seeding(opt['seed'])
    torch.set_num_threads(4)
    #torch.backends.cudnn.benchmark = True

    dtype = torch.float32
    device = torch.device("cuda")
    profile = opt['profile']
    opt['CUDA_Device'] = torch.cuda.get_device_name(0)

    # LOAD / initialize

    # input data
    print("Load settings, collect volumes and TFs")
    input_data = TrainingInputData(opt)

    # network
    print("Initialize network")
    network = SceneRepresentationNetwork(opt, input_data, dtype, device)
    network.to(device, dtype)

    # dataloader
    print("Create the dataloader")
    training_data = TrainingData(opt, dtype, device)
    training_data.create_dataset(input_data, network.output_mode(), network.supports_mixed_latent_spaces())

    if network.use_direction() and ('world' in [training_data.training_mode(), training_data.validation_mode()]):
        exit_upon_direction_data_error()

    # losses
    loss_screen, loss_world, loss_world_mode = LossFactory.createLosses(opt, dtype, device)
    loss_screen.to(device, dtype)
    loss_world.to(device, dtype)

    # optimizer
    optimizer = Optimizer(opt, network.parameters(), dtype, device)

    # evaluation helpers
    evaluator_train, evaluator_val, evaluator_vis = build_evaluation_helpers(
        device, dtype,
        input_data.default_image_evaluator(),
        loss_screen, loss_world, network, training_data
    )

    def run_training():
        partial_losses = defaultdict(float)
        network.train()
        num_batches = 0
        for data_tuple in training_data.training_dataloader():
            num_batches += 1
            data_tuple = utils.toDevice(data_tuple, device)

            def optim_closure():
                optimizer.zero_grad()
                prediction, total, lx = evaluator_train(data_tuple)
                for k, v in lx.items():
                    partial_losses[k] += v
                total.backward()
                # print("Grad latent:", torch.sum(network._time_latent_space.grad.detach()).item())
                # print("Batch, loss:", total.item())
                return total

            optimizer.step(optim_closure)
        return partial_losses, num_batches

    def run_validation():
        partial_losses = defaultdict(float)
        network.eval()
        num_batches = 0
        with torch.no_grad():
            for j, data_tuple in enumerate(training_data.validation_dataloader()):
                num_batches += 1
                data_tuple = utils.toDevice(data_tuple, device)
                prediction, total, lx = evaluator_val(data_tuple)
                for k, v in lx.items():
                    partial_losses[k] += v
        return partial_losses, num_batches

    def run_visualization():
        with torch.no_grad():
            visualizer = Visualizer(
                training_data.visualization_image_size(),
                input_data.num_ensembles(),
                training_data.visualization_dataloader(),
                evaluator_vis, input_data.num_tfs(), device
            )
            image = visualizer.draw_image()
        return image

    # Create the output
    storage_manager = StorageManager(opt, overwrite_output=False)
    storage_manager.print_output_directories()
    storage_manager.make_output_directories()
    storage_manager.store_script_info()
    # tensorboard logger
    storage_manager.get_tensorboard_summary()

    # compute epochs
    epochs = optimizer.num_epochs() + 1
    epochs_with_save = set(list(range(0, epochs - 1, opt['save_frequency'])) + [epochs - 1])

    # HDF5-output for summaries and export
    with storage_manager.get_hdf5_summary() as hdf5_file:
        storage_manager.initialize_hdf5_storage(
            hdf5_file, epochs, len(epochs_with_save),
            evaluator_val.loss_names(), network
        )

        start_time = time.time()
        with ExitStack() as stack:
            iteration_bar = stack.enter_context(tqdm.tqdm(total=epochs))
            if profile:
                profiler = build_profiler(stack)
            for epoch in range(epochs):
                # update network
                if network.start_epoch():
                    optimizer.reset(network.parameters())
                # update training data
                if training_data.is_rebuild_dataset():
                    if (epoch + 1) % training_data.rebuild_dataset_epoch_frequency() == 0:
                        training_data.rebuild_dataset(input_data, network.output_mode(), network)
                # TRAIN
                partial_losses, num_batches = run_training()
                storage_manager.update_training_metrics(epoch, partial_losses, num_batches, optimizer.get_lr()[0])

                # save checkpoint
                if epoch in epochs_with_save:
                    # save to torch
                    storage_manager.store_torch_checkpoint(epoch, network)
                    # save to HDF5-file
                    storage_manager.store_hdf5_checkpoint(network)

                # VALIDATE
                partial_losses, num_batches = run_validation()
                end_time = time.time()
                storage_manager.update_validation_metrics(epoch, partial_losses, num_batches, end_time - start_time)

                # VISUALIZE
                if epoch in epochs_with_save:
                    with torch.no_grad():
                        # the vis dataset contains one entry per tf-timestep-ensemble
                        # -> concatenate them into one big image
                        image = run_visualization()
                        storage_manager.store_image(epoch, image)

                # done with this epoch
                if profile:
                    profiler.step()
                optimizer.post_epoch()
                iteration_bar.update(1)
                final_loss = partial_losses['total'] / max(1, num_batches)
                iteration_bar.set_description("Loss: %7.5f" % (final_loss))
                if np.isnan(final_loss):
                    break

    print("Done in", (time.time()-start_time),"seconds")


if __name__ == '__main__':
    main()
