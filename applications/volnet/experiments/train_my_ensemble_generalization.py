"""
From a pre-trained model on certain timesteps and ensembles,
re-train for a new ensemble by re-learning the ensemble grid
"""

import sys
import os
from typing import Dict

from volnet.experiments.profiling import build_profiler
from volnet.modules.storage_manager import StorageManager
from volnet.modules.visualizer import Visualizer

sys.path.insert(0, os.getcwd())

import numpy as np
import torch
import os
import tqdm
import time
import h5py
import argparse
import io
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
    def _extra_input_args(g):
        g.add_argument('trained_network', type=str,
                       help=".hdf5 file with the pre-trained network.")
        g.add_argument('--trained_network_epoch', type=int, default=-1,
                       help="The checkpoint to use for loading the weights")
    parser = argparse.ArgumentParser(
        description='Scene representation networks',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    TrainingInputData.init_parser(parser, _extra_input_args)
    TrainingData.init_parser(parser)
    # SceneRepresentationNetwork.init_parser(parser) #loaded from checkpoint
    LossFactory.init_parser(parser)
    Optimizer.init_parser(parser)

    this_folder = os.path.split(__file__)[0]
    StorageManager.init_parser(parser, this_folder)
    parser.add_argument('--seed', type=int, default=124, help='random seed to use. Default=124')
    return parser


def apply_seeding(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


def load_pretrained_model(opt: Dict, dtype, device):
    print("Open pre-trained model from", opt['trained_network'])
    with h5py.File(opt['trained_network'], 'r') as f:
        pretrained_opt = defaultdict(lambda: None)
        pretrained_opt.update(f.attrs)
        weights_np = f['weights'][opt['trained_network_epoch'], :]
        weights_bytes = io.BytesIO(weights_np.tobytes())

    # Important: the time keyframes must match!!
    _num_input_keyframes = len(range(*map(int, opt['time_keyframes'].split(':'))))
    if _num_input_keyframes > 1 and opt['time_keyframes'] != pretrained_opt['time_keyframes']:
        print("ERROR: For generalization, the time keyframes must match!\nFrom the network: ",
              pretrained_opt['time_keyframes'], ", from the command line:", opt['time_keyframes'])
        exit(-1)

    # for loading, I need the original number of keyframes and ensembles
    class FakeTrainingInputData(TrainingInputData):
        # noinspection PyMissingConstructor
        def __init__(self):
            # super().__init__() deliberatly not called
            self._num_timekeyframes = len(range(*map(int, pretrained_opt['time_keyframes'].split(':'))))
            self._num_ensembles = len(range(*map(int, pretrained_opt['ensembles'].split(':'))))

        def num_timekeyframes(self):
            return self._num_timekeyframes

        def num_ensembles(self):
            return self._num_ensembles

    network = SceneRepresentationNetwork(pretrained_opt, FakeTrainingInputData(), dtype, device)
    network.load_state_dict(
        torch.load(weights_bytes, map_location=device), strict=True)

    return network, pretrained_opt


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


def main():
    parser = build_parser()
    opt = vars(parser.parse_args())
    apply_seeding(opt['seed'])
    torch.set_num_threads(4)

    dtype = torch.float32
    device = torch.device("cuda")
    profile = opt['profile']

    # LOAD / initialize

    # input data
    input_data = TrainingInputData(opt)

    # network
    network, pretrained_opt = load_pretrained_model(opt, dtype, device)
    data_to_train = network.generalize_to_new_ensembles(input_data.num_ensembles())
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
    optimizer = Optimizer(opt, [data_to_train], dtype, device)

    # evaluation helpers
    evaluator_train, evaluator_val, evaluator_vis = build_evaluation_helpers(
        device, dtype,
        input_data.default_image_evaluator(),
        loss_screen, loss_world, network, training_data
    )

    # update opt-dictionary with missing keys from pretrained_opt
    # so that loading the new network without the pretrained one works
    opt = {**pretrained_opt, **opt}  # first pretrained, then normal ops, as the latter overwrites the former

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
    epochs_with_save = set(list(range(0, epochs - 1, opt['save_frequency'])) + [-1, epochs - 1])

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

            for epoch in range(-1, epochs):
                # special case epoch==-1 -> only visualize
                if epoch >= 0:
                    # update network
                    if network.start_epoch():
                        optimizer.reset(network.parameters())
                    # update training data
                    if training_data.is_rebuild_dataset():
                        if (epoch + 1) % training_data.rebuild_dataset_epoch_frequency() == 0:
                            training_data.rebuild_dataset(
                                input_data, network.output_mode(), network)

                    # TRAIN
                    partial_losses, num_batches = run_training()
                    storage_manager.update_training_metrics(epoch, partial_losses, num_batches, optimizer.get_lr()[0])
                    # print("Training epoch done, total:", partial_losses['total']/num_batches)

                # save checkpoint
                if epoch in epochs_with_save:
                    # save to tensorboard
                    storage_manager.store_torch_checkpoint(epoch, network)
                    # save to HDF5-file
                    storage_manager.store_hdf5_checkpoint(network)

                # VALIDATE
                if epoch >= 0:
                    partial_losses, num_batches = run_validation()
                    end_time = time.time()
                    storage_manager.update_validation_metrics(epoch, partial_losses, num_batches, end_time - start_time)

                # VISUALIZE
                if epoch in epochs_with_save:
                    image = run_visualization()
                    storage_manager.store_image(epoch, image)

                # done with this epoch
                if profile:
                    profiler.step()
                iteration_bar.update(1)
                if epoch >= 0:
                    optimizer.post_epoch()
                    final_loss = partial_losses['total'] / max(1, num_batches)
                    iteration_bar.set_description("Loss: %7.5f" % (final_loss))
                    if np.isnan(final_loss):
                        break

    print("Done in", (time.time() - start_time), "seconds")


if __name__ == '__main__':
    main()
