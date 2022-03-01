import argparse
import os.path
import time
from collections import defaultdict
from contextlib import ExitStack
from typing import Dict, Any

import numpy as np
import torch
import tqdm

from common import utils
from common.helpers.automation.devices import DeviceManager
from volnet.evaluation import EvaluateWorld, EvaluateScreen
from volnet.experiments.profiling import build_profiler
from volnet.lossnet import LossFactory
from volnet.modules.datasets.input_data_emulator import InputDataEmulator
from volnet.modules.datasets.position_sampler import PositionSampler
from volnet.modules.datasets.visualization_dataset import WorldSpaceVisualizationData
from volnet.modules.datasets.volume_data_storage import VolumeDataStorage
from volnet.modules.datasets.evaluation.volume_evaluator import VolumeEvaluator
from volnet.modules.datasets.world_dataset import WorldSpaceDensityData, DatasetType
from volnet.modules.render_tool import RenderTool
from volnet.modules.storage_manager import StorageManager
from volnet.modules.visualizer import Visualizer
from volnet.network import SceneRepresentationNetwork
from volnet.optimizer import Optimizer


def build_parser():
    parser = argparse.ArgumentParser()
    StorageManager.init_parser(parser, os.path.split(__file__)[0])
    VolumeDataStorage.init_parser(parser)
    PositionSampler.init_parser(parser)
    WorldSpaceDensityData.init_parser(parser)
    WorldSpaceVisualizationData.init_parser(parser)
    RenderTool.init_parser(parser)
    SceneRepresentationNetwork.init_parser(parser)
    LossFactory.init_parser(parser)
    Optimizer.init_parser(parser)
    parser.add_argument('--global-seed', type=int, default=124, help='random seed to use. Default=124')
    return parser


def select_device(force_on_single=False):
    device_manager = DeviceManager()
    device_manager.set_free_devices_as_visible(num_devices=1, force_on_single=force_on_single)
    return device_manager.get_torch_devices()[0]


def build_dataset(mode: DatasetType, args: Dict[str, Any], volume_data_storage: VolumeDataStorage, volume_evaluator: VolumeEvaluator):
    sampler = PositionSampler.from_dict(args, mode=mode)
    data = WorldSpaceDensityData.from_dict(args, volume_data_storage, mode=mode, volume_evaluator=volume_evaluator, position_sampler=sampler)
    return data


def build_evaluation_helpers(device, dtype, image_evaluator, loss_screen, loss_world, network, visualization_image_size):
    evaluator_train = EvaluateWorld(network, image_evaluator, loss_world, dtype, device)
    evaluator_val = EvaluateWorld(network, image_evaluator, loss_world, dtype, device)
    evaluator_vis = EvaluateScreen(network, image_evaluator, loss_screen, *visualization_image_size, False, False, dtype, device)
    return evaluator_train, evaluator_val, evaluator_vis


def main():
    parser = build_parser()
    args = vars(parser.parse_args())

    dtype = torch.float32
    device = select_device(force_on_single=True)

    print('[INFO] Initializing volume data storage.')
    volume_data_storage = VolumeDataStorage.from_dict(args)

    print('[INFO] Initializing rendering tool.')
    render_tool = RenderTool.from_dict(args, device,)
    volume_evaluator = render_tool.get_volume_evaluator()
    if not volume_evaluator.interpolator.grid_resolution_new_behavior:
        volume_evaluator.interpolator.grid_resolution_new_behavior = True
    image_evaluator = render_tool.get_image_evaluator()

    print('[INFO] Creating training dataset.')
    training_data = build_dataset(DatasetType.TRAINING, args, volume_data_storage, volume_evaluator)

    print('[INFO] Creating validation dataset.')
    validation_data = build_dataset(DatasetType.VALIDATION, args, volume_data_storage, volume_evaluator)

    print('[INFO] Creating visualization dataset')
    visualization_data = WorldSpaceVisualizationData.from_dict(args, volume_data_storage, render_tool=render_tool)

    print('[INFO] Initializing network')
    input_data_emulator = InputDataEmulator(volume_data_storage, training_data, validation_data)
    network = SceneRepresentationNetwork(args, input_data_emulator, dtype, device)
    network.to(device, dtype)

    print('[INFO] Building loss modules')
    loss_screen, loss_world, loss_world_mode = LossFactory.createLosses(args, dtype, device)
    loss_screen.to(device, dtype)
    loss_world.to(device, dtype)

    print('[INFO] Building optimizer')
    optimizer = Optimizer(args, network.parameters(), dtype, device)

    print('[INFO] Creating evaluation helpers')
    evaluator_train, evaluator_val, evaluator_vis = build_evaluation_helpers(device, dtype, image_evaluator, loss_screen, loss_world, network, visualization_data.image_size())
    profile = args['profile']

    def run_training():
        partial_losses = defaultdict(float)
        network.train()
        num_batches = 0
        for data_tuple in training_data.get_dataloader(shuffle=True, drop_last=False):
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
            for j, data_tuple in enumerate(validation_data.get_dataloader(shuffle=False, drop_last=False)):
                num_batches += 1
                data_tuple = utils.toDevice(data_tuple, device)
                prediction, total, lx = evaluator_val(data_tuple)
                for k, v in lx.items():
                    partial_losses[k] += v
        return partial_losses, num_batches

    def run_visualization():
        with torch.no_grad():
            visualizer = Visualizer(
                visualization_data.image_size(),
                visualization_data.num_members(),
                visualization_data.get_dataloader(),
                evaluator_vis, visualization_data.num_tfs(), device
            )
            image = visualizer.draw_image()
        return image

    print('[INFO] Setting up storage directories')

    storage_manager = StorageManager(args, overwrite_output=False)
    storage_manager.print_output_directories()
    storage_manager.make_output_directories()
    storage_manager.store_script_info()
    storage_manager.get_tensorboard_summary()

    epochs = optimizer.num_epochs() + 1
    epochs_with_save = set(list(range(0, epochs - 1, args['save_frequency'])) + [epochs - 1])

    # HDF5-output for summaries and export
    with storage_manager.get_hdf5_summary() as hdf5_file:
        storage_manager.initialize_hdf5_storage(
            hdf5_file, epochs, len(epochs_with_save),
            evaluator_val.loss_names(), network
        )

        print('[INFO] Running training loop')

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
                # TODO: Implement dataset rebuilding
                # if training_data.is_rebuild_dataset():
                #     if (epoch + 1) % training_data.rebuild_dataset_epoch_frequency() == 0:
                #         training_data.rebuild_dataset(input_data, network.output_mode(), network)
                # TRAIN
                partial_losses, num_batches = run_training()
                storage_manager.update_training_metrics(epoch, partial_losses, num_batches, optimizer.get_lr()[0])

                # save checkpoint
                if epoch in epochs_with_save:
                    storage_manager.store_torch_checkpoint(epoch, network)
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


if __name__== '__main__':
    main()