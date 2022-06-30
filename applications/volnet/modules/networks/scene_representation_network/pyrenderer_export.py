import argparse
import torch
import os

import common.utils as utils
import pyrenderer

from volnet.modules.networks.scene_representation_network.interface import ISceneRepresentationNetwork
from volnet.modules.networks.input_parameterization import IInputParameterization
from volnet.modules.networks.latent_features import ILatentFeatures
from volnet.modules.networks.core_network import ICoreNetwork
from volnet.modules.networks.output_parameterization import IOutputParameterization
from volnet.modules.networks.pyrenderer import PyrendererSRN


def export(checkpoint_file: str, compiled_file_prefix: str,
           world_size_x:float=10, world_size_y:float=10, world_size_z:float=1):
    state = torch.load(checkpoint_file)
    print("state keys:", state.keys())
    model = state['model']
    print(model)
    print(model.__dir__())
    assert isinstance(model, PyrendererSRN)
    if model.uses_time():
        raise ValueError("Time dependency not supported")
    if model.output_channels() != 1:
        raise ValueError("Only a single output channel supported, not %d"%model.output_channels())
    num_members = model.num_members()
    print("Num members:", num_members)

    grid_encoding = pyrenderer.SceneNetwork.LatentGrid.Float
    for m in range(num_members):
        net = model.export_to_pyrenderer(grid_encoding, ensemble=m)
        filename = compiled_file_prefix + "-ensemble%03d.volnet"%m
        net.box_min = pyrenderer.float3(-world_size_x/2, -world_size_y/2, -world_size_z/2)
        net.box_size = pyrenderer.float3(world_size_x, world_size_y, world_size_z)
        net.save(filename)
        print(f"Saved ensemble {m} to {filename}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-directory', type=str, required=True)
    parser.add_argument('--checkpoint-file-name', type=str, required=True)
    args = vars(parser.parse_args())

    checkpoint_base_dir = os.path.join(args['experiment_directory'], 'results', 'model')
    runs = os.listdir(checkpoint_base_dir)
    volnet_base_dir = os.path.join(args['experiment_directory'], 'results', 'volnet')

    for run in runs:
        output_dir = os.path.join(volnet_base_dir, run)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        checkpoint_path = os.path.join(checkpoint_base_dir, run, args['checkpoint_file_name'])
        output_file_prefix = os.path.join(output_dir, os.path.splitext(args['checkpoint_file_name'])[0])

        export(checkpoint_path, output_file_prefix)


if __name__ == '__main__':
    # #TEST_PATH = "D:/SceneNetworks/Kevin/ensemble/multi_core/num_channels/12-176-125_32_1-65_fast/results/model/run00001"
    # TEST_PATH = "/home/hoehlein/PycharmProjects/results/fvsrn/paper/ensemble/multi_grid/num_channels/6-88-63_32_1-65_fast/results/model/run00001"
    # input_file = "model_epoch_50.pth"
    # output_file = "compiled"
    # export(os.path.join(TEST_PATH, input_file), os.path.join(TEST_PATH, output_file))
    main()
