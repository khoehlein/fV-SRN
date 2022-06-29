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

def export(checkpoint_file: str, compiled_file_prefix: str):
    state = torch.load(checkpoint_file)
    print("state keys:", state.keys())
    model = state['model']
    print(model)
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
        net.save(filename)
        print(f"Saved ensemble {m} to {filename}")


if __name__ == '__main__':
    #TEST_PATH = "D:/SceneNetworks/Kevin/ensemble/multi_core/num_channels/12-176-125_32_1-65_fast/results/model/run00001"
    TEST_PATH = "D:/SceneNetworks/Kevin/ensemble/multi_grid/num_channels/6-88-63_32_1-65_fast/results/model/run00001"
    input_file = "model_epoch_50.pth"
    output_file = "compiled"
    export(os.path.join(TEST_PATH, input_file), os.path.join(TEST_PATH, output_file))
