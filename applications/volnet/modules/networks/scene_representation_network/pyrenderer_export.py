import torch
import os

import common.utils as utils
import pyrenderer

from volnet.modules.networks.scene_representation_network.interface import ISceneRepresentationNetwork
from volnet.modules.networks.input_parameterization import IInputParameterization
from volnet.modules.networks.latent_features import ILatentFeatures
from volnet.modules.networks.core_network import ICoreNetwork
from volnet.modules.networks.output_parameterization import IOutputParameterization

def export(checkpoint_file: str, compiled_file: str):
    state = torch.load(checkpoint_file)
    print("state keys:", state.keys())
    model = state['model']
    print(model)


if __name__ == '__main__':
    TEST_PATH = "D:/SceneNetworks/Kevin/ensemble/multi_core/num_channels/12-176-125_32_1-65_fast/results/model/run00001"
    input_file = "model_epoch_50.pth"
    output_file = "compiled.volnet"
    export(os.path.join(TEST_PATH, input_file), os.path.join(TEST_PATH, output_file))
