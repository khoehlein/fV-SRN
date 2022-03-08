import matplotlib.pyplot as plt
import pyrenderer
import torch

from volnet.modules.datasets.cvol_reader import VolumetricScene
from volnet.modules.datasets.evaluation import VolumeEvaluator

# checkpoint_file = '/home/hoehlein/PycharmProjects/deployment/delllat94/fvsrn/applications/volnet/experiments/refactored_network/results/model/run00021/model_epoch_200.pth'
checkpoint_file = '/home/hoehlein/PycharmProjects/deployment/delllat94/fvsrn/applications/volnet/experiments/refactored_network/results/model/run00045/model_epoch_200.pth'

if __name__ == '__main__':
    device = torch.device('cuda:0')

    checkpoint = torch.load(checkpoint_file)

    model = checkpoint['model'].to(device)
    params = checkpoint['parameters']

    ensemble_index = params['data_storage:ensemble:index_range']
    active_member = 1
    if ensemble_index is not None:
        ensemble_index = list(range(*map(int, ensemble_index.split(':'))))
    else:
        ensemble_index = [0]
    fig, ax = plt.subplots(2, len(ensemble_index), dpi=300, sharex='all', sharey='row')
    for active_member, _ in enumerate(ensemble_index):
        data = pyrenderer.Volume(params['data_storage:filename_pattern'].format(member=ensemble_index[active_member]))
        feature = data.get_feature(0)
        resolution = feature.base_resolution()

        interpolator = pyrenderer.VolumeInterpolationGrid()
        interpolator.setSource(data, 0)
        interpolator.setInterpolation(pyrenderer.VolumeInterpolationGrid.Trilinear)
        interpolator.grid_resolution_new_behavior = True

        grid = torch.meshgrid(*reversed([(torch.arange(s) + 0.5) / s for s in [resolution.x, resolution.y, resolution.z]]))
        positions = torch.stack([g.flatten() for g in grid], dim=-1).to(device)

        time = torch.zeros_like(positions[:, 0])
        member = torch.full_like(positions[:, 0], active_member)

        evaluator = VolumeEvaluator(interpolator, device)
        evaluator.set_source(data, mipmap_level=0)

        with torch.no_grad():
            ground_truth = evaluator.evaluate(positions)[:, 0].data.cpu().numpy()
            prediction = model.forward(positions, None, time, member, 'world')[:, 0].data.cpu().numpy()

        print('Plotting')

        ax[0, active_member].hist(ground_truth,bins=100)
        ax[1, active_member].scatter(ground_truth, prediction, alpha=0.01)
        min_val = min(prediction.min(), ground_truth.min())
        max_val = max(prediction.max(), ground_truth.max())
        ax[1, active_member].plot([min_val, max_val], [min_val, max_val], c='red')
        ax[1, active_member].set(xlabel='Target', ylabel='Prediction')
    plt.tight_layout()
    plt.show()
    print('Finished')
