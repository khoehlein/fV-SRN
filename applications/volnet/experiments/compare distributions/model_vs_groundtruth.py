import matplotlib.pyplot as plt
import pyrenderer
import torch

from volnet.modules.datasets.cvol_reader import VolumetricScene
from volnet.modules.datasets.evaluation import VolumeEvaluator

# checkpoint_file = '/home/hoehlein/PycharmProjects/deployment/delllat94/fvsrn/applications/volnet/experiments/refactored_network/results/model/run00021/model_epoch_200.pth'
checkpoint_file = '/home/hoehlein/PycharmProjects/deployment/delllat94/fvsrn/applications/volnet/experiments/refactored_network/results/model/run00032/model_epoch_200.pth'

if __name__ == '__main__':
    device = torch.device('cuda:0')

    checkpoint = torch.load(checkpoint_file)

    model = checkpoint['model'].to(device)
    params = checkpoint['parameters']

    data = pyrenderer.Volume(params['data_storage:filename_pattern'])
    feature = data.get_feature(0)
    resolution = feature.base_resolution()

    interpolator = pyrenderer.VolumeInterpolationGrid()
    interpolator.setSource(data, 0)
    interpolator.setInterpolation(pyrenderer.VolumeInterpolationGrid.Trilinear)
    interpolator.grid_resolution_new_behavior = True

    grid = torch.meshgrid(*reversed([(torch.arange(s) + 0.5) / s for s in [resolution.x, resolution.y, resolution.z]]))
    positions = torch.stack([g.flatten() for g in grid], dim=-1).to(device)
    time = torch.zeros_like(positions[:, 0])
    member = torch.zeros_like(positions[:, 0])

    evaluator = VolumeEvaluator(interpolator, device)
    evaluator.set_source(data, mipmap_level=0)

    with torch.no_grad():
        ground_truth = evaluator.evaluate(positions)[:, 0].data.cpu().numpy()
        prediction = model.forward(positions, None, time, member, 'world')[:, 0].data.cpu().numpy()

    print('Plotting')
    fig, ax = plt.subplots(2, 1, dpi=300, sharex='all')
    ax[0].hist(ground_truth,bins=100)
    ax[1].scatter(ground_truth, prediction, alpha=0.01)
    min_val = min(prediction.min(), ground_truth.min())
    max_val = max(prediction.max(), ground_truth.max())
    ax[1].plot([min_val, max_val], [min_val, max_val], c='red')
    ax[1].set(xlabel='Target', ylabel='Prediction')
    plt.show()

    print('Finished')
