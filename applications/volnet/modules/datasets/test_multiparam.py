import pyrenderer
import torch

cpu = torch.device('cpu')
cuda = torch.device('cuda')

vol = pyrenderer.Volume()
vol.worldX = 1.
vol.worldY = 1.
vol.worldZ = 1.

a = torch.ones(7, 4, 4, 4, device=cpu) * 1
b = torch.ones(4, 6, 6, 6, device=cpu) * 2
v = torch.ones(3, 6, 6, 6, device=cpu) * 3

vol.add_feature_from_tensor('a', a)
vol.add_feature_from_tensor('b', b)
vol.add_feature_from_tensor('v', v)

interpolator = pyrenderer.VolumeInterpolationGrid()
interpolator.grid_resolution_new_behavior = True
interpolator.setInterpolation(interpolator.Trilinear)
interpolator.setSource(vol, 0)



positions = torch.tensor([[0.5, 0.5, 0.5]], device=cuda)

output = interpolator.evaluate(positions)

print('Finished')
