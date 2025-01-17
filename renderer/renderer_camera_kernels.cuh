#pragma once

#include "renderer_tensor.cuh"
#include "renderer_utils.cuh"

/**
 * Defines:
 * IMAGE_EVALUATOR__CAMERA_T
 */

__global__ void CameraGenerateRayKernel(
	dim3 virtual_size, 
	kernel::Tensor4RW<real_t> rayStartOutput,
	kernel::Tensor4RW<real_t> rayDirectionOutput)
{
	using Camera_t = IMAGE_EVALUATOR__CAMERA_T;
	Camera_t camera;

	//LOOP
	KERNEL_3D_LOOP(x, y, b, virtual_size)
	{
		real_t nx = x;
		real_t ny = y;
		//compute normalized device coordinates in [-1,+1]^2
		real2 ndc = make_real2(
			2 * (nx + real_t(0.5)) / rayStartOutput.size(2) - 1,
			2 * (ny + real_t(0.5)) / rayStartOutput.size(1) - 1);
		//camera + ray evaluation
		const auto [rayStart, rayDir] = camera.eval(ndc, b);
		//write output
		rayStartOutput[b][y][x][0] = rayStart.x;
		rayStartOutput[b][y][x][1] = rayStart.y;
		rayStartOutput[b][y][x][2] = rayStart.z;
		rayDirectionOutput[b][y][x][0] = rayDir.x;
		rayDirectionOutput[b][y][x][1] = rayDir.y;
		rayDirectionOutput[b][y][x][2] = rayDir.z;
	}
	KERNEL_3D_LOOP_END
}
