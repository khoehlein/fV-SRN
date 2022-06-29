import torch
import numpy as np
import os
import imageio

import common.utils as utils
import pyrenderer


def convert_image(img):
    out_img = img[0,:3].cpu().detach().numpy()
    out_img *= 255.0
    out_img = out_img.clip(0, 255)
    out_img = np.uint8(out_img)
    out_img = np.moveaxis(out_img, (1, 2, 0), (0, 1, 2))
    return out_img

def evaluate(settings_file: str, volnet_folder: str, output_folder: str,
             width:int, height:int, stepsize_world: float=None):
    # Load settings file
    print("Load settings from", settings_file)
    image_evaluator = pyrenderer.load_from_json(settings_file)
    assert isinstance(image_evaluator, pyrenderer.ImageEvaluatorSimple)
    default_volume = image_evaluator.volume.volume()
    if stepsize_world is not None:
        assert isinstance(image_evaluator.ray_evaluator, pyrenderer.RayEvaluationSteppingDvr)
        if hasattr(image_evaluator.ray_evaluator, "stepsizeIsObjectSpace"):
            # old renderer version
            image_evaluator.ray_evaluator.stepsizeIsObjectSpace = False
        image_evaluator.ray_evaluator.stepsize = stepsize_world

    timer = pyrenderer.GPUTimer()

    print("Save results to", output_folder)
    os.makedirs(output_folder, exist_ok=True)
    with open(os.path.join(output_folder, "timings.csv"), "w") as stats:
        stats.write("File,Time (sec)\n")

        print("Render reference")
        img = image_evaluator.render(width, height)
        timer.start()
        img = image_evaluator.render(width, height)
        timer.stop()
        imageio.imwrite(
            os.path.join(output_folder, 'reference.png'),
            convert_image(img))
        stats.write("Reference,%.5f\n"%(timer.elapsed_milliseconds()/1000.0))

        networks = []
        for n in os.listdir(volnet_folder):
            if n.endswith('.volnet'):
                networks.append(os.path.join(volnet_folder, n))
        print("Now render", len(networks), " networks")

        volume_network = pyrenderer.VolumeInterpolationNetwork()
        image_evaluator.volume = volume_network
        for n in networks:
            base_name = os.path.split(n)[-1]
            print("Render", base_name)
            base_name = os.path.splitext(base_name)[0]
            srn = pyrenderer.SceneNetwork.load(n)
            volume_network.set_network(srn)
            img = image_evaluator.render(width, height)
            timer.start()
            img = image_evaluator.render(width, height)
            timer.stop()
            imageio.imwrite(
                os.path.join(output_folder, base_name+'.png'),
                convert_image(img))
            stats.write("%s,%.5f\n" % (base_name, timer.elapsed_milliseconds() / 1000.0))
            stats.flush()

    print("Done")


if __name__ == '__main__':
    SETTINGS_FILE = "C:/Users/ga38cat/Documents/fV-SRN-Kevin/applications/config-files/meteo-ensemble_tk_local-min-max-Sebastian.json"
    VOLNET_DIR = "D:/SceneNetworks/Kevin/ensemble/multi_grid/num_channels/6-88-63_32_1-65_fast/results/model/run00001"
    OUTPUT_DIR = "D:/SceneNetworks/Kevin/ensemble/multi_grid/num_channels/6-88-63_32_1-65_fast/results/model/run00001/img"
    WIDTH = 1024
    HEIGHT = 1024
    STEPSIZE_WORLD = 1/256

    evaluate(SETTINGS_FILE, VOLNET_DIR, OUTPUT_DIR, WIDTH, HEIGHT, STEPSIZE_WORLD)
