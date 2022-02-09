# the vis dataset contains one entry per tf-timestep-ensemble
# -> concatenate them into one big image
import numpy as np
from PIL import ImageDraw, Image, ImageFont

from common import utils


class Visualizer(object):

    def __init__(self, image_size, num_ensembles, data_loader, evaluator, num_tfs, device, left=50, bottom=100, text_fill=None, vis_fnt=None):
        if text_fill is None:
            text_fill = (120, 120, 120, 255)
        self.image_size = image_size
        self.num_ensembles = num_ensembles
        self.data_loader = data_loader
        self.evaluator = evaluator
        self.device = device
        self.num_tfs = num_tfs
        self.left = left
        self.bottom = bottom
        self.text_fill = text_fill
        self.width = left + (len(data_loader) // num_ensembles) * image_size
        self.single_height = bottom + 2 * image_size
        self.height = self.single_height * num_ensembles
        if vis_fnt is None:
            try:
                vis_fnt = ImageFont.truetype("arial.ttf", 12)
            except OSError:
                # Unix, try free-font
                try:
                    vis_fnt = ImageFont.truetype("FreeSans.ttf", 12)
                except OSError:
                    vis_fnt = ImageDraw.getfont()  # fallback
        self.vis_fnt = vis_fnt
        self.draw = None

    @staticmethod
    def _convert_image(img):
        out_img = img[0].cpu().detach().numpy()
        out_img *= 255.0
        out_img = out_img.clip(0, 255)
        out_img = np.uint8(out_img)
        out_img = np.moveaxis(out_img, (1, 2, 0), (0, 1, 2))
        return Image.fromarray(out_img)

    def _centered_text(self, text, x_left, y):
        w, h = self.draw.textsize(text)
        x = int(x_left + self.image_size / 2 - w / 2)
        self.draw.text((x, y), text, fill=self.text_fill, font=self.vis_fnt)
        return y + h

    def draw_image(self):
        img_size = self.image_size
        image = Image.new('RGBA', (self.width, self.height))
        self.draw = ImageDraw.Draw(image)
        self.draw.text((5, img_size // 2), "pred", fill=self.text_fill, font=self.vis_fnt)
        self.draw.text((5, img_size + img_size // 2), "gt", fill=self.text_fill, font=self.vis_fnt)
        for j, data_tuple in enumerate(self.data_loader):
            target = data_tuple[1]
            tf_index = data_tuple[2].item()
            time_index = data_tuple[3].item()
            ensemble_index = data_tuple[4].item()
            data_tuple = utils.toDevice(data_tuple, self.device)
            prediction, total, lx = self.evaluator(data_tuple)

            posX = j // self.num_ensembles
            posY = j % self.num_ensembles

            image.paste(self._convert_image(prediction),
                        box=(self.left + posX * img_size, posY * self.single_height))
            image.paste(self._convert_image(target),
                        box=(self.left + posX * img_size, posY * self.single_height + img_size))

            y = self._centered_text(
                "TF=%d, Time=%.2f, Ensemble=%.2f" % (tf_index, time_index, ensemble_index),
                self.left + posX * img_size, posY * self.single_height + 2 * img_size + 10)
            y = self._centered_text("DSSIM: %.5f" % lx['dssim'], self.left + posX * img_size, y + 5)
            y = self._centered_text("LPIPS: %.5f" % lx['lpips'], self.left + posX * img_size, y + 5)
        self.draw = None
        return image