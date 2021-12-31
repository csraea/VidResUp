# imports
from PIL import Image
import numpy as np


# ImageWrapper class that has following functionality:
# - downscale image
# - upscale image with in one of isr models methods
class ImageWrapper:
    def __init__(self, image_name, image, crop):
        self.image_name = image_name
        self.image = image
        self.crop = crop

    def get_cropped_image(self):
        return self.image.crop(self.crop)

    def spawn_new_downscaled_image(self, scale_factor=2):
        image_copy = self.image.copy()
        # donwscale image
        image_copy.thumbnail(tuple(size // scale_factor for size in self.image.size), Image.ANTIALIAS)
        # generate new crop
        crop = [size // scale_factor for size in self.crop]
        return ImageWrapper("downscaled_" + self.image_name, image_copy, crop)

    def common_upscale_to_original_size(self, model, model_name, original_crop, original_size=(1920, 1080)):
        # get image array
        image_arr = np.array(self.image)

        # do prediction
        model_image_arr = model.predict(image_arr)
        model_new_image = Image.fromarray(model_image_arr)
        # check if size the same as original image size
        while model_new_image.size[0] < original_size[0]:
            model_image_arr = model.predict(model_image_arr)
            model_new_image = Image.fromarray(model_image_arr)
        model_new_image.thumbnail(original_size, Image.ANTIALIAS)

        # define new image name
        new_image_name = self.image_name.replace("downscaled", model_name)
        return ImageWrapper(new_image_name, model_new_image, original_crop)

    def upscale_rrdn(self, model, original_crop, original_size=(1920, 1080)):
        return self.common_upscale_to_original_size(model, "rrdn", original_crop, original_size)

    def upscale_rdn_lg_nc(self, model, original_crop, original_size=(1920, 1080)):
        return self.common_upscale_to_original_size(model, "rdn_lg_nc", original_crop, original_size)

    def upscale_rdn_lg(self, model, original_crop, original_size=(1920, 1080)):
        return self.common_upscale_to_original_size(model, "rdn_lg", original_crop, original_size)

    def upscale_rdn_sm(self, model, original_crop, original_size=(1920, 1080)):
        return self.common_upscale_to_original_size(model, "rdn_sm", original_crop, original_size)