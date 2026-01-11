import torch
import cv2
import numpy as np

from torchvision.io import encode_jpeg, decode_jpeg

import random

def apply_random_transform(original_gen_image, perturbed_gen_image):

    transform_options = [
        "no_transform",
        "compress_jpeg",
        "denoise_opencv",
        "median_filter",
        "random_resize_padding",
        "random_image_transforms"
    ]


    selected_transform = random.choice(transform_options)
    print(f"Selected transform: {selected_transform}")


    if selected_transform == "no_transform":

        return original_gen_image, perturbed_gen_image

    elif selected_transform == "compress_jpeg":


        quality = random.randint(50, 95)
        compressed_original = compress_jpeg(original_gen_image, quality)
        compressed_perturbed = compress_jpeg(perturbed_gen_image, quality)
        return compressed_original, compressed_perturbed

    elif selected_transform == "denoise_opencv":

        denoised_original = denoise_opencv(original_gen_image)
        denoised_perturbed = denoise_opencv(perturbed_gen_image)
        return denoised_original, denoised_perturbed

    elif selected_transform == "median_filter":

        denoised_original = denoise_scikit(original_gen_image)
        denoised_perturbed = denoise_scikit(perturbed_gen_image)
        return denoised_original, denoised_perturbed

    elif selected_transform == "random_resize_padding":

        transformed_original, transformed_perturbed = random_resize_padding(original_gen_image, perturbed_gen_image)
        return transformed_original, transformed_perturbed

    elif selected_transform == "random_image_transforms":

        transformed_original, transformed_perturbed = random_image_transforms(original_gen_image, perturbed_gen_image)
        return transformed_original, transformed_perturbed

    else:

        print("Unknown transform option selected. Returning original images.")
        return original_gen_image, perturbed_gen_image


def compress_jpeg(x_adv, quality=75):

    device = x_adv.device


    x_adv_cpu = x_adv.to('cpu')


    x_adv_uint8 = ((x_adv_cpu + 1) * 127.5).clamp(0, 255).to(torch.uint8)


    compressed_batch = []
    for i in range(x_adv_uint8.size(0)):

        encoded = encode_jpeg(x_adv_uint8[i], quality=quality)


        decoded = decode_jpeg(encoded)

        compressed_batch.append(decoded)


    compressed = torch.stack(compressed_batch)


    compressed_float = (compressed.float() / 127.5) - 1


    compressed_float = compressed_float.to(device)

    return compressed_float


def denoise_opencv(x_adv):

    device = x_adv.device
    x_cpu = x_adv.detach().cpu()


    batch_size = x_cpu.shape[0]


    result = torch.zeros_like(x_cpu)

    for i in range(batch_size):

        img_np = x_cpu[i].numpy()


        img_np = np.transpose(img_np, (1, 2, 0))


        img_np = ((img_np + 1) * 127.5).astype(np.uint8)






        denoised_img = cv2.fastNlMeansDenoisingColored(img_np, None, 10, 10, 7, 21)


        denoised_img = (denoised_img.astype(np.float32) / 127.5) - 1


        denoised_img = np.transpose(denoised_img, (2, 0, 1))


        result[i] = torch.from_numpy(denoised_img)


    return result.to(device)


from skimage.restoration import denoise_tv_chambolle, denoise_bilateral, denoise_wavelet, denoise_nl_means, estimate_sigma
from skimage import filters

def denoise_scikit(x_adv):

    x_np = x_adv.detach().cpu().numpy()


    x_img = np.squeeze(x_np, axis=0)
    x_img = np.transpose(x_img, (1, 2, 0))


    x_img = (x_img + 1) / 2


    sigma_est = estimate_sigma(x_img, channel_axis=-1)






























    denoised_img = np.zeros_like(x_img)
    for i in range(x_img.shape[2]):
        denoised_img[:, :, i] = filters.median(x_img[:, :, i])











    denoised_img = denoised_img * 2 - 1


    denoised_img = np.transpose(denoised_img, (2, 0, 1))
    denoised_img = np.expand_dims(denoised_img, axis=0)


    denoised_tensor = torch.from_numpy(denoised_img).to(x_adv.device)

    return denoised_tensor


import torch.nn.functional as F
import random

def random_resize_padding(x_real, x_adv):

    original_size = 256


    resize_sizes = [208, 224, 240]
    resize_size = random.choice(resize_sizes)
    print(f"Resize selected for this step: {resize_size}")


    x_adv_resized = F.interpolate(x_adv, size=(resize_size, resize_size), mode='bilinear', align_corners=False)
    x_real_resized = F.interpolate(x_real, size=(resize_size, resize_size), mode='bilinear', align_corners=False)


    pad_diff = original_size - resize_size


    pad_left = random.randint(0, pad_diff)
    pad_right = pad_diff - pad_left
    pad_top = random.randint(0, pad_diff)
    pad_bottom = pad_diff - pad_top


    padding = (pad_left, pad_right, pad_top, pad_bottom)

    x_adv_padded = F.pad(x_adv_resized, padding, mode='constant', value=0)
    x_real_padded = F.pad(x_real_resized, padding, mode='constant', value=0)

    return x_real_padded, x_adv_padded


import torchvision.transforms.functional as TF

def random_image_transforms(x_real, x_adv):
    batch_size, channels, height, width = x_adv.shape


    apply_shear = random.random() > 0.5
    apply_shift = random.random() > 0.5
    apply_zoom = random.random() > 0.5
    apply_rotation = random.random() > 0.5


    angle = random.uniform(-15, 15) if apply_rotation else 0
    shear = (random.uniform(-10, 10) if apply_shear else 0,
            random.uniform(-10, 10) if apply_shear else 0)
    translate = (int(random.uniform(-0.1, 0.1) * width) if apply_shift else 0,
               int(random.uniform(-0.1, 0.1) * height) if apply_shift else 0)
    scale = random.uniform(0.9, 1.1) if apply_zoom else 1.0


    x_real_01 = (x_real + 1) / 2
    x_adv_01 = (x_adv + 1) / 2


    transformed_x_real = []
    transformed_x_adv = []

    for i in range(batch_size):

        trans_img_real = TF.affine(x_real_01[i], angle=angle, translate=translate,
                            scale=scale, shear=shear,
                            interpolation=TF.InterpolationMode.BILINEAR)

        trans_img_adv = TF.affine(x_adv_01[i], angle=angle, translate=translate,
                            scale=scale, shear=shear,
                            interpolation=TF.InterpolationMode.BILINEAR)


        transformed_x_real.append(trans_img_real)
        transformed_x_adv.append(trans_img_adv)


    transformed_x_real = torch.stack(transformed_x_real)
    transformed_x_adv = torch.stack(transformed_x_adv)


    transformed_x_real = transformed_x_real * 2 - 1
    transformed_x_adv = transformed_x_adv * 2 - 1

    return transformed_x_real, transformed_x_adv
