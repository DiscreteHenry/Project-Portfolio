
import os
import random
import nibabel as nib
import numpy as np
import cv2
from scipy.ndimage import rotate
from glob import glob
import argparse

def apply_motion_blur(img, size=10, angle=0):
    kernel = np.zeros((size, size))
    kernel[(size-1)//2, :] = np.ones(size)
    kernel = cv2.warpAffine(kernel, cv2.getRotationMatrix2D((size/2-0.5, size/2-0.5), angle, 1.0), (size, size))
    kernel = kernel / np.sum(kernel)
    return cv2.filter2D(img, -1, kernel)

def add_gaussian_noise(img, mean=0, std=10):
    noise = np.random.normal(mean, std, img.shape).astype(np.float32)
    return np.clip(img + noise, 0, 255)

def reduce_contrast(img, factor=0.5):
    mean = np.mean(img)
    return np.clip(mean + factor * (img - mean), 0, 255)

def crop_random_part(img, region='top', fraction=0.25):
    h = img.shape[0]
    img_copy = img.copy()
    if region == 'top':
        img_copy[:int(h*fraction), :] = 0
    elif region == 'bottom':
        img_copy[int(h*(1 - fraction)):, :] = 0
    return img_copy

def rotate_image(img, angle=10):
    return rotate(img, angle, reshape=False, mode='constant', cval=0)

def add_black_border(img, top=10, bottom=10, left=0, right=0):
    return np.pad(img, ((top, bottom), (left, right)), mode='constant', constant_values=0)

def resize_misfit(img, intermediate_size=(80, 80), target_size=(128, 128)):
    img_small = cv2.resize(img, intermediate_size)
    return cv2.resize(img_small, target_size)

def apply_random_degradations(slice_img):
    transforms = [apply_motion_blur, add_gaussian_noise, reduce_contrast, crop_random_part,
                  rotate_image, add_black_border, resize_misfit]
    num_transforms = random.randint(1, 4)
    chosen = random.sample(transforms, num_transforms)

    img = slice_img.copy()
    for t in chosen:
        if t == crop_random_part:
            img = t(img, region=random.choice(['top', 'bottom']), fraction=random.uniform(0.1, 0.3))
        elif t == rotate_image:
            img = t(img, angle=random.uniform(-15, 15))
        elif t == add_black_border:
            img = t(img, top=random.randint(5, 20), bottom=random.randint(5, 20))
        elif t == resize_misfit:
            img = t(img)
        elif t == apply_motion_blur:
            img = t(img, size=random.randint(5, 15), angle=random.randint(0, 360))
        else:
            img = t(img)
    return img

def generate_bad_scans_from_nifti(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    nifti_files = glob(os.path.join(input_folder, '*.nii*'))

    for file_path in nifti_files:
        img = nib.load(file_path)
        data = img.get_fdata()

        degraded = np.zeros_like(data)
        for i in range(data.shape[2]):
            slice_img = data[:, :, i]
            slice_img_norm = cv2.normalize(slice_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            degraded_slice = apply_random_degradations(slice_img_norm)
            degraded[:, :, i] = cv2.resize(degraded_slice, (slice_img.shape[1], slice_img.shape[0]))

        new_img = nib.Nifti1Image(degraded, img.affine, img.header)
        output_path = os.path.join(output_folder, os.path.basename(file_path).replace('.nii', '_bad.nii'))
        nib.save(new_img, output_path)

    print(f"Processed {len(nifti_files)} scans and saved to '{output_folder}'.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate degraded versions of clean NIfTI scans.')
    parser.add_argument('--input_folder', type=str, required=True, help='Path to input folder with clean NIfTI files.')
    parser.add_argument('--output_folder', type=str, required=True, help='Path to output folder to save degraded scans.')
    args = parser.parse_args()

    generate_bad_scans_from_nifti(args.input_folder, args.output_folder)
