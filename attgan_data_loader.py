from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import torch
import os
import random
import pandas as pd
import face_alignment
import numpy as np



class CelebA(data.Dataset):
    """Dataset class for the CelebA dataset."""
    def __init__(self, image_dir, attr_path, selected_attrs, transform, mode):
        """Initialize and preprocess the CelebA dataset."""
        self.image_dir = image_dir
        self.attr_path = attr_path
        # selected_attrs is no longer used for label generation, but only to specify the attributes to be attacked in the main logic.
        self.selected_attrs = selected_attrs
        self.transform = transform
        self.mode = mode
        self.train_dataset = []
        self.attr2idx = {}
        self.idx2attr = {}

        # Define the 13 attributes used by AttGAN for training within the class.
        # The order of this list is very important as it must match the order of the model's input channels.
        self.attgan_attrs = [
            'Bald', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Bushy_Eyebrows',
            'Eyeglasses', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'No_Beard', 'Pale_Skin', 'Young'
        ]

        # The original code generated labels only for the selected_attrs (5 attributes).
        # Since AttGAN expects all 13 attributes as input, we explicitly define the list of attributes
        # to be used as a basis for creating a 13-dimensional label vector.
        self.preprocess()

        self.num_images = len(self.train_dataset)

    def preprocess(self):
        """Preprocess the CelebA attribute file."""
        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        all_attr_names = lines[1].split()
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        lines = lines[2:]
        random.seed(1234)
        random.shuffle(lines)
        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:]


            label = []
            # Instead of the 5 selected attributes, add the values for all 13 AttGAN attributes to the label list in order.
            for attr_name in self.attgan_attrs:
                idx = self.attr2idx[attr_name]
                label.append(values[idx] == '1')
            # As a result, `label` will always be a list with 13 boolean values.

            self.train_dataset.append([filename, label])

        print('Finished preprocessing the CelebA dataset...')


    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        dataset = self.train_dataset
        filename, label = dataset[index]
        image = Image.open(os.path.join(self.image_dir, filename))
        
        # The label returned with self.transform(image) is now a 13-dimensional tensor.
        return self.transform(image), torch.FloatTensor(label), filename

    def __len__(self):
        """Return the number of images."""
        return self.num_images





class MAADFace(data.Dataset):
    """Dataset class for the MAAD-Face dataset."""
    def __init__(self, image_dir, attr_path, selected_attrs, transform, mode, start_index=0):
        self.image_dir = image_dir  # e.g., /scratch/.../train
        self.attr_path = attr_path  # e.g., /scratch/.../MAAD_Face_filtered.csv
        # selected_attrs is now used only to specify the target attributes for the attack.
        self.selected_attrs = selected_attrs
        self.transform = transform
        self.mode = mode
        self.all_dataset = []
        self.attr2idx = {}
        self.idx2attr = {}

        # Define the 13 attributes used by AttGAN for training within the class.
        # This is explicitly defined to maintain consistency with the CelebA class and
        # to serve as a basis for generating the 13-dimensional label vector required by AttGAN.
        self.attgan_attrs = [
            'Bald', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Bushy_Eyebrows',
            'Eyeglasses', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'No_Beard', 'Pale_Skin', 'Young'
        ]

        self.preprocess(start_index=start_index)
        self.num_images = len(self.all_dataset)


    def preprocess(self, start_index=0):
        df = pd.read_csv(self.attr_path, encoding='utf-8-sig')
        maad_attr_names = list(df.columns) # Includes 'Filename'

        def process_row(row):
            filename = row['Filename'].strip()
            label = []
            # Generate labels based on the list of 13 AttGAN attributes.
            for attr_name in self.attgan_attrs:
                # If the corresponding attribute column exists in the MAAD-Face dataset.
                if attr_name in maad_attr_names:
                    label.append(row[attr_name] == 1)
                # If the attribute is not in the MAAD-Face dataset (e.g., 'Mouth_Slightly_Open', 'Pale_Skin').
                else:
                    label.append(False) # Treat it as False since the attribute is missing.
            return (filename, label)
        
        # The original code generated labels only for the selected_attrs (5 attributes).
        # The modified code generates labels for all 13 AttGAN attributes.
        # For attributes not present in MAAD-Face, it fills in False, thus semantically correctly
        # constructing the 13-dimensional input required by the model.
        # This ensures that the label formats of the CelebA and MAAD-Face datasets are perfectly unified.
        full_dataset = [process_row(row) for _, row in df.iterrows()]

        self.all_dataset = full_dataset[start_index:]

    def __getitem__(self, index):
        dataset = self.all_dataset
        filename, label = dataset[index]

        image_path = os.path.join(self.image_dir, filename)
        image = Image.open(image_path).convert('RGB')
        # The face alignment feature is retained.
        image = align_face(image, crop_size=178)
        # The label returned with self.transform(image) is now a 13-dimensional tensor.
        return self.transform(image), torch.FloatTensor(label), filename

    def __len__(self):
        return self.num_images




fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device='cpu')
def align_face(image: Image.Image, crop_size=178) -> Image.Image:

    img_np = np.array(image)
    preds = fa.get_landmarks(img_np)

    # If face detection fails, return the original image.
    if preds is None or len(preds) == 0:
        print(f"[align_face] Face detection failed: {getattr(image, 'filename', 'unknown')}")
        return image.resize((crop_size, crop_size))

    # Select the largest face.
    landmarks = max(preds, key=lambda x: x[:, 1].ptp())

    # Calculate the tight bounding box of the face area.
    x_min = max(int(np.min(landmarks[:, 0])) - 10, 0)
    x_max = min(int(np.max(landmarks[:, 0])) + 10, image.width)
    y_min = max(int(np.min(landmarks[:, 1])) - 10, 0)
    y_max = min(int(np.max(landmarks[:, 1])) + 10, image.height)

    # Crop and resize the face box.
    face_box = image.crop((x_min, y_min, x_max, y_max))
    face_resized = face_box.resize((crop_size, crop_size), Image.BILINEAR)

    return face_resized




def get_loader(image_dir, attr_path, selected_attrs, crop_size=178, image_size=128,
               batch_size=16, dataset=None, mode='train', num_workers=1, start_index=0):
    """Build and return a data loader."""
    transform = []
    transform.append(T.RandomHorizontalFlip())
    transform.append(T.CenterCrop(crop_size))
    transform.append(T.Resize(image_size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)

    if dataset == 'CelebA':
        dataset = CelebA(image_dir, attr_path, selected_attrs, transform, mode)
    elif dataset == 'MAADFace':
        dataset = MAADFace(image_dir, attr_path, selected_attrs, transform, mode, start_index=start_index)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=1,
                                  shuffle = False,
                                  num_workers=num_workers)
    return data_loader