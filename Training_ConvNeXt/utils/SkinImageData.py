import cv2
import os
import glob

from torch.utils.data import Dataset


class SkinImage(Dataset):
  def __init__(self, path_folder_dataset, label_names=['benign_skin','malignant_skin'], label_encoder = [0,1], transform=None):
    super().__init__()
    self.path_folder_dataset = path_folder_dataset
    self.label_names = label_names
    self.label_encoder = label_encoder
    self.transform = transform
    self.class_folder = [os.path.join(self.path_folder_dataset, label) for label in self.label_names]
    self.images_name = []
    self.target = []
    for i, folder in enumerate(self.class_folder):
        path_image_of_class = glob.glob(folder + '/*.jpg')
        target_of_class = [self.label_encoder[i]] * len(path_image_of_class)
        self.images_name += path_image_of_class
        self.target += target_of_class

  def __len__(self):
    if len(self.images_name) == len(self.target):
      return len(self.images_name)
    return 0

  def __getitem__(self, idx):
    image_path = self.images_name[idx]
    target = self.target[idx]
    if os.path.exists(image_path):
      image = cv2.imread(image_path)
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      if self.transform:
        image = self.transform(image=image)['image']
      return image, target


if __name__ == '__main__':
    path_folder_dataset = 'Data/ISIC2020/test'

    dataset = SkinImage(path_folder_dataset)

    image, target = dataset.__getitem__(82)

    print('len dataset: ', len(dataset))
    print('image shape: ', image.shape)
    print('target: ', target)