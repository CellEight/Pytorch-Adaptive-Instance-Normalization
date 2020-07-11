from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import glob
import torch

class ImageDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        self.files = glob.glob(path+'/*.jpg')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """ Load an input image from filename and convert to a format vgg will accept.
        Taken essentialy unedited from docs: https://pytorch.org/hub/pytorch_vision_vgg/"""
        input_image = Image.open(self.files[idx]).convert("RGB")
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(input_image)
        return input_tensor

def mu(x):
    """ Takes a (n,c,h,w) tensor as input and returns the average across
    it's spatial dimensions as (h,w) tensor [See eq. 5 of paper]"""
    return torch.sum(x,(2,3))/(x.shape[2]*x.shape[3])

def sigma(x):
    """ Takes a (n,c,h,w) tensor as input and returns the standard deviation
    across it's spatial dimensions as (h,w) tensor [See eq. 6 of paper] Note
    the permutations are required for broadcasting"""
    return torch.sqrt(torch.sum((x.permute([2,3,0,1])-mu(x)).permute([2,3,0,1])**2,(2,3))/(x.shape[2]*x.shape[3]))
