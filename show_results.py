import itertools

import torch.nn
from torch.utils.data import DataLoader
from torch import optim
from torchvision.transforms import transforms
from dataset import Datasets
from tensorboardX import SummaryWriter
from SGU_Net import SGU_Net
from metrics import *
from autoED import conv_encoder,conv_decoder
from utils import *
from losses import DC_and_HDBinary_loss
import os
import PIL.Image as Image
import cv2

x_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

ckp_path = './ckp_256/best_SGU_NetepochG.pth'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = SGU_Net(1,2).to(device)
model.load_state_dict(torch.load(ckp_path))
model = model.eval()


total_loss = 0
total_acc = 0
predict_dir = './val_preds/preds'
predict_label_dir = './val_preds/labels'
os.makedirs(predict_dir,exist_ok=True)
os.makedirs(predict_label_dir,exist_ok=True)
val_images_dir = './datasets/images/val'
val_labels_dir = './datasets/labels/val'
val_images_list = os.listdir(val_images_dir)

for img in val_images_list:
    print(img)
    img_path = os.path.join(val_images_dir,img)
    label_path = os.path.join(val_labels_dir, img)
    save_path = os.path.join(predict_dir,img)
    label_save_path =os.path.join(predict_label_dir,img)
    img_x = Image.open(img_path)
    img_x = img_x.resize((256, 256))
    img_x = img_x.convert('L')

    img_y = Image.open(label_path)
    img_y = img_y.resize((256, 256))
    img_y = img_y.convert('L')
    img_y.save(label_save_path)

    img_x = x_transforms(img_x)
    img_x = img_x.unsqueeze(dim=0)
    inputs = img_x.to(device)

    output = model(inputs)
    output = torch.softmax(output, dim=1).detach().cpu().numpy()
    output_ = output[0, 1]
    for i_x in range(256):
        for i_y in range(256):
            if output_[i_x, i_y] > 0.5:
                output_[i_x, i_y] = 255
            else:
                output_[i_x, i_y] = 0
    cv2.imwrite(save_path,(output_).astype('uint8'))
    # print(outputs.shape)