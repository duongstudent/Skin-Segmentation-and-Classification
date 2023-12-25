import os
import cv2
import glob
import torch

import numpy as np
import albumentations as A
import torch.nn.functional as F

from tqdm import tqdm
from albumentations.pytorch import ToTensorV2
from transformers import SegformerForSemanticSegmentation
from Training_ConvNeXt.utils.ConvNeXT_V1 import SkinClassifier

# ----------------- Config -----------------
MODEL_SEGMENTATION_PATH = os.getcwd() + '/model/segformers/modelSegformer_best.pth'
MODEL_CLASSIFICATION_PATH = os.getcwd() + '/model/convnext/modelConvNeXt_best.pth'

segmentation_size = 512
padding_percent = (20,20)
segmentation_transform = A.Compose([
    A.Resize(width=segmentation_size, height=segmentation_size),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0),
    ToTensorV2(), # numpy.array -> torch.tensor (B, 3, H, W)
])


classficationsize = 384
classfiation_transform = A.Compose([
        A.Resize(width=classficationsize, height=classficationsize),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255),
        ToTensorV2(), # numpy.array -> torch.tensor (B, 3, H, W)
        ])

# ----------------- Load Model -----------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_segmentation = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b4-finetuned-ade-512-512").to(device)
model_classfication = SkinClassifier('convnext_small_in22ft1k',n_class=2)


def find_bbox_from_mask(mask):
    """
    mask: numpy array (H, W)
    """
    mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    else:
        contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contour)
        return x, y, w, h


def convert_bbox_size(x,y,w,h,mask_size, image_size, padding =(0,0)):
    """
    x,y,w,h: bbox size in mask
    mask_size: size of mask
    image_size: size of image
    """
    x_new = int((x*image_size[1])/mask_size[1]) - padding[0]
    y_new = int((y*image_size[0])/mask_size[0]) - padding[1]
    w_new = int((w*image_size[1])/mask_size[1]) + padding[0]*2
    h_new = int((h*image_size[0])/mask_size[0]) + padding[1]*2

    # check if bbox out of image
    if x_new < 0:
        x_new = 0
    if y_new < 0:
        y_new = 0

    if x_new + w_new > image_size[1]:
        w_new = image_size[1]
    if y_new + h_new > image_size[0]:
        h_new = image_size[0]

    return x_new, y_new, w_new, h_new


def check_image_new_size(w,h):
    # w,h << 50px -> not enough information
    if w == 0 or h == 0:
        return False
    if w < 50 or h < 50:
        return False
    #w = h*5 or h = w*5 -> not enough information
    elif w/h > 4 or h/w > 4:
        return False
    else:
        return True


def Segmentation_Cropt_Image(model, path_image, test_transform ,padding_percent, device, trainsize):
    image = cv2.imread(path_image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_original = image.copy()
    shape_original = image.shape
    image = test_transform(image=image)['image'].unsqueeze(0).to(device)
    image_mask_original = np.zeros((shape_original[0], shape_original[1]))

    with torch.no_grad():
        y_hat = model(image.to(device))
        y_hat = y_hat.logits
        y_hat =  F.interpolate(y_hat, size=(trainsize,trainsize), mode="bilinear").argmax(dim=1).squeeze().cpu().numpy()
        image_mask_original = cv2.resize(y_hat, (shape_original[1], shape_original[0]), interpolation=cv2.INTER_NEAREST)

    x, y, w, h = find_bbox_from_mask(y_hat)
    x1, y1, w1, h1 = convert_bbox_size(x,y,w,h,(trainsize,trainsize), \
                                                            shape_original,        \
                                                            padding=(0,0))
    image_orginal_bbox = cv2.rectangle(image_original, (x1,y1), (x1+w1,y1+h1), (255,0,0), 20)
    image_mask_original = image_mask_original*255
    image_mask_original_bbox = np.stack((image_mask_original,)*3, axis=-1)
    image_mask_original_bbox = cv2.rectangle(image_mask_original_bbox, (x1,y1), (x1+w1,y1+h1), (0,0,255), 20)
    x_img, y_img, w_img, h_img = convert_bbox_size(x,y,w,h,(trainsize,trainsize), \
                                                            shape_original,        \
                                                            padding=(int(padding_percent[0]*shape_original[0]/100),int( padding_percent[1]*shape_original[1]/100)))
    
    image_orginal_bbox = cv2.rectangle(image_original, (x_img,y_img), (x_img+w_img,y_img+h_img), (0,255,0), 20)
    image_mask_original_bbox = cv2.rectangle(image_mask_original_bbox, (x_img,y_img), (x_img+w_img,y_img+h_img), (0,255,0), 20)
    image_orginal_bbox = cv2.cvtColor(image_orginal_bbox, cv2.COLOR_RGB2BGR)
    name_image = path_image.split('/')[-1].split('.')[0]
    # cv2.imwrite(os.getcwd() + '/Results/image_with_bbox.jpg', image_orginal_bbox)
    # cv2.imwrite(os.getcwd() + '/Results/mask_with_bbox.jpg', image_mask_original_bbox)
    image_integrate_bbox = np.concatenate((image_orginal_bbox, image_mask_original_bbox), axis=1)
    cv2.imwrite(os.getcwd() + '/Results/image_integrate_bbox.jpg', image_integrate_bbox)

    image = cv2.imread(path_image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    new_image = image[y_img:y_img+h_img, x_img:x_img+w_img]
    image_save = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    # Image compare between image_original and image_save
    path_image_save = os.getcwd() + '/Results/image_cropt.jpg'
    cv2.imwrite(path_image_save, image_save)
    
    image_save_padding_height = np.zeros((shape_original[0], new_image.shape[1] + 150, 3))
    image_save_padding_height[y_img:y_img+h_img, 150:] = image_save
    Image_compare = np.concatenate((cv2.cvtColor(image, cv2.COLOR_RGB2BGR), image_save_padding_height), axis=1)
    cv2.imwrite(os.getcwd() + '/Results/Image_compare.jpg', Image_compare)

    if check_image_new_size(new_image.shape[1], new_image.shape[0]):
        return path_image_save
    else:
        return None


def Classification_Image(model, image_original_path, test_transform, device):
    image_original = cv2.imread(image_original_path)
    image_original = cv2.cvtColor(image_original, cv2.COLOR_BGR2RGB)
    x = test_transform(image=image_original)['image'].unsqueeze(0).to(device)
    y_pred = model(x).float()  # Convert y_pred to float
    probs = F.softmax(y_pred, dim=-1)
    predictions = torch.argmax(probs, dim=-1)
    if int(predictions) == 1:
        return 'Malignant Skin' + ' - ' + str(round(probs[0][1].item()*100,2)) + '%'
    else:
        return 'Benign Skin' + ' - ' + str(round(probs[0][0].item()*100,2)) + '%'


def load_model():
    global MODEL_SEGMENTATION_PATH 
    global MODEL_CLASSIFICATION_PATH
    global model_segmentation
    global model_classfication
    global device
    # load segmentation model
    model_segmentation.load_state_dict(torch.load(MODEL_SEGMENTATION_PATH))
    model_segmentation.to(device)
    model_segmentation.eval()

    # load classification model
    model_classfication.load_state_dict(torch.load(MODEL_CLASSIFICATION_PATH))
    model_classfication.to(device)
    model_classfication.eval()


def predict_skin(path_image):
    global device
    global padding_percent
    global segmentation_size
    global model_segmentation
    global model_classfication
    global segmentation_transform
    global classfiation_transform

    path_image_cropt = Segmentation_Cropt_Image(model_segmentation, path_image, segmentation_transform ,padding_percent, device, segmentation_size)
    if path_image_cropt is not None:
        label_predict = Classification_Image(model_classfication, path_image_cropt, classfiation_transform, device)
        return label_predict
    
    return 'Check your image again!'

if __name__ == "__main__":
    load_model()
    path_image = 'Data_demo/benign_skin/ISIC_8515281.jpg'
    label_predict = predict_skin(path_image)
    print(label_predict)
    # for i in glob.glob('Data_demo/malignant_skin/*.jpg'):
    #     label_predict = predict_skin(i)
    #     print(i,' | ', label_predict)
