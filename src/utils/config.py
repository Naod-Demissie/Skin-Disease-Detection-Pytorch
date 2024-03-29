import os
import torch
from torchvision.models import (resnet18, ResNet18_Weights,
                                resnet50, ResNet50_Weights,
                                efficientnet_b0, EfficientNet_B0_Weights,
                                efficientnet_b1, EfficientNet_B1_Weights, 
                                efficientnet_b2, EfficientNet_B2_Weights, 
                                efficientnet_b3, EfficientNet_B3_Weights, 
                                efficientnet_b4, EfficientNet_B4_Weights, 
                                efficientnet_v2_m, EfficientNet_V2_M_Weights, 
                                efficientnet_v2_s, EfficientNet_V2_S_Weights)

# Hyperparameters
NUM_CLASSES = 3
LEARNING_RATE = 0.0001
BATCH_SIZE = 16
NUM_EPOCHS = 50

# Paths
ROOT_DIR = '/home/rdadmin/Documents/Naod-Projects/Skin-Disease-Detection-Pytorch'
LOCAL_DATA_DIR = '/home/rdadmin/Documents/Naod-Projects/Skin-Disease-Detection/dataset'
DERMNET_DATA_DIR = '/home/rdadmin/Documents/Datasets/Dermnet'
TB_LOG_DIR = f'{ROOT_DIR}/logs/tensorboard/'
CKPT_DIR = f'{ROOT_DIR}/checkpoints/'


DERMNET_LABEL_NAME = os.listdir(f'{DERMNET_DATA_DIR}/train')

# Compute related
NUM_WORKERS = 2
DEVICES = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# For base model search
RESIZE_SIZE = {
    'resnet18': (256, 256, 3),
    'resnet50': (256, 256, 3),
    'efficientnet_b0': (256, 256, 3), 
    'efficientnet_b1': (255, 255, 3), 
    'efficientnet_b2': (288, 288, 3), 
    'efficientnet_b3': (320, 320, 3), 
    'efficientnet_b4': (384, 384, 3), 
    'efficientnet_v2_s': (384, 384, 3), 
    'efficientnet_v2_m': (480, 480, 3), 
}

CROP_SIZE = {
    'resnet18': (224, 224, 3),
    'resnet50': (224, 224, 3),
    'efficientnet_b0': (224, 224, 3), 
    'efficientnet_b1': (240, 240, 3), 
    'efficientnet_b2': (288, 288, 3), 
    'efficientnet_b3': (300, 300, 3), 
    'efficientnet_b4': (380, 380, 3), 
    'efficientnet_v2_s': (384, 384, 3), 
    'efficientnet_v2_m': (480, 480, 3), 
}

def get_base_model(base_model_name):
    assert base_model_name in [
        'resnet18',
        'resnet50',
        'efficientnet_b0', 
        'efficientnet_b1', 
        'efficientnet_b2', 
        'efficientnet_b3', 
        'efficientnet_b4', 
        'efficientnet_v2_s', 
        'efficientnet_v2_m'
    ], f'Invalid base model name: {base_model_name}'

    return (
        resnet18(weights=ResNet18_Weights.DEFAULT) if base_model_name == 'resnet18' else
        resnet50(weights=ResNet50_Weights.DEFAULT) if base_model_name == 'resnet50' else
        efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT) if base_model_name == 'efficientnet_b0' else
        efficientnet_b1(weights=EfficientNet_B1_Weights.DEFAULT) if base_model_name == 'efficientnet_b1' else
        efficientnet_b2(weights=EfficientNet_B2_Weights.DEFAULT) if base_model_name == 'efficientnet_b2' else
        efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT) if base_model_name == 'efficientnet_b3' else
        efficientnet_b4(weights=EfficientNet_B4_Weights.DEFAULT) if base_model_name == 'efficientnet_b4' else
        efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT) if base_model_name == 'efficientnet_v2_s' else
        efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.DEFAULT)
    )
