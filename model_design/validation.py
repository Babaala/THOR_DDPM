import os
import random
import torch
from torch.nn import MSELoss
from PIL import Image
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
import yaml
import sys
import argparse
sys.path.append('/home/tanzl/code/githubdemo/THOR_DDPM')
from dl_utils.config_utils import *
from optim.losses import PerceptualLoss

# 添加命令行参数解析
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--pt_path', type=str, required=True, help='Path to the model .pt file')
parser.add_argument('--img_root', type=str, required=True, help='Root directory of the images')
parser.add_argument('--task', type=str, required=True, help='Task name')
parser.add_argument('--pic_num', type=int, required=True, help='Number of pictures to process')

args = parser.parse_args()

pt_path = args.pt_path
img_root = args.img_root
task = args.task
pic_num = args.pic_num

print(os.getcwd())  # 检查当前工作目录
os.chdir('/home/tanzl/code/githubdemo/THOR_DDPM')
print(os.getcwd())  # 

img_names = [os.path.join(img_root, img_name) for img_name in os.listdir(img_root) if img_name.endswith('.png')]
# traned_45epoch_names = os.listdir('/home/tanzl/code/githubdemo/THOR_DDPM/model_design/trained_45epoch/mood_brainMRI_train')
# img_names = [os.path.join(img_root, img_name.replace("_result.png",".png")) for img_name in traned_45epoch_names if img_name.endswith('.png')]

print('len(img_names)',len(img_names))

output_dir = '/home/tanzl/code/githubdemo/THOR_DDPM/model_design/'+task+'/'
os.makedirs(output_dir, exist_ok=True)

# 随机选择图片
pic_num = 300
if len(img_names) > pic_num:
    img_names = random.sample(img_names, pic_num)


def load_image_as_tensor(image_path):
    """
    读取图像并转换为形状为 (1, 1, 128, 128) 的 PyTorch 张量。
    """
    image = Image.open(image_path).convert('L')
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)
    return image_tensor

def display_images_with_losses(img, rec, diff, losses, save_path):
    """
    显示 img, rec 和 np.abs(img - rec) 并标记损失值在右侧。
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # 显示图像
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Image')
    axes[0].axis('off')

    axes[1].imshow(rec, cmap='gray')
    axes[1].set_title('Reconstructed')
    axes[1].axis('off')

    axes[2].imshow(diff, cmap='gray')
    axes[2].set_title('Absolute Difference')
    axes[2].axis('off')

    # 显示损失值
    axes[3].text(0.5, 0.5, f'Loss_rec: {losses["loss_rec"]:.4f}\nLoss_mse: {losses["loss_mse"]:.4f}\nLoss_pl: {losses["loss_pl"]:.4f}', 
                 horizontalalignment='center', verticalalignment='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
    axes[3].axis('off')

    # 调整布局
    plt.subplots_adjust(wspace=0.3)
    plt.savefig(save_path)
    plt.close()



# 加载config_file
config_path = '/home/tanzl/code/githubdemo/THOR_DDPM/projects/thor/configs/brain/thor.yaml'
stream_file = open(config_path, 'r')
config_file = yaml.load(stream_file, Loader=yaml.FullLoader)


# 初始化DLConfigurator 
configurator_class = import_module(config_file['configurator']['module_name'],config_file['configurator']['class_name'])
configurator = configurator_class(config_file=config_file, log_wandb=True) # core.Configurator.DLConfigurator
exp_name = configurator.dl_config['experiment']['name'] # 'THOR'
method_name = configurator.dl_config['name'] # 'THOR [Gaussian] AD 350'


# 加载模型
device = configurator.device
test_model = configurator.model # 也可以直接等于configurator.model
test_model.load_state_dict(torch.load(pt_path)['model_weights'])
test_model.eval()
test_model.to(device)

# 损失
training_params = configurator.dl_config['trainer']['params']
loss_class = import_module(training_params['loss']['module_name'],
                            training_params['loss']['class_name'])
criterion_rec = loss_class(**(training_params['loss']['params'])) \
            if training_params['loss']['params'] is not None else loss_class()
criterion_MSE = MSELoss().to(device)
criterion_PL = PerceptualLoss(device=device)

metrics = {
    task + '_loss_rec': 0,
    task + '_loss_mse': 0,
    task + '_loss_pl': 0,
}
test_total = 0

# 处理每张图片
for img_path in tqdm(img_names):
    with torch.no_grad():
        x = load_image_as_tensor(img_path)
        
        b, c, h, w = x.shape
        test_total += b
        x = x.to(device)
        x = (x * 2) - 1
        x_, _ = test_model.sample_from_image(x, noise_level=test_model.noise_level_recon)
        x = (x + 1) / 2 
        x_ = (x_ + 1) / 2
        loss_rec = criterion_rec(x_, x)
        loss_mse = criterion_MSE(x_, x)
        loss_pl = criterion_PL(x_, x)
        
        metrics[task + '_loss_rec'] += loss_rec.item() * x.size(0)
        metrics[task + '_loss_mse'] += loss_mse.item() * x.size(0)
        metrics[task + '_loss_pl'] += loss_pl.item() * x.size(0)
        
        # 可视化
        rec = x_.detach().cpu()[0].numpy().squeeze()
        img = x.detach().cpu()[0].numpy().squeeze()
        diff = np.abs(img - rec)
        
        losses = {
            "loss_rec": loss_rec.item(),
            "loss_mse": loss_mse.item(),
            "loss_pl": loss_pl.item()
        }
        
        save_path = os.path.join(output_dir, os.path.basename(img_path).replace('.png', '_result.png'))
        display_images_with_losses(img, rec, diff, losses, save_path)

# 打印平均损失
for metric_key in metrics.keys():
    metric_name = task + '/' + str(metric_key)
    metric_score = metrics[metric_key] / test_total
    print({metric_name: metric_score})