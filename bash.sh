# train
conda activate thor1
cd /home/tanzl/code/githubdemo/THOR_DDPM

CUDA_VISIBLE_DEVICES=1 nohup python core/Main.py --config_path ./projects/thor/configs/brain/thor.yaml >> /home/tanzl/code/githubdemo/THOR_DDPM/weights/thor/thor/thor_sample.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python core/Main.py --config_path ./projects/thor/configs/brain/thor_subtrain.yaml >> /home/tanzl/code/githubdemo/THOR_DDPM/weights/thor/thor_subtrain/thor_sample.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python core/Main.py --config_path ./projects/thor/configs/mood_brainMRI/mood_brainMRI.yaml >> /home/tanzl/code/githubdemo/THOR_DDPM/weights/thor/mood_brainMRI/mood_brainMRI.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 python core/Main.py --config_path ./projects/thor/configs/mood_brainMRI/mood_brainMRI_augmented.yaml

# validation
conda activate thor1
cd /home/tanzl/code/githubdemo/THOR_DDPM/model_design

CUDA_VISIBLE_DEVICES=1 nohup python validation.py --pt_path /home/tanzl/code/githubdemo/THOR_DDPM/weights/thor/mood_brainMRI/trained_90epoch/best_model.pt --img_root /home/tanzl/code/githubdemo/THOR_DDPM/data/brainMRI/png/val/ --task trained_90epoch/mood_brainMRI_va --pic_num 300 >> trained_90epoch/mood_brainMRI_val.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python validation.py --pt_path /home/tanzl/code/githubdemo/THOR_DDPM/weights/thor/mood_brainMRI/trained_90epoch/best_model.pt --img_root /home/tanzl/code/githubdemo/THOR_DDPM/data/brainMRI/png/train/ --task trained_90epoch/mood_brainMRI_train --pic_num 300 >> trained_90epoch/mood_brainMRI_train.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python validation.py --pt_path /home/tanzl/code/githubdemo/THOR_DDPM/weights/thor/mood_brainMRI/trained_90epoch/best_model.pt --img_root /home/tanzl/code/githubdemo/THOR_DDPM/data/CAPS_IXI/png/ --task trained_90epoch/CAPS_IXI --pic_num 300 >> trained_45epoch/CAPS_IXI.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup python validation.py --pt_path /home/tanzl/code/githubdemo/THOR_DDPM/weights/thor/mood_brainMRI/trained_90epoch/best_model.pt --img_root /home/tanzl/data/mood/atlas/ATLAS_val_png_selected/ --task trained_90epoch/ATLAS_val_png_selected --pic_num 300 >> trained_45epoch/ATLAS_val_png_selected.log 2>&1 &



# git
# 清除缓存
 git rm -r --cached .

cd /home/tanzl/code/githubdemo/THOR_DDPM
git add --all .
git commit -m "add some key code"
git push origin main




conda activate thor1
cd /home/tanzl/code/githubdemo/THOR_DDPM/model_design

CUDA_VISIBLE_DEVICES=2 python test2.py --noise_level 350 --batch_size 1 --verbose --config_path '/home/tanzl/code/githubdemo/THOR_DDPM/projects/thor/configs/mood_brainMRI/mood_brainMRI.yaml' --pt_path '/home/tanzl/code/githubdemo/THOR_DDPM/weights/thor/mood_brainMRI/trained_90epoch/best_model.pt' --path_png '/home/tanzl/code/githubdemo/THOR_DDPM/data/brainMRI/png/toy/'  --save_path '/home/tanzl/code/githubdemo/THOR_DDPM/model_design/test2/'



CUDA_VISIBLE_DEVICES=2 nohup python test2.py --noise_level 350 --batch_size 4 --verbose --config_path '/home/tanzl/code/githubdemo/THOR_DDPM/projects/thor/configs/mood_brainMRI/mood_brainMRI.yaml' --pt_path '/home/tanzl/code/githubdemo/THOR_DDPM/weights/thor/mood_brainMRI/augtrained/2024_08_06_21_53_47_417436/best_model.pt' --path_png '/home/tanzl/code/githubdemo/THOR_DDPM/data/brainMRI/png/toy/'  --save_path_data '/home/tanzl/code/githubdemo/THOR_DDPM/model_design/test_aug_data/' >> log_temp_aug.log 2>&1 &



CUDA_VISIBLE_DEVICES=2  python test2.py --noise_level 350 --batch_size 2 --verbose --config_path '/home/tanzl/code/githubdemo/THOR_DDPM/projects/thor/configs/mood_brainMRI/mood_brainMRI.yaml' --pt_path '/home/tanzl/code/githubdemo/THOR_DDPM/weights/thor/mood_brainMRI/augtrained/2024_08_06_21_53_47_417436/best_model.pt' --path_png '/home/tanzl/code/githubdemo/THOR_DDPM/data/brainMRI/png/toy/'


CUDA_VISIBLE_DEVICES=2 nohup python test2.py --noise_level 350 --batch_size 4 --verbose \
                            --config_path '/home/tanzl/code/githubdemo/THOR_DDPM/projects/thor/configs/mood_brainMRI/mood_brainMRI.yaml' \
                            --pt_path '/home/tanzl/code/githubdemo/THOR_DDPM/weights/thor/mood_brainMRI/augtrained/2024_08_06_21_53_47_417436/best_model.pt' \
                            --path_png '/home/tanzl/code/githubdemo/THOR_DDPM/data/brainMRI/png/toy/'  \
                            --save_path_data '/home/tanzl/code/githubdemo/THOR_DDPM/model_design/test_aug_data/' >> log_temp_aug.log 2>&1 &



CUDA_VISIBLE_DEVICES=2 python test2.py --noise_level 350 --batch_size 1 --verbose \
                            --config_path '/home/tanzl/code/githubdemo/THOR_DDPM/projects/thor/configs/mood_brainMRI/mood_brainMRI.yaml' \
                            --pt_path '/home/tanzl/code/githubdemo/THOR_DDPM/weights/thor/mood_brainMRI/augtrained/2024_08_06_21_53_47_417436/best_model.pt' \
                            --path_png '/home/tanzl/code/githubdemo/THOR_DDPM/data/brainMRI/png/toy/'  \
                            --save_path_data '/home/tanzl/code/githubdemo/THOR_DDPM/model_design/test_aug_data/'


CUDA_VISIBLE_DEVICES=2 nohup python test2.py --noise_level 350 --batch_size 1 --verbose \
                            --config_path '/home/tanzl/code/githubdemo/THOR_DDPM/projects/thor/configs/mood_brainMRI/mood_brainMRI.yaml' \
                            --pt_path '/home/tanzl/code/githubdemo/THOR_DDPM/weights/thor/mood_brainMRI/augtrained/2024_08_06_21_53_47_417436/best_model.pt' \
                            --path_png '/home/tanzl/code/githubdemo/THOR_DDPM/data/brainMRI/png/toy/'  \
                            --save_path_data '/home/tanzl/code/githubdemo/THOR_DDPM/model_design/test_aug_data/' >> log_temp_aug_data.log 2>&1 &


CUDA_VISIBLE_DEVICES=2 nohup python test2.py --noise_level 350 --batch_size 1 --verbose \
                            --config_path '/home/tanzl/code/githubdemo/THOR_DDPM/projects/thor/configs/mood_brainMRI/mood_brainMRI.yaml' \
                            --pt_path '/home/tanzl/code/githubdemo/THOR_DDPM/weights/thor/mood_abtomMRI/augtrained/best_model.pt' \
                            --path_png '/home/tanzl/code/githubdemo/THOR_DDPM/data/abtomMRI/png/toy/'  \
                            --save_path_img '/home/tanzl/code/githubdemo/THOR_DDPM/model_design/test_aug_img_abtom/' \
                            --save_path_data '/home/tanzl/code/githubdemo/THOR_DDPM/model_design/test_aug_data_abtom/' >> log_temp_aug_abtom.log 2>&1 &
