import os
import torch

# Advanced CUDA memory optimization
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,roundup_power2_divisions:16")
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")  # For better error reporting

# Enable memory efficient attention if available
if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
    torch.backends.cuda.enable_flash_sdp(True)

os.system(
    "python \"diffusionclip_main.py\" "
    "--mode train "
    "--dataset CelebA "     
    "--training_image_num 500 "
    "--image_size 256 "
    "--c_dim 1 "
    "--selected_attrs Male "
    "--model_save_dir=checkpoints/models/diffusionclip "
    "--result_dir=result_diffusionclip "
    "--test_iters 200000 "
    "--batch_size 1 "
    "--edit_images_from_dataset "
    "--config celeba.yml "
    "--exp ./runs/test "
    "--n_test_img 50 "
    "--t_0 400 "
    "--n_inv_step 40 "
    "--n_test_step 40 "
    "--model_path DiffusionCLIP/checkpoint/human_male_t401.pth "
    "--attack_epsilon 0.12 "           
    "--attack_alpha 0.035 "            
    "--pgd_iter 15 "                   
    "--dct_iter 10 "                   
    "--dct_clamp 0.20 "                
    "--attack_perceptual_weight 0.25 "
    "--attack_tv_weight 0.008 "
    "--max_steps_per_episode 10 "      
    "--action_dim 4 "                  
    "--noise_level 0.005 "             
    "--reward_weight 0.7 "
)