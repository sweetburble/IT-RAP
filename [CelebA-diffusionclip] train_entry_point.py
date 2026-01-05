import os
os.system(
    "python \"diffusionclip_main.py\" "         # if using AttGAN, change to "attgan_main.py"
    "--mode train "
    "--dataset CelebA "     
    "--training_image_num 5 "
    "--image_size 256 "
    "--c_dim 1 "
    "--selected_attrs Male "
    "--model_save_dir=checkpoints/models/diffusionclip "
    "--result_dir=result_test "
    "--test_iters 200000 "
    "--batch_size 1 "
    "--edit_images_from_dataset "            # added part for DiffusionCLIP
    "--config celeba.yml "
    "--exp ./runs/test "
    "--n_test_img 50 "
    "--t_0 500 "
    "--n_inv_step 40 "
    "--n_test_step 40 "
    "--model_path DiffusionCLIP/checkpoint/human_male_t401.pth "
)