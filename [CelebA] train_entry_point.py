import os
os.system(
    "python \"diffusionclip_main.py\" "         # if using AttGAN, change to "attgan_main.py"
    "--mode train "
    "--dataset CelebA "     
    "--training_image_num 5 "
    "--image_size 256 "
    "--c_dim 5 "
    "--selected_attrs Black_Hair Blond_Hair Brown_Hair Male Young "
    "--model_save_dir=stargan_celeba_256/models "
    "--result_dir=result_test "
    "--test_iters 200000 "
    "--batch_size 1"
)