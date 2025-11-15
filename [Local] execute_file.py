# StarGAN Attack Test
import os
import subprocess

# # Run with [CelebA] dataset
# os.system(
#     "python \"attgan_main.py\" "
#     "--mode train "
#     "--dataset CelebA "     
#     "--training_image_num 20 "
#     "--image_size 256 "
#     "--c_dim 5 "
#     "--selected_attrs Black_Hair Blond_Hair Brown_Hair Male Young "
#     "--model_save_dir=stargan_celeba_256/models "
#     "--result_dir=result_test "
#     "--test_iters 200000 "
#     "--batch_size 1"
# )


# # Run with [MAAD-Face] dataset
os.system(
    "python \"attgan_main.py\" "
    "--mode train " 
    "--dataset MAADFace "
    "--training_image_num 20 "
    "--image_size 256 "
    "--c_dim 5 "
    "--selected_attrs Black_Hair Blond_Hair Brown_Hair Male Young "
    "--images_dir=MAAD-Face/data/train "
    "--attr_path=MAAD-Face/MAAD_Face_filtered.csv "
    "--model_save_dir=stargan_celeba_256/models "
    "--result_dir=result_test "
    "--test_iters 200000 "
    "--batch_size 1 "
    # The starting image i-index for training (starts from 0-index, training begins from the i+1th image) -> inference dataset starts from 300.
    "--start_index 900" 
)
