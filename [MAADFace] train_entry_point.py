import os
os.system(
    "python \"stargan_main.py\" "      # if using AttGAN, change to "attgan_main.py"
    "--mode train " 
    "--dataset MAADFace "
    "--training_image_num 200 "
    "--image_size 256 "
    "--c_dim 5 "
    "--selected_attrs Black_Hair Blond_Hair Brown_Hair Male Young "
    "--images_dir=MAAD-Face/data/train "
    "--attr_path=MAAD-Face/MAAD_Face_filtered.csv "
    "--model_save_dir=checkpoints/models "
    "--result_dir=result_test "
    "--test_iters 200000 "
    "--batch_size 1 "
    "--start_index 900" 
)