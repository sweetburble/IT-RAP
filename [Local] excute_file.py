# StarGAN Attack Test
import os
import subprocess

# Change current directory
# os.chdir("C:\\Users\\Bandi\\Desktop\\stargan & attgan")

# [Cloud Server] Set the environment variable to use the second GPU (index 1).
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# # Specify the number of runs (default = 1).
# num_runs = 1
# # Run the StarGAN attack test and save the results.
# for i in range(1, num_runs + 1):
#     output_file = f"[Optuna] Test_Result_4.txt"
#     print(f"Running test iteration {i} and saving output to {output_file}...")
#     with open(output_file, "w") as f:
#         process = subprocess.Popen(
#             [
#                 "python", "/scratch/x3092a02/stargan2/optuna_main.py",
#                 "--mode", "test",
#                 "--dataset", "CelebA",
#                 "--image_size", "256",
#                 "--c_dim", "5",
#                 "--selected_attrs", "Black_Hair", "Blond_Hair", "Brown_Hair", "Male", "Young",
#                 "--celeba_image_dir", "/scratch/x3092a02/stargan/data/celeba/images",
#                 "--attr_path", "/scratch/x3092a02/stargan/data/celeba/list_attr_celeba.txt",
#                 "--model_save_dir", "stargan_celeba_256/models",
#                 "--result_dir", "results_test",
#                 "--test_iters", "200000",
#                 "--batch_size", "1"
#             ],
#             stdout=f, stderr=subprocess.STDOUT
#         )
#         process.wait()  # Wait for the process to complete.
#     print(f"Test iteration {i} completed. Output saved to {output_file}.")


# Run with [CelebA] dataset
os.system(
    "python \"attgan_main.py\" "
    "--mode test "
    "--dataset CelebA "
    "--image_size 256 "
    "--c_dim 5 "
    "--selected_attrs Black_Hair Blond_Hair Brown_Hair Male Young "
    "--model_save_dir=stargan_celeba_256\\models "
    "--result_dir=result_test "
    "--test_iters 200000 "
    "--batch_size 1"
)

# Run with [MAAD-Face] dataset
# os.system(
#     "python \"attgan_main.py\" "
#     "--mode test "         
#     "--dataset MAADFace "
#     "--image_size 256 "
#     "--c_dim 5 "
#     "--selected_attrs Black_Hair Blond_Hair Brown_Hair Male Young "
#     "--celeba_image_dir=MAAD-Face\\data\\train "
#     "--attr_path=MAAD-Face\\MAAD_Face_filtered.csv "
#     "--model_save_dir=stargan_celeba_256\\models "
#     "--result_dir=result_test "
#     "--test_iters 200000 "
#     "--batch_size 1 "
#     # The starting image i-index for training (starts from 0-index, training begins from the i+1th image) -> inference dataset starts from 300.
#     "--start_index 900" 
# )