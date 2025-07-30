# Inference Mode
# The table result values are always obtained after running inference for all methods (PGD, baseline, DF-RAP, Ours).

import subprocess
import os

# Change the working directory.
os.chdir("/scratch/x3092a02/stargan2")

# Set environment variables (for GPU memory optimization).
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Specify the number of runs.
num_runs = 1

# Run inference using the saved Rainbow DQN weights and save the results.
for i in range(1, num_runs + 1):
    output_file = f"rainbow_inference_{i}.txt"
    print(f"Running inference iteration {i} and saving output to {output_file}...")

    with open(output_file, "w") as f:
        process = subprocess.Popen(
            [
                # CelebA dataset
                # "python", "/scratch/x3092a02/stargan2/normal_main.py",
                # "--mode", "inference",
                # "--dataset", "CelebA",
                # "--image_size", "256",
                # "--c_dim", "5",
                # "--selected_attrs", "Black_Hair", "Blond_Hair", "Brown_Hair", "Male", "Young",
                # "--celeba_image_dir", "/scratch/x3092a02/stargan/data/celeba/images",
                # "--attr_path", "/scratch/x3092a02/stargan/data/celeba/list_attr_celeba.txt",
                # "--model_save_dir", "stargan_celeba_256/models",
                # "--result_dir", "results_inference",
                # "--test_iters", "200000",
                # "--batch_size", "1"


                # MAAD-Face dataset
                "python", "/scratch/x3092a02/stargan2/normal_main.py",
                "--mode", "inference",
                "--dataset", "MAADFace",
                "--image_size", "256",
                "--c_dim", "5",
                "--selected_attrs", "Black_Hair", "Blond_Hair", "Brown_Hair", "Male", "Young",
                "--celeba_image_dir", "/scratch/x3092a02/stargan2/MAAD-Face/data/train",
                "--attr_path", "/scratch/x3092a02/stargan2/MAAD-Face/MAAD_Face_filtered.csv",
                "--model_save_dir", "stargan_celeba_256/models",
                "--result_dir", "results_inference",
                "--test_iters", "200000",
                "--batch_size", "1"

            ],
            stdout=f, stderr=subprocess.STDOUT
        )
        process.wait()

    print(f"Inference iteration {i} completed. Output saved to {output_file}.")