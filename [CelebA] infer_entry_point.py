import subprocess
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
num_runs = 1
for i in range(1, num_runs + 1):
    output_file = f"rainbow_inference_{i}.txt"
    print(f"Running inference iteration {i} and saving output to {output_file}...")

    with open(output_file, "w", encoding='utf-8') as f:
        process = subprocess.Popen(
            [
                "python", "attgan_main.py",     # if using StarGAN, change to "stargan_main.py"
                "--mode", "inference",
                "--dataset", "CelebA",
                "--inference_image_num", "5",
                "--image_size", "256",
                "--c_dim", "5",
                "--selected_attrs", "Black_Hair", "Blond_Hair", "Brown_Hair", "Male", "Young",
                "--images_dir", "data/celeba/images",
                "--attr_path", "data/celeba/list_attr_celeba.txt",
                "--model_save_dir", "checkpoints/models",
                "--result_dir", "result_inference",
                "--test_iters", "200000",
                "--batch_size", "1"
            ],
            stdout=f, stderr=subprocess.STDOUT
        )
        process.wait()
    print(f"Inference iteration {i} completed. Output saved to {output_file}.")