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
                "python", "main.py",
                "--edit_one_image",
                "--config", "celeba.yml",
                "--exp", "./runs/test_edit_one",
                "--t_0", "500",
                "--n_inv_step", "40",
                "--n_test_step", "40",
                "--n_iter", "1",
                "--img_path", "imgs/1.png",
                "--model_path", "checkpoints/human_neanderthal_t601.pth",
            ],
            stdout=f, stderr=subprocess.STDOUT
        )
        process.wait()
    print(f"Inference iteration {i} completed. Output saved to {output_file}.")
