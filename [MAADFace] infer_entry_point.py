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
                "python", "stargan_main.py",    # if using AttGAN, change to "attgan_main.py"
                "--mode", "inference",
                "--dataset", "MAADFace",
                "--inference_image_num", "100",
                "--image_size", "256",
                "--c_dim", "5",
                "--selected_attrs", "Black_Hair", "Blond_Hair", "Brown_Hair", "Male", "Young",
                "--images_dir", "MAAD-Face/data/train",
                "--attr_path", "MAAD-Face/MAAD_Face_filtered.csv",
                "--model_save_dir", "checkpoints/models",
                "--result_dir", "result_inference",
                "--test_iters", "200000",
                "--max_steps_per_episode", "18",
                "--feature_extractor_name", "edgeface",
                "--feature_extractor_frequency", "3",
                "--batch_size", "1"
            ],
            stdout=f, stderr=subprocess.STDOUT
        )
        process.wait()
    print(f"Inference iteration {i} completed. Output saved to {output_file}.")
