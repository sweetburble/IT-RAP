"""
Random Policy Baseline Inference: CelebA dataset + StarGAN model
=================================================================
Ablation study for IJCAI rebuttal: demonstrates the necessity of RL by
comparing against a uniform-random action selection baseline.

At every step, one of the 4 actions {PGD, Freq-LOW, Freq-MID, Freq-HIGH}
is chosen uniformly at random (no learned policy).

Run:
    python "[CelebA_StarGAN] random_infer_entry_point.py"
"""
import subprocess
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

output_file = "random_celeba_stargan_inference.txt"
print(f"[Random Baseline] CelebA / StarGAN  ->  {output_file}")

with open(output_file, "w", encoding="utf-8") as f:
    process = subprocess.Popen(
        [
            "python", "stargan_main.py",
            "--mode", "inference",
            "--action_policy", "random",          # <-- Random baseline
            "--dataset", "CelebA",
            "--inference_image_num", "100",
            "--image_size", "256",
            "--c_dim", "5",
            "--selected_attrs", "Black_Hair", "Blond_Hair", "Brown_Hair", "Male", "Young",
            "--images_dir", "data/celeba/images",
            "--attr_path", "data/celeba/list_attr_celeba.txt",
            "--model_save_dir", "checkpoints/models",
            "--result_dir", "result_random_celeba_stargan",
            "--test_iters", "200000",
            "--max_steps_per_episode", "18",
            "--feature_extractor_name", "edgeface",
            "--feature_extractor_frequency", "3",
            "--batch_size", "1",
        ],
        stdout=f, stderr=subprocess.STDOUT,
    )
    process.wait()

print(f"[Random Baseline] CelebA / StarGAN done. Output: {output_file}")
