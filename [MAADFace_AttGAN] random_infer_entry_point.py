"""
Random Policy Baseline Inference: MAADFace dataset + AttGAN model
==================================================================
Ablation study for IJCAI rebuttal: demonstrates the necessity of RL by
comparing against a uniform-random action selection baseline.

Run:
    python "[MAADFace_AttGAN] random_infer_entry_point.py"
"""
import subprocess
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

output_file = "random_maadface_attgan_inference.txt"
print(f"[Random Baseline] MAADFace / AttGAN  ->  {output_file}")

with open(output_file, "w", encoding="utf-8") as f:
    process = subprocess.Popen(
        [
            "python", "attgan_main.py",
            "--mode", "inference",
            "--action_policy", "random",          # <-- Random baseline
            "--dataset", "MAADFace",
            "--inference_image_num", "100",
            "--image_size", "256",
            "--c_dim", "5",
            "--selected_attrs", "Black_Hair", "Blond_Hair", "Brown_Hair", "Male", "Young",
            "--images_dir", "MAAD-Face/data/train",
            "--attr_path", "MAAD-Face/MAAD_Face_filtered.csv",
            "--model_save_dir", "checkpoints/models",
            "--result_dir", "result_random_maadface_attgan",
            "--test_iters", "200000",
            "--max_steps_per_episode", "18",
            "--feature_extractor_name", "edgeface",
            "--feature_extractor_frequency", "3",
            "--batch_size", "1",
        ],
        stdout=f, stderr=subprocess.STDOUT,
    )
    process.wait()

print(f"[Random Baseline] MAADFace / AttGAN done. Output: {output_file}")
