"""
Greedy Policy Baseline Inference: CelebA dataset + StarGAN model
=================================================================
Ablation study for IJCAI rebuttal: demonstrates the necessity of RL by
comparing against a calibration-based greedy action selection baseline.

Calibration phase (50 images from training split, not used during IT-RAP training):
  For each of the 5 transform types {JPEG, OpenCV, Median, Resize&Pad, Affine},
  each of the 4 actions is applied once per image and the resulting L2 gain
  (MSE between original_gen and perturbed_gen after transform) is measured.
  The action with the highest average L2 gain is selected for that transform.

Inference phase:
  At step t, the transform type is determined by (t % 5) cycling through
  [jpeg, opencv, median, padding, affine], and the best-ranked action for
  that transform is always selected.

Run:
    python "[CelebA_StarGAN] greedy_infer_entry_point.py"
"""
import subprocess
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

output_file = "greedy_celeba_stargan_inference.txt"
print(f"[Greedy Baseline] CelebA / StarGAN  ->  {output_file}")

with open(output_file, "w", encoding="utf-8") as f:
    process = subprocess.Popen(
        [
            "python", "stargan_main.py",
            "--mode", "inference",
            "--action_policy", "greedy",          # <-- Greedy baseline
            "--calib_image_num", "50",            # calibration set size
            "--dataset", "CelebA",
            "--inference_image_num", "100",
            "--image_size", "256",
            "--c_dim", "5",
            "--selected_attrs", "Black_Hair", "Blond_Hair", "Brown_Hair", "Male", "Young",
            "--images_dir", "data/celeba/images",
            "--attr_path", "data/celeba/list_attr_celeba.txt",
            "--model_save_dir", "checkpoints/models",
            "--result_dir", "result_greedy_celeba_stargan",
            "--test_iters", "200000",
            "--max_steps_per_episode", "18",
            "--feature_extractor_name", "edgeface",
            "--feature_extractor_frequency", "3",
            "--batch_size", "1",
        ],
        stdout=f, stderr=subprocess.STDOUT,
    )
    process.wait()

print(f"[Greedy Baseline] CelebA / StarGAN done. Output: {output_file}")
