## Abstract

Adversarial perturbations that proactively defend against generative AI technologies, such as Deepfakes, often lose their effectiveness when subjected to common image transformations.
This is because existing schemes focus on perturbations in the spatial domain (i.e., pixel) while image transformation often targets both the spatial and frequency domains.
To overcome this, this paper presents Image Transformation-Robust Adversarial Perturbation (IT-RAP), a framework that learns a robust, multi-domain perturbation policy using Deep Reinforcement Learning (DRL).
ITRAP employs DRL to strategically allocate perturbations across a hybrid action space that includes both spatial and frequency domains.
This allows the agent to discover optimal strategies for perturbation and improve robustness against image transformations.
Our comprehensive experiments demonstrate that IT-RAP successfully disrupts deepfakes with an average success rate of 64.62% when targeting various image transformations.

"IT-RAP: overview"

<img width="500" alt="figure6" src="https://github.com/user-attachments/assets/9a0e9cda-f8f4-4470-b08c-a91219d901fc" />

The figure below shows the situation in which deepfakes are defended using adversarial perturbations, and the situation in which image transformation degrades the effectiveness of the perturbation by reducing its defense capability

<img width="350" alt="figure3" src="https://github.com/user-attachments/assets/8a10db90-1266-4844-b450-3d2aba176749" />

## Datasets and Models

**StarGAN Dataset**

```
cd stargan
bash download.sh celeba
```

**StarGAN Models**

```
bash download.sh pretrained-celeba-256x256
```

**MAAD-Face Dataset**
Follow instruction in the [MAAD-Face official repository](https://github.com/pterhoer/MAAD-Face) for downloading their data.

**AttGAN Models**
Follow instruction in the [AttGAN official repository](https://github.com/elvisyjlin/AttGAN-PyTorch) for downloading their model.

## Attack Training

### **Training Models on Your Local Machine**

You can train the StarGAN or AttGAN models by running the `[Local] execute_file.py` script. Depending on the combination of the model and dataset you wish to train, you can comment or uncomment the relevant sections of the script.

To execute, simply run the following command in your terminal:

```bash
python "[Local] execute_file.py"
```

Below are detailed instructions for the four possible training combinations.

<br>

#### **1. StarGAN with CelebA Dataset**

This method trains the **StarGAN** model using the CelebA dataset. In the `[Local] execute_file.py` file, enable the CelebA dataset code and set the execution file to `stargan_main.py`.

**Setup:**

-   Activate the CelebA code block (lines 40-52).
-   Comment out the MAAD-Face code block (lines 55-71).
-   Ensure the execution file on line 41 is `stargan_main.py`.

```python
# [Local] execute_file.py

# Run with [CelebA] dataset
os.system(
    "python \"stargan_main.py\" "  # <-- 1. Confirm StarGAN execution file
    "--mode train "
    "--dataset CelebA "
    "--training_image_num 5 "
    ...
)

# Run with [MAAD-Face] dataset
# os.system(
#     "python \"attgan_main.py\" "
#     ...
# )
```

<br>

#### **2. AttGAN with CelebA Dataset**

This method trains the **AttGAN** model using the CelebA dataset. Use the same code block as in method 1, but change the execution file to `attgan_main.py`.

**Setup:**

-   Activate the CelebA code block (lines 40-52).
-   Comment out the MAAD-Face code block (lines 55-71).
-   Modify the execution file on line 41 to `attgan_main.py`.

```python
# Run with [CelebA] dataset
os.system(
    "python \"attgan_main.py\" "  # <-- 2. Modify to AttGAN execution file
    "--mode train "
    "--dataset CelebA "
    ...
)

# Run with [MAAD-Face] dataset
# os.system(
#     "python \"attgan_main.py\" "
#     ...
# )
```

<br>

#### **3. StarGAN with MAAD-Face Dataset**

This method trains the **StarGAN** model using the MAAD-Face dataset. In `[Local] execute_file.py`, uncomment the MAAD-Face dataset code and set the execution file to `stargan_main.py`.

**Setup:**

-   Comment out the CelebA code block (lines 40-52).
-   Uncomment the MAAD-Face code block (lines 55-71).
-   Modify the execution file on line 56 to `stargan_main.py`.

```python
# Run with [CelebA] dataset
# os.system(
#     "python \"stargan_main.py\" "
#     ...
# )

# Run with [MAAD-Face] dataset
os.system(
    "python \"stargan_main.py\" " # <-- 3. Modify to StarGAN execution file
    "--mode train "
    "--dataset MAADFace "
    ...
)
```

<br>

#### **4. AttGAN with MAAD-Face Dataset**

This method trains the **AttGAN** model using the MAAD-Face dataset. Activate the MAAD-Face code block as in method 3 and ensure the execution file is `attgan_main.py`.

**Setup:**

-   Comment out the CelebA code block (lines 40-52).
-   Uncomment the MAAD-Face code block (lines 55-71).
-   Ensure the execution file on line 56 is `attgan_main.py`.

```python
# Run with [CelebA] dataset
# os.system(
#     "python \"stargan_main.py\" "
#     ...
# )

# Run with [MAAD-Face] dataset
os.system(
    "python \"attgan_main.py\" " # <-- 4. Confirm AttGAN execution file
    "--mode train "
    "--dataset MAADFace "
    ...
)
```

### **Changing Other Parameters**

Other hyperparameters, such as training image number (`--training_image_num`), selected_attrs (`--selected_attrs`), can be modified directly within each code block to suit your training requirements.

<br>

## Attack Inference

You can run inference on your local machine using a pre-trained model. The `[Infer] execute_file_infer.py` script allows you to evaluate the performance of the StarGAN or AttGAN models.

To run inference, enter the following command in your terminal:

```bash
python "[Infer] execute_file_infer.py"
```

**Prerequisites:**

Before proceeding with inference, you need the pre-trained weight file for the Rainbow DQN model. This file must be located at the following path:

-   **Weight Path:** `stargan_celeba_256/models/final_rainbow_dqn.pth`

<br>

### **Detailed Inference Instructions**

You can run four different combinations of inference by modifying the settings within the `[Infer] execute_file_infer.py` file. Select your desired model and dataset by editing line 25 and the commented-out sections of the script.

<br>

#### **1. StarGAN with CelebA Dataset**

Runs inference using the **StarGAN** model and the **CelebA** dataset.

**Setup:**

-   Ensure that line 25 in `[Infer] execute_file_infer.py` is `stargan_main.py`.
-   Make sure the CelebA dataset code block (lines 25-37) is uncommented.
-   Make sure the MAAD-Face dataset code block (lines 41-53) is commented out.

```python
# [Infer] execute_file_infer.py

# ... (Omitted top of the code)

process = subprocess.Popen(
    [
        # CelebA dataset
        "python", "stargan_main.py",  # <-- 1. Confirm StarGAN execution file
        "--mode", "inference",
        "--dataset", "CelebA",
        # ... (rest of the parameters)

        # MAAD-Face dataset
        # "python", "stargan_main.py",
        # ...
    ],
    # ... (Omitted bottom of the code)
)
```

<br>

#### **2. AttGAN with CelebA Dataset**

Runs inference using the **AttGAN** model and the **CelebA** dataset.

**Setup:**

-   Modify line 25 in `[Infer] execute_file_infer.py` to `attgan_main.py`.
-   Make sure the CelebA dataset code block (lines 25-37) is uncommented.
-   Make sure the MAAD-Face dataset code block (lines 41-53) is commented out.

```python
# [Infer] execute_file_infer.py

# ... (Omitted top of the code)

process = subprocess.Popen(
    [
        # CelebA dataset
        "python", "attgan_main.py",  # <-- 2. Modify to AttGAN execution file
        "--mode", "inference",
        "--dataset", "CelebA",
        # ... (rest of the parameters)

        # MAAD-Face dataset
        # "python", "attgan_main.py",
        # ...
    ],
    # ... (Omitted bottom of the code)
)
```

<br>

#### **3. StarGAN with MAAD-Face Dataset**

Runs inference using the **StarGAN** model and the **MAAD-Face** dataset.

**Setup:**

-   Comment out the CelebA dataset code block (lines 25-37).
-   Uncomment the MAAD-Face dataset code block (lines 41-53).
-   Set the execution file on line 41 to `stargan_main.py`.

```python
# [Infer] execute_file_infer.py

# ... (Omitted top of the code)

process = subprocess.Popen(
    [
        # CelebA dataset
        # "python", "stargan_main.py",
        # ...

        # MAAD-Face dataset
        "python", "stargan_main.py", # <-- 3. Set to StarGAN execution file
        "--mode", "inference",
        "--dataset", "MAADFace",
        # ... (rest of the parameters)
    ],
    # ... (Omitted bottom of the code)
)
```

<br>

#### **4. AttGAN with MAAD-Face Dataset**

Runs inference using the **AttGAN** model and the **MAAD-Face** dataset.

**Setup:**

-   Comment out the CelebA dataset code block (lines 25-37).
-   Uncomment the MAAD-Face dataset code block (lines 41-53).
-   Ensure the execution file on line 41 is `attgan_main.py`.

```python
# [Infer] execute_file_infer.py

# ... (Omitted top of the code)

process = subprocess.Popen(
    [
        # CelebA dataset
        # "python", "stargan_main.py",
        # ...

        # MAAD-Face dataset
        "python", "attgan_main.py", # <-- 4. Confirm AttGAN execution file
        "--mode", "inference",
        "--dataset", "MAADFace",
        # ... (rest of the parameters)
    ],
    # ... (Omitted bottom of the code)
)
```

### **Changing Other Parameters**

Other hyperparameters, such as the number of inference images (`--inference_image_num`), and selected attributes (`--selected_attrs`), can be modified directly within the `[Infer] execute_file_infer.py` script to suit your experimental requirements.

## Results
The figure below illustrates the trade-off between effectiveness and imperceptibility. While increasing λ boosts effectiveness (higher L2 error) at the cost of imperceptibility (lower PSNR) , perturbations remain visually unnoticeable at a PSNR of over 25.
<img width="2809" height="1540" alt="Image" src="https://github.com/user-attachments/assets/6c831395-01f8-48b4-97b6-e9e3e08f959b" />


<table>
  <thead>
    <tr>
      <th>Dataset</th>
      <th>Target Model</th>
      <th>Improvement Over<br>PGD / DF-RAP</th>
      <th>w/o Image Trans.</th>
      <th>Image Trans. (Avg.)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="4">CelebA</td>
      <td rowspan="2">StarGAN</td>
      <td>PGD → ITRAP</td>
      <td>8.66%</td>
      <td><b>25.11%</b></td>
    </tr>
    <tr>
      <td>DF-RAP → ITRAP</td>
      <td><b>407.14%</b></td>
      <td>103.38%</td>
    </tr>
    <tr>
      <td rowspan="2">AttGAN</td>
      <td>PGD → ITRAP</td>
      <td>92.69%</td>
      <td><b>114.01%</b></td>
    </tr>
    <tr>
      <td>DF-RAP → ITRAP</td>
      <td><b>99.19%</b></td>
      <td>10.62%</td>
    </tr>
    <tr>
      <td rowspan="4">MAAD-Face</td>
      <td rowspan="2">StarGAN</td>
      <td>PGD → ITRAP</td>
      <td>1.53%</td>
      <td><b>46.82%</b></td>
    </tr>
    <tr>
      <td>DF-RAP → ITRAP</td>
      <td><b>360.17%</b></td>
      <td>56.68%</td>
    </tr>
    <tr>
      <td rowspan="2">AttGAN</td>
      <td>PGD → ITRAP</td>
      <td>1.97%</td>
      <td><b>5.55%</b></td>
    </tr>
    <tr>
      <td>DF-RAP → ITRAP</td>
      <td><b>66.34%</b></td>
      <td>0.65%</td>
    </tr>
  </tbody>
</table>


<!-- 논문 2장에서 나온 내용 중 일부 핵심 내용을 가져와도 되고, Disrupting Deepfakes 또는 DF-RAP 논문만 언급하면 좋을 듯합니다. -->

## Related Works

Our work, IT-RAP, builds upon previous research focused on creating adversarial perturbations to defend against deepfake models. Below are some of the key studies that inspired our approach:

-   **[Disrupting Deepfakes (Ruiz et al., 2020)](https://github.com/natanielruiz/disrupting-deepfakes)**: This work introduced a PGD-based spatial domain attack to protect images from unauthorized manipulation. However, as noted in our research, these perturbations are often vulnerable to common image transformations, a challenge that our IT-RAP framework directly addresses.

-   **[DF-RAP (Qu et al., 2024)](https://github.com/ZOMIN28/DF_RAP)**: This study highlighted that adversarial perturbations can be significantly weakened by lossy compression, such as that used by online social networks. It underscores the importance of creating perturbations that are robust not just to simple transformations but also to compression artifacts, which is a key goal of our project.

## License

## Acknowledges

Our work is based on:

[1] [StarGAN](https://github.com/yunjey/stargan)

[2] [AttGAN](https://github.com/elvisyjlin/AttGAN-PyTorch)

[3] [Disrupting](https://github.com/natanielruiz/disrupting-deepfakes)

[4] [DiffJPEG](https://github.com/mlomnitz/DiffJPEG)
