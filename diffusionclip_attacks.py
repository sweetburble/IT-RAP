import torch
import torch.nn as nn
import numpy as np

import torch_dct as dct
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

"""
Role: A class that comprehensively implements Diffusion-aware adversarial attacks.

Based on validated attack techniques from:
1. Mist/AdvDM (ICML 2023): Monte-Carlo score matching loss + VAE encoder attack
2. PhotoGuard (Salman et al. 2023): Encoder attack + diffusion attack for image immunization
3. DiffAttack (NeurIPS 2023): Deviated-reconstruction at intermediate timesteps

Key Insights from Literature:
- Mist: Attacks VAE encoder latent space + score function, ε=16/255, 100 steps
- PhotoGuard: Two-phase attack (encoder + diffusion), targets latent representations
- DCT alone is insufficient - must combine with latent space attacks

References:
- AdvDM/Mist: arXiv:2302.04578 (ICML 2023 Oral)
- PhotoGuard: arXiv:2302.06588
- DiffAttack: arXiv:2311.16124

Initialization:
epsilon: Maximum allowed noise size for the attack (based on L∞ norm).
alpha: Noise update size at each step.
"""
class AttackFunction(object):
    def __init__(self, config, diffusion_model, device=None, epsilon=0.05, alpha=0.01):
        self.model = diffusion_model
        self.model.model.eval()
        for p in self.model.model.parameters():
            p.requires_grad_(False)


        self.epsilon = max(epsilon, 0.08)
        self.alpha = max(alpha, 0.012)

        self.loss_fn = nn.MSELoss().to(device)
        self.lpips_loss = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(device)
        self.device = device

        self.config = config


        self.rand = True


        self.freq_mask_all = self.create_frequency_masks([1, 3, 256, 256], "ALL")
        self.freq_mask_low = self.create_frequency_masks([1, 3, 256, 256], "LOW")
        self.freq_mask_mid = self.create_frequency_masks([1, 3, 256, 256], "MID")
        self.freq_mask_high = self.create_frequency_masks([1, 3, 256, 256], "HIGH")


        self.num_segments = 4
        self.deviated_weight = 1.5


        self._mist_target = None


        self.score_attack_weight = 0.6
        self.textural_weight = 0.25
        self.semantic_weight = 0.15


        self.ssim_preserve_weight = 0.3

    def Diff_PGD(self, X_nat, target_img, target_attr, X_base=None):
        base = X_nat if X_base is None else X_base
        X = X_nat.clone().detach_()


        iter_count = max(self.config.pgd_iter, 12)


        max_timestep = getattr(self.config, 'diffusion_t0', 400)


        if target_attr:
            self.model.load_attr_weights(target_attr)


        if self._mist_target is None or self._mist_target.shape != X.shape:
            self._mist_target = self._create_mist_texture(X.shape).to(self.device)

        print(f"Starting Enhanced MIST Attack... (Iter: {iter_count}, ε: {self.epsilon:.4f}, t_0: {max_timestep})")

        for i in range(iter_count):
            X.requires_grad = True
            batch_size = X.shape[0]


            num_timesteps = 8

            timesteps = torch.linspace(int(max_timestep*0.1), max_timestep-1, num_timesteps).long().to(self.device)

            total_loss = 0

            for t_idx, t in enumerate(timesteps):
                t_batch = torch.ones(batch_size, dtype=torch.long, device=self.device) * t

                with torch.autocast(device_type="cuda", dtype=torch.float16):


                    noise = torch.randn_like(X).to(self.device)
                    sqrt_alpha_bar = self.model.sqrt_alphas_cumprod[t].view(batch_size, 1, 1, 1)
                    sqrt_one_minus_alpha_bar = self.model.sqrt_one_minus_alphas_cumprod[t].view(batch_size, 1, 1, 1)

                    x_t = sqrt_alpha_bar * X + sqrt_one_minus_alpha_bar * noise


                    epsilon_pred = self.model.ft_model(x_t, t_batch.float())
                    if self.model.config.data.dataset in ['FFHQ', 'AFHQ']:
                        epsilon_pred, _ = torch.split(epsilon_pred, 3, dim=1)

                    self.model.model.zero_grad()


                    score_loss = -torch.mean(epsilon_pred ** 2)


                    pred_x0 = (x_t - sqrt_one_minus_alpha_bar * epsilon_pred) / sqrt_alpha_bar
                    pred_x0 = torch.clamp(pred_x0, -1, 1)

                    deviated_loss = -self.loss_fn(pred_x0, target_img)


                    textural_loss = self.loss_fn(pred_x0, self._mist_target)


                    try:
                        lpips_val = -self.lpips_loss(pred_x0.clamp(-1, 1), target_img.clamp(-1, 1)).mean()
                    except:
                        lpips_val = torch.tensor(0.0, device=self.device)


                    t_normalized = t.float() / max_timestep
                    timestep_weight = 0.3 + 0.7 * (1 - (t_normalized - 0.5)**2 * 4)


                    loss_t = timestep_weight * (
                        self.score_attack_weight * score_loss +
                        self.textural_weight * textural_loss +
                        self.semantic_weight * (deviated_loss + lpips_val)
                    )
                    total_loss += loss_t

            total_loss = total_loss / num_timesteps
            total_loss = total_loss.float()

            total_loss.backward()


            grad = X.grad
            if grad is None:
                print(f"Warning: Gradient is None at iter {i}")
                break


            grad_magnitude = torch.abs(grad)

            adaptive_alpha = self.alpha * (1.0 + 0.5 * (grad_magnitude / (grad_magnitude.max() + 1e-8)))
            adaptive_alpha = torch.clamp(adaptive_alpha, self.alpha * 0.8, self.alpha * 1.5)

            X_adv = X + adaptive_alpha * grad.sign()


            eta = torch.clamp(X_adv - base, min=-self.epsilon, max=self.epsilon)
            X = torch.clamp(base + eta, min=-1, max=1).detach()

        return X, X - base

    def _create_mist_texture(self, shape):
        B, C, H, W = shape


        texture = torch.zeros(shape)


        for i in range(4):
            freq = 2 ** i
            noise = torch.randn(B, C, H // freq + 1, W // freq + 1)
            noise = nn.functional.interpolate(noise, size=(H, W), mode='bilinear', align_corners=True)
            texture += noise * (0.5 ** i)


        hf_noise = torch.randn(shape) * 0.3
        texture += hf_noise


        texture = (texture - texture.mean()) / (texture.std() + 1e-8)
        texture = torch.clamp(texture * 0.8, -1, 1)

        return texture

    def score_matching_attack(self, X_nat, target_attr, X_base=None):
        base = X_nat if X_base is None else X_base
        X = X_nat.clone().detach_()

        iter_count = max(getattr(self.config, 'score_attack_iter', 10), 10)
        max_timestep = getattr(self.config, 'diffusion_t0', 400)

        if target_attr:
            self.model.load_attr_weights(target_attr)

        print(f"Starting Pure Score Matching Attack... (Iter: {iter_count})")

        for i in range(iter_count):
            X.requires_grad = True
            batch_size = X.shape[0]


            K_timesteps = 6
            N_noise = 2

            total_loss = 0

            for k in range(K_timesteps):

                t = torch.randint(int(max_timestep * 0.2), max_timestep, (1,), device=self.device).long()
                t_batch = t.repeat(batch_size)

                sqrt_alpha_bar = self.model.sqrt_alphas_cumprod[t].view(batch_size, 1, 1, 1)
                sqrt_one_minus_alpha_bar = self.model.sqrt_one_minus_alphas_cumprod[t].view(batch_size, 1, 1, 1)

                for n in range(N_noise):
                    with torch.autocast(device_type="cuda", dtype=torch.float16):

                        noise = torch.randn_like(X).to(self.device)


                        x_t = sqrt_alpha_bar * X + sqrt_one_minus_alpha_bar * noise


                        epsilon_pred = self.model.ft_model(x_t, t_batch.float())
                        if self.model.config.data.dataset in ['FFHQ', 'AFHQ']:
                            epsilon_pred, _ = torch.split(epsilon_pred, 3, dim=1)

                        self.model.model.zero_grad()


                        score_loss = -torch.mean(epsilon_pred ** 2)


                        deviation_loss = -torch.mean((epsilon_pred - noise) ** 2) * 0.3

                        total_loss += (score_loss + deviation_loss)

            total_loss = total_loss / (K_timesteps * N_noise)
            total_loss = total_loss.float()
            total_loss.backward()

            grad = X.grad
            if grad is None:
                print(f"Warning: Gradient is None at iter {i}")
                break


            X_adv = X + self.alpha * 1.2 * grad.sign()


            eta = torch.clamp(X_adv - base, min=-self.epsilon, max=self.epsilon)
            X = torch.clamp(base + eta, min=-1, max=1).detach()

        return X, X - base

    def attack_inversion_phase(self, X_nat, target_attr, X_base=None, iter_count=None):
        base = X_nat if X_base is None else X_base
        X = X_nat.clone().detach_()


        iter_count = iter_count or max(getattr(self.config, 'inv_attack_iter', 4), 6)


        t_0 = getattr(self.config, 'diffusion_t0', 200)
        n_inv_step = min(getattr(self.config, 'n_inv_step', 40), 20)

        if target_attr:
            self.model.load_attr_weights(target_attr)

        print(f"Starting Enhanced Inversion Attack... (Iter: {iter_count}, n_inv_step: {n_inv_step})")


        fd_eps = 0.015


        with torch.no_grad():
            clean_latent = self._compute_inversion_latent_fast(base, t_0, n_inv_step)


            target_latent = torch.randn_like(clean_latent) * 0.5

        for i in range(iter_count):

            with torch.no_grad():
                current_latent = self._compute_inversion_latent_fast(X, t_0, n_inv_step)


            grad = torch.zeros_like(X)


            num_samples = 6

            for _ in range(num_samples):

                direction = torch.randn_like(X)
                direction = direction / (direction.norm() + 1e-8)


                with torch.no_grad():
                    X_plus = torch.clamp(X + fd_eps * direction, -1, 1)
                    latent_plus = self._compute_inversion_latent_fast(X_plus, t_0, n_inv_step)


                    deviation_loss = -torch.mean((latent_plus - clean_latent) ** 2)


                    target_loss = torch.mean((latent_plus - target_latent) ** 2)


                    magnitude_loss = -torch.mean(latent_plus ** 2)


                    variance_loss = -torch.var(latent_plus)

                    loss_plus = 0.4 * deviation_loss + 0.3 * target_loss + 0.2 * magnitude_loss + 0.1 * variance_loss


                    X_minus = torch.clamp(X - fd_eps * direction, -1, 1)
                    latent_minus = self._compute_inversion_latent_fast(X_minus, t_0, n_inv_step)

                    deviation_loss_m = -torch.mean((latent_minus - clean_latent) ** 2)
                    target_loss_m = torch.mean((latent_minus - target_latent) ** 2)
                    magnitude_loss_m = -torch.mean(latent_minus ** 2)
                    variance_loss_m = -torch.var(latent_minus)

                    loss_minus = 0.4 * deviation_loss_m + 0.3 * target_loss_m + 0.2 * magnitude_loss_m + 0.1 * variance_loss_m


                    grad_estimate = (loss_plus - loss_minus) / (2 * fd_eps)
                    grad += grad_estimate * direction

            grad = grad / num_samples


            X_adv = X + self.alpha * 2.5 * grad.sign()


            eta = torch.clamp(X_adv - base, min=-self.epsilon, max=self.epsilon)
            X = torch.clamp(base + eta, min=-1, max=1).detach()

            torch.cuda.empty_cache()

        return X, X - base

    def _compute_inversion_latent_fast(self, x, t_0, n_inv_step):
        n = x.shape[0]

        with torch.no_grad():

            seq_inv = np.linspace(0, 1, n_inv_step) * t_0
            seq_inv = [int(s) for s in list(seq_inv)]
            seq_inv_next = [-1] + list(seq_inv[:-1])

            x_t = x.clone()

            for idx, (i, j) in enumerate(zip(seq_inv_next[1:], seq_inv[1:])):
                t = (torch.ones(n) * i).to(self.device)


                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    out = self.model.base_model(x_t, t.float())

                    if self.model.config.data.dataset in ['FFHQ', 'AFHQ']:
                        epsilon_pred, _ = torch.split(out, 3, dim=1)
                    else:
                        epsilon_pred = out


                    alpha_t = self.model.sqrt_alphas_cumprod[max(0, i)] ** 2
                    alpha_next = self.model.sqrt_alphas_cumprod[max(0, j)] ** 2


                    sqrt_alpha_ratio = torch.sqrt(alpha_next / (alpha_t + 1e-8))
                    x_t = (sqrt_alpha_ratio * x_t +
                           (torch.sqrt(1 - alpha_next) - sqrt_alpha_ratio * torch.sqrt(1 - alpha_t)) * epsilon_pred)

            return x_t

    """
        Role: Performs a basic I-FGSM attack.

        Operation:
        1. Use the original image.
        2. Calculate gradient and update noise.
        3. At each step, limit the noise size to within epsilon and clip the image pixel values to [-1, 1].
    """
    def PGD(self, X_nat, target_img, target_attr):
        X = X_nat.clone().detach_()

        iter_count = self.config.pgd_iter

        for i in range(iter_count):
            X.requires_grad = True


            output = self.model.forward_edit(X, target_attr,
                                            t_0=200, n_inv_step=10, n_test_step=10)


            self.model.model.zero_grad()


            loss = self.loss_fn(output, target_img)

            print(f"Iter {i+1}/{iter_count}, Loss: {loss.item():.4f}")

            loss.backward()


            grad = X.grad
            if grad is None:
                print("Error: Gradient is None. Check if the model allows gradient flow.")
                break


            X_adv = X - self.alpha * grad.sign()


            eta = torch.clamp(X_adv - X_nat, min=-self.epsilon, max=self.epsilon)
            X = torch.clamp(X_nat + eta, min=-1, max=1).detach()

        return X, X - X_nat


    def perturb_frequency_domain(self, X_nat, target_attr, freq_band='ALL', iter=1, X_base=None, original_gen_image=None):


        dct_clamp = max(getattr(self.config, 'dct_clamp', 0.1), 0.15)
        iter = max(iter, 3)

        base = X_nat if X_base is None else X_base


        X_nat_dct = torch.zeros_like(base)
        for b in range(base.shape[0]):
            for c in range(base.shape[1]):
                X_nat_dct[b, c] = dct.dct_2d(base[b, c])


        if freq_band == 'ALL': freq_mask = self.freq_mask_all
        elif freq_band == 'LOW': freq_mask = self.freq_mask_low
        elif freq_band == 'MID': freq_mask = self.freq_mask_mid
        elif freq_band == 'HIGH': freq_mask = self.freq_mask_high
        else: raise ValueError(f"Unknown band: {freq_band}")

        if X_nat.shape[0] != freq_mask.shape[0]:
            freq_mask = freq_mask[:1].repeat(X_nat.shape[0], 1, 1, 1)


        eta_dct = torch.zeros_like(X_nat_dct)


        if freq_band == 'LOW':

            eta_dct = torch.randn_like(X_nat_dct) * 0.02 * freq_mask
        elif freq_band == 'MID':

            eta_dct = torch.randn_like(X_nat_dct) * 0.015 * freq_mask
        elif freq_band == 'HIGH':

            eta_dct = torch.randn_like(X_nat_dct) * 0.01 * freq_mask
        else:
            eta_dct = torch.randn_like(X_nat_dct) * 0.015 * freq_mask

        print(f"[*] Starting Enhanced Frequency Attack on band '{freq_band}' (iter: {iter})")

        if target_attr:
            self.model.load_attr_weights(target_attr)

        for i in range(iter):


            X_dct_current = X_nat_dct + eta_dct
            X_current = torch.zeros_like(base)
            for b in range(base.shape[0]):
                for c in range(base.shape[1]):
                    X_current[b, c] = dct.idct_2d(X_dct_current[b, c])
            X_current = torch.clamp(X_current, min=-1, max=1)


            X_current.requires_grad = True


            with torch.autocast(device_type="cuda", dtype=torch.float16):

                t = torch.tensor([100], device=self.device)
                pred_x0 = self.model.predict_x0_from_xt(X_current, t, use_autocast=True)
                self.model.model.zero_grad()


                if original_gen_image is not None:
                    deviated_loss = -torch.mean((pred_x0 - original_gen_image) ** 2)
                else:
                    deviated_loss = -torch.mean((pred_x0 - base) ** 2)


                if self._mist_target is not None and self._mist_target.shape == pred_x0.shape:
                    texture_loss = torch.mean((pred_x0 - self._mist_target) ** 2) * 0.3
                else:
                    texture_loss = 0


                variance_loss = -torch.var(pred_x0) * 0.2


                total_loss = deviated_loss + texture_loss + variance_loss

            total_loss = total_loss.float()
            total_loss.backward()


            grad_img = X_current.grad
            if grad_img is None:
                print(f"  [!] No gradient at iter {i}")
                continue


            grad_dct = torch.zeros_like(eta_dct)
            for b in range(base.shape[0]):
                for c in range(base.shape[1]):
                    grad_dct[b, c] = dct.dct_2d(grad_img[b, c])


            grad_dct = grad_dct * freq_mask


            eta_dct = eta_dct + self.alpha * 3.0 * grad_dct.sign()
            eta_dct = torch.clamp(eta_dct, min=-dct_clamp, max=dct_clamp)
            eta_dct = eta_dct * freq_mask

            X_current = X_current.detach()
            torch.cuda.empty_cache()


        with torch.no_grad():
            X_dct_final = X_nat_dct + eta_dct
            X_final = torch.zeros_like(base)
            for b in range(base.shape[0]):
                for c in range(base.shape[1]):
                    X_final[b, c] = dct.idct_2d(X_dct_final[b, c])
            X_final = torch.clamp(X_final, min=-1, max=1)


            eta_pix = torch.clamp(X_final - base, min=-self.epsilon, max=self.epsilon)
            X_final = torch.clamp(base + eta_pix, min=-1, max=1)

        return X_final, X_final - base


    def create_frequency_masks(self, shape, freq_band='ALL'):
        B, C, H, W = shape
        masks = torch.zeros(shape).to(self.device)


        if freq_band == 'ALL':
            return torch.ones(shape).to(self.device)


        for b in range(B):
            for c in range(C):


                i_coords = torch.arange(H).reshape(-1, 1).repeat(1, W).to(self.device)
                j_coords = torch.arange(W).reshape(1, -1).repeat(H, 1).to(self.device)


                frequency_map = torch.sqrt(i_coords**2 + j_coords**2)


                max_freq = torch.sqrt(torch.tensor((H-1)**2 + (W-1)**2)).to(self.device)
                frequency_map = frequency_map / max_freq


                if freq_band == 'LOW':
                    masks[b, c] = (frequency_map <= 1/3).float()
                elif freq_band == 'MID':
                    masks[b, c] = ((frequency_map > 1/3) & (frequency_map <= 2/3)).float()
                elif freq_band == 'HIGH':
                    masks[b, c] = (frequency_map > 2/3).float()

        return masks
