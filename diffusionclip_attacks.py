import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_dct as dct
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


def gaussian_blur(x, kernel_size=5, sigma=1.0):
    """Apply Gaussian blur to tensor for smooth perturbation."""
    channels = x.shape[1]
    
    # Create Gaussian kernel
    coords = torch.arange(kernel_size, dtype=torch.float32, device=x.device) - (kernel_size - 1) / 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    
    # Create 2D kernel
    kernel = g.view(1, -1) * g.view(-1, 1)
    kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)
    
    # Apply padding
    padding = kernel_size // 2
    x_padded = F.pad(x, (padding, padding, padding, padding), mode='reflect')
    
    return F.conv2d(x_padded, kernel, groups=channels)


"""
Role: A class that comprehensively implements adversarial attacks for DiffusionCLIP.

Key change: All attacks now MAXIMIZE loss to make DiffusionCLIP output differ from original,
which increases ASR (Attack Success Rate).

NEW: Smooth perturbation techniques for better invisibility while maintaining high ASR.

Initialization:
epsilon: Maximum allowed perturbation size (L∞ norm) - controls invisibility
alpha: Step size for each iteration
"""
class AttackFunction(object):
    def __init__(self, config, diffusion_model, device=None, epsilon=0.05, alpha=0.01):
        """
        diffusion_model: DiffusionCLIPWrapper 인스턴스
        config에서 attack_epsilon, attack_alpha를 읽어 우선 적용
        """
        self.model = diffusion_model
        self.device = device
        self.config = config
        
        # Config에서 epsilon/alpha 읽기 (비가시성 제어 핵심)
        self.epsilon = getattr(config, 'attack_epsilon', epsilon)
        self.alpha = getattr(config, 'attack_alpha', alpha)
        
        self.loss_fn = nn.MSELoss().to(device)

        # Select PGD or I-FGSM (whether to use random noise for PGD initialization)
        self.rand = True

        # Frequency mask member variables (initialize and reuse)
        self.freq_mask_all = self.create_frequency_masks([1, 3, 256, 256], "ALL")
        self.freq_mask_low = self.create_frequency_masks([1, 3, 256, 256], "LOW")
        self.freq_mask_mid = self.create_frequency_masks([1, 3, 256, 256], "MID")
        self.freq_mask_high = self.create_frequency_masks([1, 3, 256, 256], "HIGH")
        
        # Perceptual & TV regularization weights (to trade off invisibility)
        self.perceptual_weight = getattr(config, 'attack_perceptual_weight', 0.6)
        self.tv_weight = getattr(config, 'attack_tv_weight', 0.01)
        
        # Smooth attack parameters (NEW)
        self.smooth_kernel_size = getattr(config, 'smooth_kernel_size', 5)
        self.smooth_sigma = getattr(config, 'smooth_sigma', 1.5)
        self.momentum_decay = getattr(config, 'momentum_decay', 0.9)

        # LPIPS metric for input-space perceptual regularization
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=False).to(self.device)

    def Diff_PGD(self, X_nat, target_img, target_attr):
        """
        End-to-End DiffusionCLIP Attack (DiffAttack-inspired)
        최적화: 더 적은 iteration으로 동일 효과, print 제거
        """
        X = X_nat.clone().detach_()
        iter_count = self.config.pgd_iter
        
        # 최적화된 step 수 (기존 6-12 -> 4-8로 축소해도 효과 유지)
        t_0 = getattr(self.config, 't_0', 400)
        n_step = max(4, min(8, t_0 // 50))  # 더 적은 step으로 효율화
        
        if target_attr:
            self.model.load_attr_weights(target_attr)
        
        momentum = torch.zeros_like(X)

        for i in range(iter_count):
            X.requires_grad = True
            
            output = self.model.forward_edit(
                X, target_attr, 
                t_0=t_0, 
                n_inv_step=n_step,
                n_test_step=n_step,
                require_grad=True
            )
            
            self.model.model.zero_grad()
            
            # Defense loss
            loss_mse = self.loss_fn(output, target_img)
            loss_lpips = self.lpips(output.clamp(-1, 1), target_img.clamp(-1, 1)).mean()
            loss_def = loss_mse + 0.5 * loss_lpips
            
            percept = self.lpips(X.clamp(-1, 1), X_nat.clamp(-1, 1)).mean()
            
            eta = X - X_nat
            tv = torch.mean(torch.abs(eta[:, :, 1:, :] - eta[:, :, :-1, :])) + \
                 torch.mean(torch.abs(eta[:, :, :, 1:] - eta[:, :, :, :-1]))

            objective = loss_def - 0.3 * percept - 0.005 * tv
            objective.backward()

            grad = X.grad
            if grad is None:
                break

            grad_smooth = gaussian_blur(grad, kernel_size=5, sigma=0.9)
            grad_l2 = torch.sqrt(torch.sum(grad_smooth ** 2, dim=[1,2,3], keepdim=True) + 1e-12)
            grad_normalized = grad_smooth / grad_l2
            
            momentum = 0.9 * momentum + grad_normalized
            
            step_size = self.alpha * 30.0  # 더 큰 step으로 적은 iter에서 동일 효과
            X_adv = X + step_size * momentum

            eta = X_adv - X_nat
            eta_smooth = gaussian_blur(eta, kernel_size=3, sigma=0.6)
            eta_smooth = torch.clamp(eta_smooth, min=-self.epsilon, max=self.epsilon)
            
            X = torch.clamp(X_nat + eta_smooth, min=-1, max=1).detach()
            momentum = momentum.detach()

        return X, X - X_nat

    def PGD(self, X_nat, target_img, target_attr):
        """
        Full-loop PGD Attack with Smooth Perturbation
        최적화: print 제거, 더 적은 step으로 효율화
        """
        X = X_nat.clone().detach_()
        iter_count = self.config.pgd_iter
        
        t_0 = getattr(self.config, 't_0', 400)
        n_step = max(4, min(8, t_0 // 50))  # 더 적은 step
        
        momentum = torch.zeros_like(X)

        for i in range(iter_count):
            X.requires_grad = True
            
            output = self.model.forward_edit(X, target_attr, 
                                            t_0=t_0, n_inv_step=n_step, n_test_step=n_step,
                                            require_grad=True)
            self.model.model.zero_grad()

            loss_mse = self.loss_fn(output, target_img)
            loss_lpips_def = self.lpips(output.clamp(-1, 1), target_img.clamp(-1, 1)).mean()
            loss_def = loss_mse + 0.5 * loss_lpips_def
            
            percept = self.lpips(X.clamp(-1, 1), X_nat.clamp(-1, 1)).mean()
            eta_sp = X - X_nat
            tv = torch.mean(torch.abs(eta_sp[:, :, 1:, :] - eta_sp[:, :, :-1, :])) + \
                 torch.mean(torch.abs(eta_sp[:, :, :, 1:] - eta_sp[:, :, :, :-1]))

            objective = loss_def - 0.3 * percept - 0.005 * tv
            objective.backward()

            grad = X.grad
            if grad is None:
                break

            grad_smooth = gaussian_blur(grad, kernel_size=5, sigma=0.9)
            grad_l2 = torch.sqrt(torch.sum(grad_smooth ** 2, dim=[1,2,3], keepdim=True) + 1e-12)
            grad_normalized = grad_smooth / grad_l2
            
            momentum = 0.9 * momentum + grad_normalized
            
            step_size = self.alpha * 30.0  # 더 큰 step으로 적은 iter에서 동일 효과
            X_adv = X + step_size * momentum

            eta = X_adv - X_nat
            eta_smooth = gaussian_blur(eta, kernel_size=3, sigma=0.6)
            eta_smooth = torch.clamp(eta_smooth, min=-self.epsilon, max=self.epsilon)
            
            X = torch.clamp(X_nat + eta_smooth, min=-1, max=1).detach()
            momentum = momentum.detach()

        return X, X - X_nat


    def perturb_frequency_domain(self, X_nat, target_attr, freq_band='ALL', iter=1):
        """Low-Frequency Focused Attack: 최적화된 버전"""
        
        dct_clamp = getattr(self.config, 'dct_clamp', 0.15)
        dct_iter = getattr(self.config, 'dct_iter', 4)
        t_0 = getattr(self.config, 't_0', 400)
        
        # 1. 원본 이미지의 DCT 계수 계산
        X_nat_dct = torch.zeros_like(X_nat)
        for b in range(X_nat.shape[0]):
            for c in range(X_nat.shape[1]):
                X_nat_dct[b, c] = dct.dct_2d(X_nat[b, c])
        
        # 2. 마스크 선택 (LOW frequency 권장 - 비가시성 최고)
        if freq_band == 'ALL': freq_mask = self.freq_mask_all
        elif freq_band == 'LOW': freq_mask = self.freq_mask_low
        elif freq_band == 'MID': freq_mask = self.freq_mask_mid
        elif freq_band == 'HIGH': freq_mask = self.freq_mask_high
        else: raise ValueError(f"Unknown band: {freq_band}")
        
        if X_nat.shape[0] != freq_mask.shape[0]:
            freq_mask = freq_mask[:1].repeat(X_nat.shape[0], 1, 1, 1)

        # 3. 노이즈 초기화 및 Momentum
        eta_dct = torch.zeros_like(X_nat_dct).to(self.device)
        eta_dct.requires_grad = True
        momentum_dct = torch.zeros_like(eta_dct)

        for i in range(dct_iter):
            current_eta = eta_dct * freq_mask
            X_dct = X_nat_dct + current_eta
            
            X_adv_input = torch.zeros_like(X_nat)
            for b in range(X_nat.shape[0]):
                for c in range(X_nat.shape[1]):
                    X_adv_input[b, c] = dct.idct_2d(X_dct[b, c])
            
            X_adv_input = torch.clamp(X_adv_input, min=-1, max=1)

            batch_size = X_adv_input.shape[0]
            t = torch.randint(low=max(1, t_0//4), high=t_0, size=(batch_size,), device=self.device).long()
            
            if target_attr:
                self.model.load_attr_weights(target_attr)
            
            pred_x0 = self.model.predict_x0_from_xt(X_adv_input, t)
            self.model.model.zero_grad()

            loss_mse = self.loss_fn(pred_x0, X_nat)
            loss_lpips_def = self.lpips(pred_x0.clamp(-1, 1), X_nat.clamp(-1, 1)).mean()
            loss_def = loss_mse + 0.4 * loss_lpips_def
            
            percept = self.lpips(X_adv_input.clamp(-1, 1), X_nat.clamp(-1, 1)).mean()
            eta_sp = X_adv_input - X_nat
            tv = torch.mean(torch.abs(eta_sp[:, :, 1:, :] - eta_sp[:, :, :-1, :])) + \
                 torch.mean(torch.abs(eta_sp[:, :, :, 1:] - eta_sp[:, :, :, :-1]))

            objective = loss_def - 0.25 * percept - 0.003 * tv
            objective.backward()
            
            with torch.no_grad():
                grad = eta_dct.grad
                if grad is not None:
                    grad = grad * freq_mask
                    grad_l2 = torch.sqrt(torch.sum(grad ** 2) + 1e-12)
                    grad_normalized = grad / grad_l2
                    
                    momentum_dct = 0.9 * momentum_dct + grad_normalized
                    step_size = self.alpha * 25.0  # 더 큰 step으로 적은 iter에서 동일 효과
                    eta_dct = eta_dct + step_size * momentum_dct
                    eta_dct = torch.clamp(eta_dct, min=-dct_clamp, max=dct_clamp)
                
                eta_dct = eta_dct.clone().detach().requires_grad_(True)

        # Finalize with additional smoothing
        with torch.no_grad():
            final_eta = eta_dct * freq_mask
            X_dct_final = X_nat_dct + final_eta
            X_final = torch.zeros_like(X_nat)
            for b in range(X_nat.shape[0]):
                for c in range(X_nat.shape[1]):
                    X_final[b, c] = dct.idct_2d(X_dct_final[b, c])
            X_final = torch.clamp(X_final, min=-1, max=1)
            
            # Apply final Gaussian smoothing for extra invisibility
            perturbation = X_final - X_nat
            perturbation_smooth = gaussian_blur(perturbation, kernel_size=3, sigma=0.5)
            perturbation_smooth = torch.clamp(perturbation_smooth, min=-self.epsilon, max=self.epsilon)
            X_final = torch.clamp(X_nat + perturbation_smooth, min=-1, max=1)

        return X_final, X_final - X_nat


    def create_frequency_masks(self, shape, freq_band='ALL'):
        """
        Function to create masks for each frequency band.
        
        Args:
            shape: Shape of the mask (batch, channel, height, width)
            freq_band: Frequency band to select ('LOW', 'MID', 'HIGH', 'ALL')
        
        Returns:
            Mask for the selected frequency band (1: selected area, 0: unselected area)
        """
        B, C, H, W = shape
        masks = torch.zeros(shape).to(self.device)
        
        # Handle the 'ALL' frequency band case quickly
        if freq_band == 'ALL':
            return torch.ones(shape).to(self.device)
        
        # Create a mask for each image and channel
        for b in range(B):
            for c in range(C):
                # In DCT, (0,0) is the DC component (lowest frequency)
                # The distance from the origin approximates the frequency
                i_coords = torch.arange(H).reshape(-1, 1).repeat(1, W).to(self.device)
                j_coords = torch.arange(W).reshape(1, -1).repeat(H, 1).to(self.device)
                
                # Calculate frequency map (distance from the origin)
                frequency_map = torch.sqrt(i_coords**2 + j_coords**2)
                
                # Normalize the frequency map
                max_freq = torch.sqrt(torch.tensor((H-1)**2 + (W-1)**2)).to(self.device)
                frequency_map = frequency_map / max_freq
                
                # Divide frequency bands (into 3 parts)
                if freq_band == 'LOW':
                    masks[b, c] = (frequency_map <= 1/3).float()
                elif freq_band == 'MID':
                    masks[b, c] = ((frequency_map > 1/3) & (frequency_map <= 2/3)).float()
                elif freq_band == 'HIGH':
                    masks[b, c] = (frequency_map > 2/3).float()
        
        return masks
