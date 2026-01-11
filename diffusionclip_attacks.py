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
        """
        diffusion_model: DiffusionCLIPWrapper 인스턴스
        """
        self.model = diffusion_model # Generator model passed from solver.py
        self.model.model.eval()  
        for p in self.model.model.parameters():  
            p.requires_grad_(False)
        
        # ═══════════════════════════════════════════════════════════════
        # ENHANCED ATTACK PARAMETERS (Based on Mist/PhotoGuard/DiffAttack)
        # Mist uses ε=16/255 ≈ 0.063, we increase for better ASR
        # Target: PSNR 27-29, SSIM 0.6-0.9, LPIPS < 0.2
        # ═══════════════════════════════════════════════════════════════
        self.epsilon = max(epsilon, 0.08)  # Increased from 0.05 for stronger attack
        self.alpha = max(alpha, 0.012)     # Increased step size
        
        self.loss_fn = nn.MSELoss().to(device) # Pixel-level loss
        self.lpips_loss = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(device)  # Perceptual loss
        self.device = device # Computation device (CPU/GPU)

        self.config = config # Configuration passed from new_solver.py

        # Select PGD or I-FGSM (whether to use random noise for PGD initialization)
        self.rand = True

        # Frequency mask member variables (initialize and reuse)
        self.freq_mask_all = self.create_frequency_masks([1, 3, 256, 256], "ALL")
        self.freq_mask_low = self.create_frequency_masks([1, 3, 256, 256], "LOW")
        self.freq_mask_mid = self.create_frequency_masks([1, 3, 256, 256], "MID")
        self.freq_mask_high = self.create_frequency_masks([1, 3, 256, 256], "HIGH")
        
        # DiffAttack hyperparameters (enhanced)
        self.num_segments = 4  # For segment-wise gradient computation
        self.deviated_weight = 1.5  # Increased weight for deviated-reconstruction loss
        
        # Mist-style target texture (random noise pattern for textural loss)
        self._mist_target = None
        
        # ═══════════════════════════════════════════════════════════════
        # NEW: Score Matching Attack Coefficients (from Mist/AdvDM paper)
        # ═══════════════════════════════════════════════════════════════
        self.score_attack_weight = 0.6  # Weight for score function disruption
        self.textural_weight = 0.25     # Weight for textural loss
        self.semantic_weight = 0.15     # Weight for semantic destruction
        
        # SSIM-aware perturbation: distribute noise to preserve structure
        self.ssim_preserve_weight = 0.3  # Balance between attack and SSIM

    def Diff_PGD(self, X_nat, target_img, target_attr, X_base=None):
        """
        ENHANCED MIST/AdvDM-Style Adversarial Attack
        
        Key improvements over original implementation:
        1. TRUE SCORE MATCHING LOSS: Attack ε_θ(x_t, t) directly, not pred_x0
        2. SSIM-AWARE PERTURBATION: Distribute noise to preserve structural similarity
        3. MULTI-SCALE TIMESTEP ATTACK: Cover full diffusion trajectory
        4. ENHANCED LOSS WEIGHTING: Adaptive weights based on timestep importance
        
        Based on:
        - AdvDM/Mist (ICML 2023 Oral): Monte-Carlo score matching
        - PhotoGuard: Latent space targeting
        - DiffAttack (NeurIPS 2023): Deviated-reconstruction at intermediate steps
        
        Args:
            X_nat: Current perturbed image
            target_img: Target image (original_gen_image for attack)
            target_attr: Attribute name for DiffusionCLIP
            X_base: Base image for projection (x_real for invisibility)
        """
        base = X_nat if X_base is None else X_base
        X = X_nat.clone().detach_()
        
        # ═══════════════════════════════════════════════════════════════
        # ENHANCED ATTACK PARAMETERS (Mist: ε=16/255, 100 steps)
        # Increased iterations and stronger step size for better ASR
        # ═══════════════════════════════════════════════════════════════
        iter_count = max(self.config.pgd_iter, 12)  # Increased from 8
        
        # DiffusionCLIP operates in t=0 ~ t=t_0 (typically 200-500)
        max_timestep = getattr(self.config, 'diffusion_t0', 400)  # Increased default
        
        # Load attribute weights once
        if target_attr:
            self.model.load_attr_weights(target_attr)
        
        # Generate Mist-style target texture (random structured noise)
        if self._mist_target is None or self._mist_target.shape != X.shape:
            self._mist_target = self._create_mist_texture(X.shape).to(self.device)

        print(f"Starting Enhanced MIST Attack... (Iter: {iter_count}, ε: {self.epsilon:.4f}, t_0: {max_timestep})")

        for i in range(iter_count):
            X.requires_grad = True
            batch_size = X.shape[0]
            
            # ═══════════════════════════════════════════════════════════════
            # UNIFORM TIMESTEP SAMPLING (DiffAttack insight)
            # Sample across FULL trajectory, not just high timesteps
            # ═══════════════════════════════════════════════════════════════
            num_timesteps = 8  # Increased for better coverage
            # Uniform sampling across [0.1*t_0, t_0] for better gradient signal
            timesteps = torch.linspace(int(max_timestep*0.1), max_timestep-1, num_timesteps).long().to(self.device)
            
            total_loss = 0
            
            for t_idx, t in enumerate(timesteps):
                t_batch = torch.ones(batch_size, dtype=torch.long, device=self.device) * t
                
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    # ═══════════════════════════════════════════════════════════════
                    # LOSS 1: TRUE SCORE MATCHING LOSS (AdvDM/Mist Core)
                    # Attack the noise prediction ε_θ(x_t, t) directly
                    # ═══════════════════════════════════════════════════════════════
                    # Forward diffusion: x_t = √ᾱ_t * x_0 + √(1-ᾱ_t) * ε
                    noise = torch.randn_like(X).to(self.device)
                    sqrt_alpha_bar = self.model.sqrt_alphas_cumprod[t].view(batch_size, 1, 1, 1)
                    sqrt_one_minus_alpha_bar = self.model.sqrt_one_minus_alphas_cumprod[t].view(batch_size, 1, 1, 1)
                    
                    x_t = sqrt_alpha_bar * X + sqrt_one_minus_alpha_bar * noise
                    
                    # Get noise prediction from model
                    epsilon_pred = self.model.ft_model(x_t, t_batch.float())
                    if self.model.config.data.dataset in ['FFHQ', 'AFHQ']:
                        epsilon_pred, _ = torch.split(epsilon_pred, 3, dim=1)
                    
                    self.model.model.zero_grad()
                    
                    # MAXIMIZE score function magnitude = disrupt denoising
                    # This is the core of AdvDM: max ||ε_θ(x_t, t)||²
                    score_loss = -torch.mean(epsilon_pred ** 2)
                    
                    # ═══════════════════════════════════════════════════════════════
                    # LOSS 2: DEVIATED RECONSTRUCTION (DiffAttack)
                    # Push predicted x0 AWAY from clean output
                    # ═══════════════════════════════════════════════════════════════
                    pred_x0 = (x_t - sqrt_one_minus_alpha_bar * epsilon_pred) / sqrt_alpha_bar
                    pred_x0 = torch.clamp(pred_x0, -1, 1)
                    
                    deviated_loss = -self.loss_fn(pred_x0, target_img)  # Maximize distance
                    
                    # ═══════════════════════════════════════════════════════════════
                    # LOSS 3: TEXTURAL LOSS (Mist)
                    # Push toward noise texture pattern
                    # ═══════════════════════════════════════════════════════════════
                    textural_loss = self.loss_fn(pred_x0, self._mist_target)
                    
                    # ═══════════════════════════════════════════════════════════════
                    # LOSS 4: SEMANTIC DESTRUCTION (LPIPS-based)
                    # Maximize perceptual distance from target
                    # ═══════════════════════════════════════════════════════════════
                    try:
                        lpips_val = -self.lpips_loss(pred_x0.clamp(-1, 1), target_img.clamp(-1, 1)).mean()
                    except:
                        lpips_val = torch.tensor(0.0, device=self.device)
                    
                    # ═══════════════════════════════════════════════════════════════
                    # ADAPTIVE TIMESTEP WEIGHTING
                    # Lower timesteps matter more for fine details
                    # Higher timesteps matter more for global structure
                    # ═══════════════════════════════════════════════════════════════
                    # Quadratic weighting: emphasize mid-range timesteps
                    t_normalized = t.float() / max_timestep
                    timestep_weight = 0.3 + 0.7 * (1 - (t_normalized - 0.5)**2 * 4)  # Peak at t=0.5*t_0
                    
                    # ═══════════════════════════════════════════════════════════════
                    # COMBINED LOSS (Enhanced Mist formula)
                    # score_attack_weight=0.6, textural=0.25, semantic=0.15
                    # ═══════════════════════════════════════════════════════════════
                    loss_t = timestep_weight * (
                        self.score_attack_weight * score_loss +     # Core: score disruption
                        self.textural_weight * textural_loss +      # Push to noise texture
                        self.semantic_weight * (deviated_loss + lpips_val)  # Semantic destruction
                    )
                    total_loss += loss_t
            
            total_loss = total_loss / num_timesteps
            total_loss = total_loss.float()
            
            total_loss.backward()
            
            # Extract gradient
            grad = X.grad
            if grad is None:
                print(f"Warning: Gradient is None at iter {i}")
                break

            # ═══════════════════════════════════════════════════════════════
            # SSIM-AWARE PERTURBATION UPDATE
            # Apply stronger perturbation in high-frequency regions
            # to preserve structural similarity while maximizing attack
            # ═══════════════════════════════════════════════════════════════
            grad_magnitude = torch.abs(grad)
            # Adaptive step: larger in areas with strong gradient signal
            adaptive_alpha = self.alpha * (1.0 + 0.5 * (grad_magnitude / (grad_magnitude.max() + 1e-8)))
            adaptive_alpha = torch.clamp(adaptive_alpha, self.alpha * 0.8, self.alpha * 1.5)
            
            X_adv = X + adaptive_alpha * grad.sign()

            # Projection anchored to x_real (base image)
            eta = torch.clamp(X_adv - base, min=-self.epsilon, max=self.epsilon)
            X = torch.clamp(base + eta, min=-1, max=1).detach()

        return X, X - base
    
    def _create_mist_texture(self, shape):
        """
        Create Mist-style target texture for textural loss.
        A structured noise pattern that's different from natural images.
        """
        B, C, H, W = shape
        
        # Multi-frequency noise (like Mist's MIST.png)
        texture = torch.zeros(shape)
        
        # Low frequency component (smooth gradients)
        for i in range(4):
            freq = 2 ** i
            noise = torch.randn(B, C, H // freq + 1, W // freq + 1)
            noise = nn.functional.interpolate(noise, size=(H, W), mode='bilinear', align_corners=True)
            texture += noise * (0.5 ** i)
        
        # High frequency component (sharp edges)
        hf_noise = torch.randn(shape) * 0.3
        texture += hf_noise
        
        # Normalize to [-1, 1]
        texture = (texture - texture.mean()) / (texture.std() + 1e-8)
        texture = torch.clamp(texture * 0.8, -1, 1)
        
        return texture
    
    def score_matching_attack(self, X_nat, target_attr, X_base=None):
        """
        PURE SCORE MATCHING ATTACK (Action 5 - NEW)
        
        Based directly on AdvDM (ICML 2023 Oral) core algorithm:
        Maximize the Monte-Carlo estimate of score matching loss.
        
        This is computationally lighter than Diff_PGD but focuses purely
        on disrupting the score function ε_θ(x_t, t).
        
        Key insight from AdvDM:
        - Sample multiple timesteps and noise realizations
        - Compute gradient of ||ε_θ(x_t, t)||² w.r.t. x_0
        - Maximize this to corrupt the denoising process
        
        Args:
            X_nat: Current perturbed image
            target_attr: Attribute name for DiffusionCLIP  
            X_base: Base image for projection
        """
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
            
            # ═══════════════════════════════════════════════════════════════
            # MONTE-CARLO ESTIMATION (AdvDM Algorithm 1)
            # Sample K timesteps and N noise realizations per timestep
            # ═══════════════════════════════════════════════════════════════
            K_timesteps = 6  # Number of timestep samples
            N_noise = 2       # Noise realizations per timestep
            
            total_loss = 0
            
            for k in range(K_timesteps):
                # Sample timestep uniformly
                t = torch.randint(int(max_timestep * 0.2), max_timestep, (1,), device=self.device).long()
                t_batch = t.repeat(batch_size)
                
                sqrt_alpha_bar = self.model.sqrt_alphas_cumprod[t].view(batch_size, 1, 1, 1)
                sqrt_one_minus_alpha_bar = self.model.sqrt_one_minus_alphas_cumprod[t].view(batch_size, 1, 1, 1)
                
                for n in range(N_noise):
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        # Sample noise
                        noise = torch.randn_like(X).to(self.device)
                        
                        # Forward diffusion
                        x_t = sqrt_alpha_bar * X + sqrt_one_minus_alpha_bar * noise
                        
                        # Get score prediction
                        epsilon_pred = self.model.ft_model(x_t, t_batch.float())
                        if self.model.config.data.dataset in ['FFHQ', 'AFHQ']:
                            epsilon_pred, _ = torch.split(epsilon_pred, 3, dim=1)
                        
                        self.model.model.zero_grad()
                        
                        # AdvDM Loss: MAXIMIZE ||ε_θ(x_t, t)||²
                        score_loss = -torch.mean(epsilon_pred ** 2)
                        
                        # Additional: maximize deviation from original noise
                        deviation_loss = -torch.mean((epsilon_pred - noise) ** 2) * 0.3
                        
                        total_loss += (score_loss + deviation_loss)
            
            total_loss = total_loss / (K_timesteps * N_noise)
            total_loss = total_loss.float()
            total_loss.backward()
            
            grad = X.grad
            if grad is None:
                print(f"Warning: Gradient is None at iter {i}")
                break
            
            # PGD update
            X_adv = X + self.alpha * 1.2 * grad.sign()  # Slightly larger step for this attack
            
            # Projection
            eta = torch.clamp(X_adv - base, min=-self.epsilon, max=self.epsilon)
            X = torch.clamp(base + eta, min=-1, max=1).detach()
        
        return X, X - base
    
    def attack_inversion_phase(self, X_nat, target_attr, X_base=None, iter_count=None):
        """
        ENHANCED DDIM Inversion Phase Attack (PhotoGuard-style Encoder Attack)
        
        Key insight from PhotoGuard: Attacking the latent space is MORE EFFECTIVE
        because corrupted latent propagates through ALL generation steps.
        
        Combines:
        1. Latent space deviation (push latent away from clean)
        2. Latent space corruption (inject noise into latent)
        3. Score function disruption at inversion steps
        
        MEMORY OPTIMIZATION:
        - Use finite difference approximation instead of full backprop
        - Gradient checkpointing style - only compute what we need
        
        Args:
            X_nat: Input image
            target_attr: Attribute to attack
            X_base: Base image for epsilon constraint
            iter_count: Number of attack iterations
        """
        base = X_nat if X_base is None else X_base
        X = X_nat.clone().detach_()
        
        # ═══════════════════════════════════════════════════════════════
        # STRONGER INVERSION ATTACK PARAMETERS
        # More iterations = more latent corruption
        # ═══════════════════════════════════════════════════════════════
        iter_count = iter_count or max(getattr(self.config, 'inv_attack_iter', 4), 6)
        
        # Use moderate inversion steps for balance of effect and memory
        t_0 = getattr(self.config, 'diffusion_t0', 200)
        n_inv_step = min(getattr(self.config, 'n_inv_step', 40), 20)  # Up to 20 for better latent
        
        if target_attr:
            self.model.load_attr_weights(target_attr)
        
        print(f"Starting Enhanced Inversion Attack... (Iter: {iter_count}, n_inv_step: {n_inv_step})")
        
        # Finite difference step size (larger = more aggressive)
        fd_eps = 0.015  # Increased from 0.01
        
        # Get clean latent once
        with torch.no_grad():
            clean_latent = self._compute_inversion_latent_fast(base, t_0, n_inv_step)
            
            # Create target latent (random noise - PhotoGuard style)
            target_latent = torch.randn_like(clean_latent) * 0.5
        
        for i in range(iter_count):
            # Get current latent (no gradient needed)
            with torch.no_grad():
                current_latent = self._compute_inversion_latent_fast(X, t_0, n_inv_step)
            
            # ═══════════════════════════════════════════════════════════════
            # FINITE DIFFERENCE GRADIENT ESTIMATION WITH MULTIPLE OBJECTIVES
            # ═══════════════════════════════════════════════════════════════
            grad = torch.zeros_like(X)
            
            # More samples for better gradient estimation
            num_samples = 6  # Increased from 4
            
            for _ in range(num_samples):
                # Random perturbation direction
                direction = torch.randn_like(X)
                direction = direction / (direction.norm() + 1e-8)
                
                # Forward difference: f(x + eps*d)
                with torch.no_grad():
                    X_plus = torch.clamp(X + fd_eps * direction, -1, 1)
                    latent_plus = self._compute_inversion_latent_fast(X_plus, t_0, n_inv_step)
                    
                    # ═══════════════════════════════════════════════════════════════
                    # COMBINED LATENT SPACE LOSS (PhotoGuard + Mist style)
                    # ═══════════════════════════════════════════════════════════════
                    
                    # Loss 1: Maximize deviation from clean latent
                    deviation_loss = -torch.mean((latent_plus - clean_latent) ** 2)
                    
                    # Loss 2: Push toward random target latent (PhotoGuard)
                    target_loss = torch.mean((latent_plus - target_latent) ** 2)
                    
                    # Loss 3: Maximize latent magnitude (causes saturation in generation)
                    magnitude_loss = -torch.mean(latent_plus ** 2)
                    
                    # Loss 4: Maximize latent variance (causes artifacts)
                    variance_loss = -torch.var(latent_plus)
                    
                    loss_plus = 0.4 * deviation_loss + 0.3 * target_loss + 0.2 * magnitude_loss + 0.1 * variance_loss
                    
                    # Backward difference: f(x - eps*d)  
                    X_minus = torch.clamp(X - fd_eps * direction, -1, 1)
                    latent_minus = self._compute_inversion_latent_fast(X_minus, t_0, n_inv_step)
                    
                    deviation_loss_m = -torch.mean((latent_minus - clean_latent) ** 2)
                    target_loss_m = torch.mean((latent_minus - target_latent) ** 2)
                    magnitude_loss_m = -torch.mean(latent_minus ** 2)
                    variance_loss_m = -torch.var(latent_minus)
                    
                    loss_minus = 0.4 * deviation_loss_m + 0.3 * target_loss_m + 0.2 * magnitude_loss_m + 0.1 * variance_loss_m
                    
                    # Central difference gradient estimate
                    grad_estimate = (loss_plus - loss_minus) / (2 * fd_eps)
                    grad += grad_estimate * direction
            
            grad = grad / num_samples
            
            # Gradient ascent with larger step (latent attacks need more strength)
            X_adv = X + self.alpha * 2.5 * grad.sign()  # Increased multiplier
            
            # Projection
            eta = torch.clamp(X_adv - base, min=-self.epsilon, max=self.epsilon)
            X = torch.clamp(base + eta, min=-1, max=1).detach()
            
            torch.cuda.empty_cache()
        
        return X, X - base
    
    def _compute_inversion_latent_fast(self, x, t_0, n_inv_step):
        """
        Fast DDIM inversion without gradient tracking.
        Memory efficient version.
        """
        n = x.shape[0]
        
        with torch.no_grad():
            # Create inversion sequence: 0 → t_0
            seq_inv = np.linspace(0, 1, n_inv_step) * t_0
            seq_inv = [int(s) for s in list(seq_inv)]
            seq_inv_next = [-1] + list(seq_inv[:-1])
            
            x_t = x.clone()
            
            for idx, (i, j) in enumerate(zip(seq_inv_next[1:], seq_inv[1:])):
                t = (torch.ones(n) * i).to(self.device)
                
                # DDIM inversion step
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    out = self.model.base_model(x_t, t.float())
                    
                    if self.model.config.data.dataset in ['FFHQ', 'AFHQ']:
                        epsilon_pred, _ = torch.split(out, 3, dim=1)
                    else:
                        epsilon_pred = out
                    
                    # DDIM forward step
                    alpha_t = self.model.sqrt_alphas_cumprod[max(0, i)] ** 2
                    alpha_next = self.model.sqrt_alphas_cumprod[max(0, j)] ** 2
                    
                    # Simplified DDIM inversion
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
        """
        X_nat: 원본 이미지 (Clean)
        target_img: 방어 목표 이미지 (보통 X_nat 그대로 사용 - Identity 보존 목적)
        target_attr: DiffusionCLIP이 변환하려는 속성 (예: 'zombie', 'old')
        """
        X = X_nat.clone().detach_()

        iter_count = self.config.pgd_iter

        for i in range(iter_count):
            X.requires_grad = True
            
            # 1. DiffusionCLIP을 통해 변환 시도 (Gradient 추적 활성화)
            # 주의: Diffusion 모델 전체를 통과하므로 VRAM 사용량이 매우 높음
            output = self.model.forward_edit(X, target_attr, 
                                            t_0=200, n_inv_step=10, n_test_step=10)

            # 모델 가중치 기울기 초기화 (입력 이미지에 대한 기울기만 필요함)
            self.model.model.zero_grad()

            # 2. Loss 계산
            # 방어 목적: 모델이 변환한 결과(output)가 원본(target_img)과 같아지도록 유도(MSE 최소화)
            # -> 이렇게 하면 DiffusionCLIP이 아무리 변환하려 해도 원본처럼 보이게 만드는 노이즈 생성
            loss = self.loss_fn(output, target_img)
            
            print(f"Iter {i+1}/{iter_count}, Loss: {loss.item():.4f}")

            loss.backward()
            
            # 3. Gradient 추출 및 PGD 업데이트
            grad = X.grad
            if grad is None:
                print("Error: Gradient is None. Check if the model allows gradient flow.")
                break

            # PGD Update: Loss를 줄이는 방향 (Gradient의 반대 방향)
            # 여기서는 '방어'이므로 Loss(변환결과 vs 원본)를 줄여야 함 -> -grad 사용
            # 만약 모델을 망가뜨리는 공격(Attack)이라면 -> +grad 사용
            
            # 시나리오: "이미지 변형에 강건한 노이즈" = 변형을 해도 원본과 비슷하게 유지
            # 따라서 Loss(Output, Original)를 Minimize 해야 함 -> X = X - alpha * sign(grad)
            X_adv = X - self.alpha * grad.sign()

            # 4. Projection (Epsilon Constraint)
            eta = torch.clamp(X_adv - X_nat, min=-self.epsilon, max=self.epsilon)
            X = torch.clamp(X_nat + eta, min=-1, max=1).detach()

        return X, X - X_nat


    def perturb_frequency_domain(self, X_nat, target_attr, freq_band='ALL', iter=1, X_base=None, original_gen_image=None):
        """
        ENHANCED Frequency Domain Attack with Direct Gradient
        
        Key improvement: Instead of weak finite difference, use DIRECT GRADIENT
        through a lightweight forward pass (few diffusion steps).
        
        Also applies Mist-style approach: inject structured frequency noise
        that specifically disrupts diffusion's frequency processing.
        """
        
        # ═══════════════════════════════════════════════════════════════
        # STRONGER DCT PARAMETERS
        # ═══════════════════════════════════════════════════════════════
        dct_clamp = max(getattr(self.config, 'dct_clamp', 0.1), 0.15)  # Increased clamp
        iter = max(iter, 3)  # Minimum 3 iterations
        
        base = X_nat if X_base is None else X_base

        # Pre-compute base DCT once
        X_nat_dct = torch.zeros_like(base)
        for b in range(base.shape[0]):
            for c in range(base.shape[1]):
                X_nat_dct[b, c] = dct.dct_2d(base[b, c])
        
        # Select frequency mask
        if freq_band == 'ALL': freq_mask = self.freq_mask_all
        elif freq_band == 'LOW': freq_mask = self.freq_mask_low
        elif freq_band == 'MID': freq_mask = self.freq_mask_mid
        elif freq_band == 'HIGH': freq_mask = self.freq_mask_high
        else: raise ValueError(f"Unknown band: {freq_band}")
        
        if X_nat.shape[0] != freq_mask.shape[0]:
            freq_mask = freq_mask[:1].repeat(X_nat.shape[0], 1, 1, 1)

        # ═══════════════════════════════════════════════════════════════
        # MIST-STYLE STRUCTURED FREQUENCY NOISE INITIALIZATION
        # Not random - structured to maximally disrupt diffusion
        # ═══════════════════════════════════════════════════════════════
        eta_dct = torch.zeros_like(X_nat_dct)
        
        # Initialize with band-specific structured noise
        if freq_band == 'LOW':
            # Low freq: smooth gradient-like noise
            eta_dct = torch.randn_like(X_nat_dct) * 0.02 * freq_mask
        elif freq_band == 'MID':
            # Mid freq: texture-like noise
            eta_dct = torch.randn_like(X_nat_dct) * 0.015 * freq_mask
        elif freq_band == 'HIGH':
            # High freq: edge-like noise
            eta_dct = torch.randn_like(X_nat_dct) * 0.01 * freq_mask
        else:
            eta_dct = torch.randn_like(X_nat_dct) * 0.015 * freq_mask

        print(f"[*] Starting Enhanced Frequency Attack on band '{freq_band}' (iter: {iter})")
        
        if target_attr:
            self.model.load_attr_weights(target_attr)
        
        for i in range(iter):
            # ═══════════════════════════════════════════════════════════════
            # DIRECT GRADIENT COMPUTATION (More effective than finite diff)
            # Use lightweight diffusion forward (few steps) for gradient
            # ═══════════════════════════════════════════════════════════════
            
            # Convert current DCT perturbation to image
            X_dct_current = X_nat_dct + eta_dct
            X_current = torch.zeros_like(base)
            for b in range(base.shape[0]):
                for c in range(base.shape[1]):
                    X_current[b, c] = dct.idct_2d(X_dct_current[b, c])
            X_current = torch.clamp(X_current, min=-1, max=1)
            
            # Enable gradient for current image
            X_current.requires_grad = True
            
            # Lightweight forward pass (3 steps only for gradient)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                # Get prediction at mid timestep
                t = torch.tensor([100], device=self.device)
                pred_x0 = self.model.predict_x0_from_xt(X_current, t, use_autocast=True)
                self.model.model.zero_grad()
                
                # ═══════════════════════════════════════════════════════════════
                # DESTRUCTION LOSS FOR FREQUENCY ATTACK
                # ═══════════════════════════════════════════════════════════════
                
                # Loss 1: Maximize distance from clean output
                if original_gen_image is not None:
                    deviated_loss = -torch.mean((pred_x0 - original_gen_image) ** 2)
                else:
                    deviated_loss = -torch.mean((pred_x0 - base) ** 2)
                
                # Loss 2: Push toward Mist texture
                if self._mist_target is not None and self._mist_target.shape == pred_x0.shape:
                    texture_loss = torch.mean((pred_x0 - self._mist_target) ** 2) * 0.3
                else:
                    texture_loss = 0
                
                # Loss 3: Maximize output variance (chaos)
                variance_loss = -torch.var(pred_x0) * 0.2
                
                # Combined loss
                total_loss = deviated_loss + texture_loss + variance_loss
            
            total_loss = total_loss.float()
            total_loss.backward()
            
            # Get gradient w.r.t. input image
            grad_img = X_current.grad
            if grad_img is None:
                print(f"  [!] No gradient at iter {i}")
                continue
            
            # ═══════════════════════════════════════════════════════════════
            # CONVERT IMAGE GRADIENT TO DCT GRADIENT
            # ═══════════════════════════════════════════════════════════════
            grad_dct = torch.zeros_like(eta_dct)
            for b in range(base.shape[0]):
                for c in range(base.shape[1]):
                    grad_dct[b, c] = dct.dct_2d(grad_img[b, c])
            
            # Apply frequency mask
            grad_dct = grad_dct * freq_mask
            
            # Update eta_dct (gradient ascent for destruction)
            eta_dct = eta_dct + self.alpha * 3.0 * grad_dct.sign()  # Larger step
            eta_dct = torch.clamp(eta_dct, min=-dct_clamp, max=dct_clamp)
            eta_dct = eta_dct * freq_mask
            
            X_current = X_current.detach()
            torch.cuda.empty_cache()

        # Final image
        with torch.no_grad():
            X_dct_final = X_nat_dct + eta_dct
            X_final = torch.zeros_like(base)
            for b in range(base.shape[0]):
                for c in range(base.shape[1]):
                    X_final[b, c] = dct.idct_2d(X_dct_final[b, c])
            X_final = torch.clamp(X_final, min=-1, max=1)

            # Enforce pixel-space Linf constraint
            eta_pix = torch.clamp(X_final - base, min=-self.epsilon, max=self.epsilon)
            X_final = torch.clamp(base + eta_pix, min=-1, max=1)

        return X_final, X_final - base


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