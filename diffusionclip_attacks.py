import torch
import torch.nn as nn

import torch_dct as dct

"""
Role: A class that comprehensively implements FGSM, I-FGSM, and PGD attacks.

Initialization:
epsilon: Maximum allowed noise size for the attack (based on L∞ norm).
a: Noise update size at each step.
"""
class AttackFunction(object):
    def __init__(self, config, diffusion_model, device=None, epsilon=0.05, alpha=0.01):
        """
        diffusion_model: DiffusionAttackerWrapper 인스턴스
        """
        self.model = diffusion_model # Generator model passed from solver.py
        self.epsilon = epsilon # Maximum attack size (L∞ constraint)
        self.alpha = alpha # Step size (learning rate)
        self.loss_fn = nn.MSELoss().to(device) # Objective function (MSE between output and target)
        self.device = device # Computation device (CPU/GPU)

        self.config = config # Configuration passed from new_solver.py

        # Select PGD or I-FGSM (whether to use random noise for PGD initialization)
        self.rand = True

        # Frequency mask member variables (initialize and reuse)
        self.freq_mask_all = self.create_frequency_masks([1, 3, 256, 256], "ALL")
        self.freq_mask_low = self.create_frequency_masks([1, 3, 256, 256], "LOW")
        self.freq_mask_mid = self.create_frequency_masks([1, 3, 256, 256], "MID")
        self.freq_mask_high = self.create_frequency_masks([1, 3, 256, 256], "HIGH")

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


    def perturb_frequency_domain(self, X_nat, target_attr, freq_band='ALL', iter=1):
        """
        DiffusionCLIP을 속이기 위한 주파수 노이즈 생성 함수
        
        Args:
            X_nat: 원본 이미지 (Normalized [-1, 1])
            target_attr: DiffusionCLIP이 수행하려는 편집 속성 (예: 'male', 'tanned')
            freq_band: 공격할 주파수 대역
            attack_steps: PGD 반복 횟수 (메모리 문제로 적게 설정 권장)
            
        Returns:
            X_adv: 방어 노이즈가 적용된 이미지
            Perturbation: 적용된 노이즈
        """
        
        # 설정값 로드 (config에 없으면 기본값 사용)
        dct_clamp = getattr(self.config, 'dct_clamp', 0.1) # 노이즈 클램핑 범위
        
        # 1. 원본 이미지의 DCT 계수 계산
        X_nat_dct = torch.zeros_like(X_nat)
        for b in range(X_nat.shape[0]):
            for c in range(X_nat.shape[1]):
                X_nat_dct[b, c] = dct.dct_2d(X_nat[b, c])
        
        # 2. 마스크 선택
        if freq_band == 'ALL': freq_mask = self.freq_mask_all
        elif freq_band == 'LOW': freq_mask = self.freq_mask_low
        elif freq_band == 'MID': freq_mask = self.freq_mask_mid
        elif freq_band == 'HIGH': freq_mask = self.freq_mask_high
        else: raise ValueError(f"Unknown band: {freq_band}")
        
        # 배치 사이즈가 다를 경우 마스크 크기 조정
        if X_nat.shape[0] != freq_mask.shape[0]:
            freq_mask = freq_mask[:1].repeat(X_nat.shape[0], 1, 1, 1)

        # 3. 노이즈 초기화
        eta_dct = torch.zeros_like(X_nat_dct).to(self.device)
        eta_dct.requires_grad = True
        
        optimizer = torch.optim.Adam([eta_dct], lr=self.alpha) # SGD 대신 Adam 사용 가능

        print(f"[*] Starting Protection Optimization for attribute '{target_attr}' on band '{freq_band}'")
        
        for i in range(iter):
            # (1) 현재 노이즈가 섞인 이미지 복원
            # 마스크 적용: 선택된 주파수만 변형
            current_eta = eta_dct * freq_mask
            X_dct = X_nat_dct + current_eta
            
            X_adv_input = torch.zeros_like(X_nat)
            for b in range(X_nat.shape[0]):
                for c in range(X_nat.shape[1]):
                    X_adv_input[b, c] = dct.idct_2d(X_dct[b, c])
            
            # 이미지 범위 클리핑 [-1, 1] (Diffusion 입력 조건)
            X_adv_input = torch.clamp(X_adv_input, min=-1, max=1)

            # (2) DiffusionCLIP 통과 (Gradient 흐름 유지 중요)
            # 공격 시에는 VRAM 절약을 위해 step을 줄이는 것이 일반적 (예: 10 step)
            # forward_edit 내부에서 gradient 체크포인팅이나 no_grad가 없어야 함
            edited_output = self.model.forward_edit(
                X_adv_input, 
                target_attr,
                t_0=200,  # 역변환용 단축 스텝
                n_inv_step=10,  # 공격 생성용 단축 스텝
                n_test_step=10
            )

            # (3) Loss 계산: "편집된 결과"가 "원본"과 같아지도록 강제 (Defense)
            # 즉, Diffusion이 이미지를 바꾸려 해도 안 바뀌게 만듦.
            loss = self.loss_fn(edited_output, X_nat)
            
            # 만약 "이미지를 망가뜨리는 것"이 목표라면:
            # loss = -self.loss_fn(edited_output, X_nat)  # 원본과 다르게 (근데 이건 의도된 편집일 수도 있음)
            # loss = -1 * perceptual_loss(edited_output, intended_target) # 이렇게 구현하기도 함

            optimizer.zero_grad()
            loss.backward()
            
            # (4) PGD Update (DCT 도메인)
            with torch.no_grad():
                grad = eta_dct.grad
                grad = grad * freq_mask # 선택된 밴드만 업데이트
                
                # Sign update (FGSM/PGD 스타일)
                eta_dct = eta_dct - self.alpha * grad.sign() # Minimize Loss (Identity Preservation)
                
                # Projection (Clamp noise)
                eta_dct = torch.clamp(eta_dct, min=-dct_clamp, max=dct_clamp)
                eta_dct.requires_grad = True
                optimizer = torch.optim.Adam([eta_dct], lr=self.alpha) # 옵티마이저 리셋 혹은 수동 업데이트

            if i % 1 == 0:
                print(f"Step [{i}/{iter}] Loss (Identity Preservation): {loss.item():.4f}")

        # Finalize
        with torch.no_grad():
            final_eta = eta_dct * freq_mask
            X_dct_final = X_nat_dct + final_eta
            X_final = torch.zeros_like(X_nat)
            for b in range(X_nat.shape[0]):
                for c in range(X_nat.shape[1]):
                    X_final[b, c] = dct.idct_2d(X_dct_final[b, c])
            X_final = torch.clamp(X_final, min=-1, max=1)

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