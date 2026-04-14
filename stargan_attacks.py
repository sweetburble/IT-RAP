import torch
import torch.nn as nn
import torch_dct as dct

class AttackFunction(object):
    def __init__(self, config, model, device=None, epsilon=0.05, a=0.01, use_extended_bands=False):
        self.model = model
        self.epsilon = epsilon
        self.a = a
        self.loss_fn = nn.MSELoss().to(device)
        self.device = device

        self.config = config
        self.rand = True
        self.use_extended_bands = use_extended_bands

        self.freq_mask_all = self.create_frequency_masks([1, 3, 256, 256], "ALL")
        
        # 확장 모드 ON/OFF에 따라 생성할 마스크를 다르게 설정
        if self.use_extended_bands:
            self.freq_mask_lowest = self.create_frequency_masks([1, 3, 256, 256], "LOWEST")
            self.freq_mask_low = self.create_frequency_masks([1, 3, 256, 256], "LOW")
            self.freq_mask_mid = self.create_frequency_masks([1, 3, 256, 256], "MID")
            self.freq_mask_high = self.create_frequency_masks([1, 3, 256, 256], "HIGH")
            self.freq_mask_highest = self.create_frequency_masks([1, 3, 256, 256], "HIGHEST")
        else:
            self.freq_mask_low = self.create_frequency_masks([1, 3, 256, 256], "LOW")
            self.freq_mask_mid = self.create_frequency_masks([1, 3, 256, 256], "MID")
            self.freq_mask_high = self.create_frequency_masks([1, 3, 256, 256], "HIGH")

    def PGD(self, X_nat, y, c_trg, current_noise_level=None):
        X = X_nat.clone().detach_()
        iter = self.config.pgd_iter

        # 에이전트가 선택한 noise_level이 있다면 기존 epsilon 대신 사용
        eps = current_noise_level if current_noise_level is not None else self.epsilon
        # epsilon 비율에 맞춰 step size(a)도 유동적으로 조절
        alpha = (eps / self.epsilon) * self.a if (current_noise_level is not None and self.epsilon != 0) else self.a

        for i in range(iter):
            X.requires_grad = True
            output, _ = self.model(X, c_trg)
            self.model.zero_grad()

            loss = self.loss_fn(output, y)
            loss.backward()
            grad = X.grad

            X_adv = X + alpha * grad.sign()
            eta = torch.clamp(X_adv - X_nat, min=-eps, max=eps)
            X = torch.clamp(X_nat + eta, min=-1, max=1).detach_()

        self.model.zero_grad()
        return X, X - X_nat


    def perturb_frequency_domain(self, X_nat, y, c_trg, freq_band='ALL', current_noise_level=None):
        iter = self.config.dct_iter
        
        # 에이전트가 선택한 noise_level 적용
        dct_clamp = current_noise_level if current_noise_level is not None else self.config.dct_clamp
        dct_coef = (dct_clamp / self.config.dct_clamp) * self.config.dct_coefficent if (current_noise_level is not None and self.config.dct_clamp != 0) else self.config.dct_coefficent

        X_nat_dct = torch.zeros_like(X_nat)
        for b in range(X_nat.shape[0]):
            for c in range(X_nat.shape[1]):
                X_nat_dct[b, c] = dct.dct_2d(X_nat[b, c])

        # 선택된 주파수 밴드에 맞는 마스크 적용
        if freq_band == 'ALL': freq_mask = self.freq_mask_all
        elif freq_band == 'LOWEST': freq_mask = self.freq_mask_lowest
        elif freq_band == 'LOW': freq_mask = self.freq_mask_low
        elif freq_band == 'MID': freq_mask = self.freq_mask_mid
        elif freq_band == 'HIGH': freq_mask = self.freq_mask_high
        elif freq_band == 'HIGHEST': freq_mask = self.freq_mask_highest
        else: raise ValueError(f"Unsupported frequency band: {freq_band}")

        eta_dct = torch.zeros_like(X_nat_dct)
        eta_dct = eta_dct * freq_mask

        for i in range(iter):
            eta_dct.requires_grad = True
            X_dct = X_nat_dct + eta_dct

            X = torch.zeros_like(X_nat)
            for b in range(X_nat.shape[0]):
                for c in range(X_nat.shape[1]):
                    X[b, c] = dct.idct_2d(X_dct[b, c])

            output, _ = self.model(X, c_trg)
            self.model.zero_grad()
            loss = self.loss_fn(output, y)
            loss.backward()

            grad_dct = eta_dct.grad
            grad_dct = grad_dct * freq_mask

            eta_dct_adv = eta_dct.detach() + dct_coef * grad_dct.sign()
            eta_dct = torch.clamp(eta_dct_adv, min=-dct_clamp, max=dct_clamp).detach()
            eta_dct = eta_dct * freq_mask

        X_dct_final = X_nat_dct + eta_dct
        X_adv = torch.zeros_like(X_nat)
        for b in range(X_nat.shape[0]):
            for c in range(X_nat.shape[1]):
                X_adv[b, c] = dct.idct_2d(X_dct_final[b, c])

        X_adv = torch.clamp(X_adv, min=-1, max=1)
        return X_adv, X_adv - X_nat

    def create_frequency_masks(self, shape, freq_band='ALL'):
        B, C, H, W = shape

        # 'ALL' 대역일 경우 즉시 반환
        if freq_band == 'ALL':
            return torch.ones(shape, device=self.device)

        # 1. 2D 좌표 생성 (단 1번만 계산)
        # torch.meshgrid를 사용하면 코드가 훨씬 깔끔해지고 연산이 빠르다.
        i_coords = torch.arange(H, dtype=torch.float32, device=self.device)
        j_coords = torch.arange(W, dtype=torch.float32, device=self.device)
        
        # 'ij' 인덱싱을 사용하여 HxW 형태의 그리드 생성
        grid_i, grid_j = torch.meshgrid(i_coords, j_coords, indexing='ij')

        # 2. 주파수 맵 계산 (단 1번만 계산)
        frequency_map = torch.sqrt(grid_i**2 + grid_j**2)
        
        # max_freq 계산 시 텐서로 변환하여 GPU에서 연산되도록 유지
        max_freq = torch.sqrt(torch.tensor((H-1)**2 + (W-1)**2, dtype=torch.float32, device=self.device))
        frequency_map = frequency_map / max_freq

        # 3. 조건에 맞는 2D 마스크 생성 (H, W 크기의 Boolean 텐서)
        if self.use_extended_bands:
            if freq_band == 'LOWEST': mask_2d = (frequency_map <= 0.2)
            elif freq_band == 'LOW': mask_2d = ((frequency_map > 0.2) & (frequency_map <= 0.4))
            elif freq_band == 'MID': mask_2d = ((frequency_map > 0.4) & (frequency_map <= 0.6))
            elif freq_band == 'HIGH': mask_2d = ((frequency_map > 0.6) & (frequency_map <= 0.8))
            elif freq_band == 'HIGHEST': mask_2d = (frequency_map > 0.8)
            else: raise ValueError(f"Unsupported frequency band: {freq_band}")
        else:
            if freq_band == 'LOW': mask_2d = (frequency_map <= 1/3)
            elif freq_band == 'MID': mask_2d = ((frequency_map > 1/3) & (frequency_map <= 2/3))
            elif freq_band == 'HIGH': mask_2d = (frequency_map > 2/3)
            else: raise ValueError(f"Unsupported frequency band: {freq_band}")

        # 4. (H, W) -> (1, 1, H, W) -> (B, C, H, W) 차원으로 확장
        # float()로 변환한 뒤, unsqueeze로 차원을 늘리고 expand로 배치와 채널에 맞춤
        masks = mask_2d.float().unsqueeze(0).unsqueeze(0).expand(B, C, H, W)

        return masks