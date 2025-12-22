import torch
import numpy as np
import argparse
import yaml
import os

from DiffusionCLIP.models.ddpm.diffusion import DDPM
from DiffusionCLIP.models.improved_ddpm.script_util import i_DDPM
from DiffusionCLIP.utils.diffusion_utils import get_beta_schedule, denoising_step
from DiffusionCLIP.configs.paths_config import MODEL_PATHS
from DiffusionCLIP.main import parse_args_and_config

# YAML Config를 Namespace 객체로 변환하는 헬퍼 함수
def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

class DiffusionCLIPWrapper:
    def __init__(self, device, checkpoints_dict, root_path='./DiffusionCLIP'):
        self.device = device
        self.checkpoints_dict = checkpoints_dict # {'male': path, 'neanderthal': path, ...}
        self.root_path = root_path
        
        # 1. ARGS 수동 생성 (parse_args_and_config 대체)
        self.args = argparse.Namespace()
        
        # Diffusion 및 Sampling 기본 설정
        self.args.sample_type = 'ddim'   # 샘플링 방식
        self.args.eta = 0.0              # Deterministic
        self.args.model_ratio = 1.0      # Finetuned 모델 사용 비율
        self.args.hybrid_noise = 0       # Hybrid 모드 끔
        self.args.edit_attr = None
        
        # 2. CONFIG 로드 (YAML 파일 읽기)
        # 보통 얼굴 데이터셋은 celeba.yml 설정을 따릅니다.
        config_path = os.path.join(self.root_path, 'configs', 'celeba.yml')
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at: {config_path}")

        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
            
        self.config = dict2namespace(config_dict)
        self.config.device = self.device # device 설정 주입

        # 3. Diffusion 파라미터 및 모델 초기화
        self._init_diffusion_params()
        self.model = self._load_base_model()
        self.current_attr = None
        self.learn_sigma = False # CelebA 모델은 보통 False

        # Diff-PGD 계산을 위한 계수 미리 계산
        # x_0 예측 식: x_0 = (x_t - sqrt(1-alpha_bar) * epsilon) / sqrt(alpha_bar)
        alphas_cumprod = np.cumprod(self.alphas, axis=0)
        self.sqrt_alphas_cumprod = torch.tensor(np.sqrt(alphas_cumprod)).to(self.device).float()
        self.sqrt_one_minus_alphas_cumprod = torch.tensor(np.sqrt(1.0 - alphas_cumprod)).to(self.device).float()

    def _init_diffusion_params(self):
        betas = get_beta_schedule(
            beta_start=0.0001,
            beta_end=0.02,
            num_diffusion_timesteps=1000,
        )
        self.betas = torch.from_numpy(betas).float().to(self.device)
        self.alphas = 1.0 - betas

        alphas_cumprod = np.cumprod(self.alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        posterior_variance = betas * \
                             (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        
        # var_type에 따른 logvar 계산
        if self.config.model.var_type == "fixedlarge":
            self.logvar = np.log(np.append(posterior_variance[1], betas[1:]))

        elif self.config.model.var_type == 'fixedsmall':
            self.logvar = np.log(np.maximum(posterior_variance, 1e-20))

    def _load_base_model(self):
        # 데이터셋에 따른 모델 아키텍처 로드
        if self.config.data.dataset in ["CelebA_HQ", "LSUN"]:
            model = DDPM(self.config)
            learn_sigma = False
        elif self.config.data.dataset in ["FFHQ", "AFHQ"]:
            model = i_DDPM(self.config.data.dataset)
            learn_sigma = True
        else:
            raise ValueError("Unsupported dataset")
            
        model.to(self.device)
        model.eval()
        self.learn_sigma = learn_sigma
        return model

    def load_attr_weights(self, attr_name):
        """
        필요할 때만 해당 속성의 가중치를 로드 (VRAM 절약)
        """
        if self.current_attr == attr_name:
            return

        print(f"[DiffusionWrapper] Loading weights for attribute: {attr_name}")
        path = self.checkpoints_dict[attr_name]
        ckpt = torch.load(path, map_location=self.device)
        
        # DataParallel 등으로 저장된 경우 키 처리
        if 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']

        
        new_ckpt = {}
        for k, v in ckpt.items():
            if k.startswith('module.'):
                new_ckpt[k[7:]] = v
            else:
                new_ckpt[k] = v
        
        self.model.load_state_dict(new_ckpt, strict=False)
        self.current_attr = attr_name
    
    # Diff-PGD 공격을 위한 핵심 함수
    def predict_x0_from_xt(self, x_start, t, noise=None):
        """
        Diff-PGD를 위해 특정 타임스텝 t에서의 Denoising 결과를 예측합니다.
        
        Args:
            x_start: 현재 최적화 중인 Adversarial Image (Gradient 추적 대상)
            t: 공격할 타임스텝 (batch size 크기의 텐서)
            noise: x_start에 더할 노이즈 (없으면 랜덤 생성)
        
        Returns:
            predicted_x0: 모델이 예측한 깨끗한 이미지
        """
        n = x_start.shape[0]
        if noise is None:
            noise = torch.randn_like(x_start).to(self.device)

        # 1. q_sample: x_start(Clean) -> x_t (Noisy)
        # 공식: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
        sqrt_alpha_bar_t = self.sqrt_alphas_cumprod[t].view(n, 1, 1, 1)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alphas_cumprod[t].view(n, 1, 1, 1)
        
        x_t = sqrt_alpha_bar_t * x_start + sqrt_one_minus_alpha_bar_t * noise

        # 2. 모델 예측: epsilon_theta(x_t, t)
        # 모델은 노이즈가 섞인 이미지에서 '노이즈'를 예측합니다.
        output = self.model(x_t, t.float())
        
        # 모델 타입에 따라 출력 처리 (DDPM vs i_DDPM)
        if self.config.data.dataset in ["FFHQ", "AFHQ"]: # i_DDPM
            # i_DDPM은 channel을 split해서 사용 (mean, var)
            epsilon_pred, _ = torch.split(output, 3, dim=1)
        else: # DDPM
            epsilon_pred = output

        # 3. x_0 예측 (Reconstruction)
        # 공식의 역연산: predicted_x0 = (x_t - sqrt(1-alpha_bar) * predicted_epsilon) / sqrt(alpha_bar)
        # 이 predicted_x0가 Target Image와 같아지도록 Loss를 걸면, 
        # Diffusion 모델은 어떤 노이즈가 들어와도 원본을 복구하려고 하게 됨 (편집 방어).
        
        pred_x0 = (x_t - sqrt_one_minus_alpha_bar_t * epsilon_pred) / sqrt_alpha_bar_t
        
        # Pixel 값 Clipping [-1, 1]
        pred_x0 = torch.clamp(pred_x0, -1, 1)
        
        return pred_x0

    def forward_edit(self, x_tensor, attr_name, t_0=400, n_inv_step=40, n_test_step=40):
        """
        x_tensor: (1, 3, H, W) normalized [-1, 1]
        returns: (1, 3, H, W) normalized [-1, 1]
        """
        # 1. 속성 모델 가중치 로드 (가중치는 고정이므로 no_grad)
        with torch.no_grad():
            self.load_attr_weights(attr_name)
        
        # x_tensor는 requires_grad=True 상태로 들어와야 함
        # 이미지 전처리 확인: [-1, 1] 범위여야 함
        x = x_tensor
        n = x.shape[0]

        # -----------------------------------------------------------
        # Inversion (DDIM) -> Latent
        # -----------------------------------------------------------
        seq_inv = np.linspace(0, 1, n_inv_step) * t_0
        seq_inv = [int(s) for s in list(seq_inv)]
        seq_inv_next = [-1] + list(seq_inv[:-1])
        
        model_list = [self.model] # 리스트 형태로 전달해야 함

        for it, (i, j) in enumerate(zip((seq_inv_next[1:]), (seq_inv[1:]))):
            t = (torch.ones(n) * i).to(self.device)
            t_prev = (torch.ones(n) * j).to(self.device)
            
            x = denoising_step(x, t=t, t_next=t_prev, models=model_list,
                                logvars=self.logvar, sampling_type='ddim',
                                b=self.betas, eta=0, learn_sigma=self.learn_sigma, ratio=0)
            
        x_lat = x # Gradient 유지

        # -----------------------------------------------------------
        # Generation (Denoising) with Finetuned Model
        # -----------------------------------------------------------
        seq_test = np.linspace(0, 1, n_test_step) * t_0
        seq_test = [int(s) for s in list(seq_test)]
        seq_test_next = [-1] + list(seq_test[:-1])
        
        x_gen = x_lat
        
        for i, j in zip(reversed(seq_test), reversed(seq_test_next)):
            t = (torch.ones(n) * i).to(self.device)
            t_next = (torch.ones(n) * j).to(self.device)
            
            x_gen = denoising_step(x_gen, t=t, t_next=t_next, models=model_list,
                                    logvars=self.logvar, sampling_type=self.args.sample_type,
                                    b=self.betas, eta=self.args.eta, 
                                    learn_sigma=self.learn_sigma, ratio=0) # ratio=0 -> Finetuned 모델 100% 사용

        return x_gen