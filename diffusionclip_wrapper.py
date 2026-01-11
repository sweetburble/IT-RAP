import argparse
import os
from contextlib import nullcontext

import numpy as np
import torch
import yaml

from DiffusionCLIP.configs.paths_config import MODEL_PATHS
from DiffusionCLIP.models.ddpm.diffusion import DDPM
from DiffusionCLIP.models.improved_ddpm.script_util import i_DDPM
from DiffusionCLIP.utils.diffusion_utils import denoising_step, get_beta_schedule


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
        self.checkpoints_dict = checkpoints_dict
        self.root_path = root_path

        self.current_attr = None

        self.args = argparse.Namespace()
        self.args.sample_type = 'ddim'
        self.args.eta = 0.0
        self.args.model_ratio = 1.0
        self.args.hybrid_noise = 0

        config_path = os.path.join(self.root_path, 'configs', 'celeba.yml')
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at: {config_path}")
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        self.config = dict2namespace(config_dict)
        self.config.device = self.device

        self._init_diffusion_params()

        self.base_model = self._build_model()
        self._load_pretrained_weights(self.base_model)
        self.base_model.eval()
        self.base_model.requires_grad_(False)

        self.ft_model = self._build_model()
        self._load_pretrained_weights(self.ft_model)
        self.ft_model.eval()
        self.ft_model.requires_grad_(False)

        if 'male' in self.checkpoints_dict:
            self.load_attr_weights('male')

        self.model = self.ft_model

        alphas_cumprod = np.cumprod(self.alphas, axis=0)
        self.sqrt_alphas_cumprod = torch.tensor(np.sqrt(alphas_cumprod)).to(self.device).float()
        self.sqrt_one_minus_alphas_cumprod = torch.tensor(np.sqrt(1.0 - alphas_cumprod)).to(self.device).float()

    def _init_diffusion_params(self):
        betas = get_beta_schedule(beta_start=0.0001, beta_end=0.02, num_diffusion_timesteps=1000)
        self.betas = torch.from_numpy(betas).float().to(self.device)
        self.alphas = 1.0 - betas

        alphas_cumprod = np.cumprod(self.alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

        if self.config.model.var_type == 'fixedlarge':
            self.logvar = np.log(np.append(posterior_variance[1], betas[1:]))
        elif self.config.model.var_type == 'fixedsmall':
            self.logvar = np.log(np.maximum(posterior_variance, 1e-20))
        else:
            raise ValueError(f"Unsupported var_type: {self.config.model.var_type}")

        self.learn_sigma = False

    def _build_model(self):
        dataset = self.config.data.dataset
        if dataset in ['CelebA_HQ', 'LSUN']:
            self.learn_sigma = False
            model = DDPM(self.config)
        elif dataset in ['FFHQ', 'AFHQ', 'IMAGENET']:
            self.learn_sigma = True
            model = i_DDPM(dataset)
        else:
            raise ValueError(f"Unsupported dataset in config: {dataset}")
        return model.to(self.device)

    def _load_pretrained_weights(self, model):
        dataset = self.config.data.dataset

        if dataset in ['CelebA_HQ', 'LSUN']:
            ckpt_dir = os.path.join(self.root_path, 'checkpoint')
            if dataset == 'CelebA_HQ':
                local_ckpt = os.path.join(ckpt_dir, 'celeba_hq.ckpt')
                url = 'https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/celeba_hq.ckpt'
            else:
                category = getattr(self.config.data, 'category', None)
                if category == 'bedroom':
                    local_ckpt = os.path.join(ckpt_dir, 'bedroom.ckpt')
                    url = 'https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/bedroom.ckpt'
                elif category == 'church_outdoor':
                    local_ckpt = os.path.join(ckpt_dir, 'church_outdoor.ckpt')
                    url = 'https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/church_outdoor.ckpt'
                else:
                    raise ValueError(f"Unsupported LSUN category: {category}")

            if os.path.exists(local_ckpt):
                state_dict = torch.load(local_ckpt, map_location=self.device)
                source = local_ckpt
            else:
                try:
                    state_dict = torch.hub.load_state_dict_from_url(url, map_location=self.device)
                    source = url
                except Exception as e:
                    raise RuntimeError(
                        "[DiffusionCLIPWrapper] Failed to auto-download base diffusion checkpoint. "
                        f"If offline, download manually and place at: {local_ckpt}. "
                        f"Original error: {type(e).__name__}: {e}"
                    )

        elif dataset in ['FFHQ', 'AFHQ', 'IMAGENET']:
            if dataset not in MODEL_PATHS:
                raise KeyError(
                    f"MODEL_PATHS does not contain key '{dataset}'. "
                    "Check DiffusionCLIP/configs/paths_config.py"
                )
            ckpt_path = MODEL_PATHS[dataset]
            if not os.path.isabs(ckpt_path):
                ckpt_path = os.path.join(self.root_path, ckpt_path)
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"Pretrained checkpoint not found: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=self.device)
            if isinstance(ckpt, dict) and 'state_dict' in ckpt:
                ckpt = ckpt['state_dict']
            state_dict = {(k[7:] if k.startswith('module.') else k): v for k, v in ckpt.items()}
            source = ckpt_path

        else:
            raise ValueError(f"Unsupported dataset: {dataset}")

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(
            f"[DiffusionCLIPWrapper] Loaded base diffusion weights from {source} | "
            f"missing: {len(missing)}, unexpected: {len(unexpected)}"
        )

    def load_attr_weights(self, attr_name):
        if self.current_attr == attr_name:
            return
        if attr_name not in self.checkpoints_dict:
            raise KeyError(f"Unknown attribute '{attr_name}'. Available: {list(self.checkpoints_dict.keys())}")

        ckpt_path = self.checkpoints_dict[attr_name]
        ckpt = torch.load(ckpt_path, map_location=self.device)
        if isinstance(ckpt, dict) and 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        state_dict = {(k[7:] if k.startswith('module.') else k): v for k, v in ckpt.items()}
        self.ft_model.load_state_dict(state_dict, strict=False)
        self.current_attr = attr_name

    def predict_x0_from_xt(self, x_start, t, noise=None, use_autocast=True):
        n = x_start.shape[0]
        if noise is None:
            noise = torch.randn_like(x_start).to(self.device)

        sqrt_alpha_bar_t = self.sqrt_alphas_cumprod[t].view(n, 1, 1, 1)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alphas_cumprod[t].view(n, 1, 1, 1)
        x_t = sqrt_alpha_bar_t * x_start + sqrt_one_minus_alpha_bar_t * noise

        if use_autocast and x_t.is_cuda:
            amp_ctx = torch.autocast(device_type='cuda', dtype=torch.float16)
        else:
            amp_ctx = nullcontext()

        with amp_ctx:
            out = self.ft_model(x_t, t.float())

        if self.config.data.dataset in ['FFHQ', 'AFHQ']:
            epsilon_pred, _ = torch.split(out, 3, dim=1)
        else:
            epsilon_pred = out

        pred_x0 = (x_t - sqrt_one_minus_alpha_bar_t * epsilon_pred) / sqrt_alpha_bar_t
        return torch.clamp(pred_x0, -1, 1)

    def forward_edit(
        self,
        x_tensor,
        attr_name,
        t_0=400,
        n_inv_step=40,
        n_test_step=40,
        require_grad=True,
        use_autocast=True,
    ):
        with torch.no_grad():
            self.load_attr_weights(attr_name)

        x = x_tensor.detach() if not require_grad else x_tensor
        n = x.shape[0]

        grad_ctx = torch.enable_grad() if require_grad else torch.no_grad()
        if use_autocast and torch.cuda.is_available():
            amp_ctx = torch.autocast(device_type='cuda', dtype=torch.float16)
        else:
            amp_ctx = nullcontext()

        with grad_ctx, amp_ctx:
            seq_inv = np.linspace(0, 1, n_inv_step) * t_0
            seq_inv = [int(s) for s in list(seq_inv)]
            seq_inv_next = [-1] + list(seq_inv[:-1])

            for i, j in zip(seq_inv_next[1:], seq_inv[1:]):
                t = (torch.ones(n) * i).to(self.device)
                t_prev = (torch.ones(n) * j).to(self.device)
                x = denoising_step(
                    x,
                    t=t,
                    t_next=t_prev,
                    models=[self.base_model],
                    logvars=self.logvar,
                    b=self.betas,
                    sampling_type='ddim',
                    eta=0,
                    learn_sigma=self.learn_sigma,
                    ratio=0,
                )

            x_lat = x

            seq_test = np.linspace(0, 1, n_test_step) * t_0
            seq_test = [int(s) for s in list(seq_test)]
            seq_test_next = [-1] + list(seq_test[:-1])

            model_ratio = float(self.args.model_ratio)
            for i, j in zip(reversed(seq_test), reversed(seq_test_next)):
                t = (torch.ones(n) * i).to(self.device)
                t_next = (torch.ones(n) * j).to(self.device)
                x_lat = denoising_step(
                    x_lat,
                    t=t,
                    t_next=t_next,
                    models=[self.base_model, self.ft_model],
                    logvars=self.logvar,
                    b=self.betas,
                    sampling_type=self.args.sample_type,
                    eta=self.args.eta,
                    learn_sigma=self.learn_sigma,
                    ratio=model_ratio,
                )

        return x_lat
