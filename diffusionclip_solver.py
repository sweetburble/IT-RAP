from diffusionclip_wrapper import DiffusionCLIPWrapper
from DiffusionCLIP.configs.paths_config import *
from torchvision.utils import save_image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd
import numpy as np
import os
import random
import math
import gc
import time
from torchvision import transforms

from torchvision.models import vgg19, resnet50, VGG19_Weights, ResNet50_Weights

import diffusionclip_attacks
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from collections import namedtuple, deque
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from optuna_util import analyze_perturbation, print_debug, print_final_metrics, print_comprehensive_metrics, visualize_actions, calculate_and_save_metrics, plot_reward_trend, save_reward_moving_average_txt
from img_trans_methods import compress_jpeg, denoise_opencv, denoise_scikit, random_resize_padding, random_image_transforms, apply_random_transform

from segment_tree import MinSegmentTree, SumSegmentTree

from meso_net import Meso4, MesoInception4, convert_tf_weights_to_pytorch







np.random.seed(0)

"""Noisy Layer"""
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.Tensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer('bias_epsilon', torch.Tensor(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def reset_noise(self):
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x):

        we = self.weight_epsilon.clone()
        be = self.bias_epsilon.clone()
        return F.linear(x, self.weight_mu + self.weight_sigma * we, self.bias_mu + self.bias_sigma * be)

    @staticmethod
    def scale_noise(size):
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())


"""Rainbow DQN Network (DuelingNet + NoisyNet + Categorical DQN)"""
class RainbowDQNNet(nn.Module):
    def __init__(self, input_dim, action_dim, atom_size, support):
        super(RainbowDQNNet, self).__init__()
        self.atom_size = atom_size
        self.support = support

        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
        )

        self.value_hidden = NoisyLinear(128, 128)
        self.value_layer = NoisyLinear(128, atom_size)

        self.advantage_hidden = NoisyLinear(128, 128)
        self.advantage_layer = NoisyLinear(128, action_dim * atom_size)

        self.action_dim = action_dim

    def forward(self, state):
        dist = self.get_distribution(state)
        q_values = torch.sum(dist * self.support, dim=2)
        return q_values

    """Return distribution"""
    def get_distribution(self, state):
        feature = self.feature_layer(state)

        value_hid = F.relu(self.value_hidden(feature))
        value = self.value_layer(value_hid).view(-1, 1, self.atom_size)

        advantage_hid = F.relu(self.advantage_hidden(feature))
        advantage = self.advantage_layer(advantage_hid).view(-1, self.action_dim, self.atom_size)

        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)

        dist = F.softmax(q_atoms, dim=-1)
        dist = dist.clamp(min=1e-3, max=1.0)

        return dist

    """Reset Noisy Layer"""
    def reset_noise(self):
        self.value_hidden.reset_noise()
        self.value_layer.reset_noise()
        self.advantage_hidden.reset_noise()
        self.advantage_layer.reset_noise()


"""Prioritized Replay Buffer (PER) with N-step Learning: Store and sample N-step transitions"""
class PrioritizedReplayBuffer(object):
    def __init__(self, capacity, alpha, beta_start, beta_frames, n_step, gamma):
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.beta = beta_start
        self.n_step = n_step
        self.gamma = gamma

        self.memory = deque(maxlen=capacity)
        self.priorities_sum_tree = SumSegmentTree(capacity)
        self.priorities_min_tree = MinSegmentTree(capacity)
        self.max_priority = 1.0
        self.n_step_buffer = deque(maxlen=n_step)
        self.Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

        assert alpha >= 0
        self._storage = []
        self._maxsize = capacity
        self._next_idx = 0

    def beta_by_frame(self, frame_idx):
        beta_initial = self.beta_start
        beta_annealing_frames = self.beta_frames
        beta_final = 1.0
        return min(beta_final, beta_initial + frame_idx * (beta_final - beta_initial) / beta_annealing_frames)

    """Create and store N-step transition"""
    def push(self, state, action, reward, next_state, done):
        transition = (state, action, reward, next_state, done)
        self.n_step_buffer.append(transition)

        if len(self.n_step_buffer) == self.n_step:
            state, action = self.n_step_buffer[0][:2]
            n_step_reward = 0
            next_s = next_state
            for i in range(self.n_step):
                r, s, d = self.n_step_buffer[i][2:]
                n_step_reward += r * (self.gamma ** i)
                next_s, done = (s, d) if d else (next_s, done)

            max_priority = self.max_priority
            n_step_transition = (state, action, n_step_reward, next_s, torch.tensor([float(done)]))
            self.memory.append(n_step_transition)

            self.priorities_sum_tree[self._next_idx] = max_priority ** self.alpha
            self.priorities_min_tree[self._next_idx] = max_priority ** self.alpha
            self._next_idx = (self._next_idx + 1) % self._maxsize
            self.n_step_buffer.popleft()


    def _get_priority(self, index):
        return self.priorities_sum_tree[index]

    def _set_priority(self, index, priority):
        priority_alpha = priority ** self.alpha
        self.priorities_sum_tree[index] = priority_alpha
        self.priorities_min_tree[index] = priority_alpha
        self.max_priority = max(self.max_priority, priority)

    """frame_idx for beta annealing"""
    def sample(self, batch_size, frame_idx=None):
        if frame_idx is not None:
            self.beta = self.beta_by_frame(frame_idx)

        indices = self._sample_proportional(batch_size)
        transitions = [self.memory[idx] for idx in indices]

        weights = [self._calculate_weight(idx, frame_idx) for idx in indices]

        batch = self.Transition(*zip(*transitions))
        weights_tensor = torch.FloatTensor(weights).unsqueeze(1).to(device)
        indices_tensor = torch.LongTensor(indices).to(device)


        return batch, weights_tensor, indices_tensor

    """Proportional sampling"""
    def _sample_proportional(self, batch_size):
        indices = []
        p_total = self.priorities_sum_tree.sum(0, len(self) - 1)
        segment = p_total / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.priorities_sum_tree.retrieve(upperbound)
            indices.append(idx)
        return indices

    """Calculate Importance Sampling weight"""
    def _calculate_weight(self, idx, frame_idx):
        prob = self.priorities_sum_tree[idx] / self.priorities_sum_tree.sum()
        N = len(self)
        weight = (prob * N) ** -self.beta
        return weight / ((self.priorities_min_tree.min() / self.priorities_sum_tree.sum() * N) ** -self.beta)


    def update_priorities(self, indices, priorities):
        assert len(indices) == len(priorities)
        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < len(self)
            self._set_priority(idx, priority)


    def __len__(self):
        return len(self.memory)


"""SolverRainbow for training and testing StarGAN and Rainbow DQN Attack."""
class SolverRainbow(object):
    def __init__(self, dataset_loader, config, run = None):
        self.config = config


        self.run = run


        self.dataset_loader = dataset_loader

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


        self.c_dim = config.c_dim
        self.c2_dim = config.c2_dim
        self.image_size = config.image_size
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.g_repeat_num = config.g_repeat_num
        self.d_repeat_num = config.d_repeat_num
        self.lambda_cls = config.lambda_cls
        self.lambda_rec = config.lambda_rec
        self.lambda_gp = config.lambda_gp


        self.dataset = config.dataset
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.selected_attrs = config.selected_attrs
        self.training_image_num = config.training_image_num
        self.inference_image_num = config.inference_image_num
        self.reward_weight = config.reward_weight


        self.test_iters = config.test_iters


        self.model_save_dir = config.model_save_dir
        self.result_dir = config.result_dir


        self.batch_size = config.batch_size
        self.agent_lr = config.agent_lr
        self.gamma = config.gamma
        self.epsilon_start = config.epsilon_start
        self.epsilon_end = config.epsilon_end
        self.epsilon_decay = config.epsilon_decay
        self.target_update_interval = config.target_update_interval
        self.memory_capacity = config.memory_capacity
        self.max_steps_per_episode = config.max_steps_per_episode
        self.action_dim = config.action_dim
        self.noise_level = config.noise_level
        self.feature_extractor_name = config.feature_extractor_name


        self.alpha = config.alpha
        self.beta_start = config.beta_start
        self.beta_frames = config.beta_frames
        self.prior_eps = config.prior_eps


        self.v_min = config.v_min
        self.v_max = config.v_max
        self.atom_size = config.atom_size
        self.support = torch.linspace(self.v_min, self.v_max, self.atom_size).to(self.device)


        self.n_step = config.n_step


        self.lpips_loss = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(self.device)


        self.t_0 = TEST_CONFIG['t_0']
        self.n_inv_step = TEST_CONFIG['n_inv_step']
        self.n_test_step = TEST_CONFIG['n_test_step']

        self.build_model()
        self.build_rlab_agent()





































































































































































































































































    def load_rainbow_dqn_checkpoint(self, checkpoint_path):
        abs_path = os.path.abspath(checkpoint_path)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(
                f"Rainbow DQN checkpoint not found: {abs_path}\n"
                f"Tip: check --model_save_dir and that you initialized/saved the checkpoint in the same folder."
            )


        try:
            checkpoint = torch.load(abs_path, map_location=self.device, weights_only=True)
        except TypeError:
            checkpoint = torch.load(abs_path, map_location=self.device)

        ckpt_sd = checkpoint.get('rainbow_dqn_state_dict', checkpoint)
        ckpt_meta = checkpoint.get('meta', {}) if isinstance(checkpoint, dict) else {}
        if ckpt_meta:
            print(f"[INFO] Rainbow DQN checkpoint meta: {ckpt_meta}")


        try:
            self.rl_agent.dqn.load_state_dict(ckpt_sd)
            opt_sd = checkpoint.get('optimizer_state_dict') if isinstance(checkpoint, dict) else None
            if opt_sd is not None:
                self.rl_agent.optimizer.load_state_dict(opt_sd)
            print(f"[INFO] Rainbow DQN model loaded successfully: {abs_path}")
            return
        except RuntimeError as e:

            model_sd = self.rl_agent.dqn.state_dict()
            filtered_sd = {
                k: v for k, v in ckpt_sd.items()
                if (k in model_sd and hasattr(v, 'shape') and model_sd[k].shape == v.shape)
            }

            if len(filtered_sd) == 0:
                raise

            self.rl_agent.dqn.load_state_dict(filtered_sd, strict=False)
            print(
                "[WARN] Checkpoint did not match current network exactly; loaded only compatible weights.\n"
                "       This usually happens when action_dim changed (e.g., 5->6), so the advantage head is re-initialized.\n"
                f"       checkpoint: {abs_path}\n"
                f"       loaded_keys: {len(filtered_sd)}/{len(model_sd)}\n"
                "       If you want full compatibility, re-initialize and retrain the RL agent with the same action_dim."
            )

            print("[WARN] Optimizer state not loaded due to partial weight load.")
            return




    def build_model(self):

        self.target_attributes = {
            "male": "./DiffusionCLIP/checkpoint/human_male_t401.pth",
        }

        self.diffusionclip_wrapper = DiffusionCLIPWrapper(
            device=self.device,
            checkpoints_dict=self.target_attributes,
            root_path='./DiffusionCLIP'
        )


    """Build RLAB agent components (Rainbow DQN, Prioritized Replay Buffer, Feature Extractor)."""
    def build_rlab_agent(self):

        if self.feature_extractor_name == "vgg19":
            self.feature_extractor = vgg19(weights=VGG19_Weights.DEFAULT).features.to(device).eval()
        elif self.feature_extractor_name == "resnet50":
            extractor = resnet50(weights=ResNet50_Weights.DEFAULT)
            self.feature_extractor = nn.Sequential(*list(extractor.children())[:-2]).to(device).eval()
        elif self.feature_extractor_name == "mesonet":
            meso4_model = Meso4()
            convert_tf_weights_to_pytorch("./weights/Meso4_DF.h5", meso4_model, 'meso4')
            self.meso4_extractor = meso4_model.to(device).eval()

            meso_inception4_model = MesoInception4()
            convert_tf_weights_to_pytorch("./weights/MesoInception_DF.h5", meso_inception4_model, 'mesoinception4')
            self.meso4_inception_extractor = meso_inception4_model.to(device).eval()
        elif self.feature_extractor_name == "ghostfacenets":


            model = GhostFaceNetsV2(image_size=256, width=1, dropout=0.)
            self.feature_extractor = model.to(device).eval()
        elif self.feature_extractor_name == "edgeface":


            self.feature_extractor = torch.hub.load(
                'otroshi/edgeface',
                'edgeface_xs_gamma_06',
                source='github',
                pretrained=True,
                trust_repo="check"
            ).to(device).eval()
        else:
            raise ValueError("Invalid FEATURE_EXTRACTOR_NAME")


        self.memory = PrioritizedReplayBuffer(capacity=self.memory_capacity, alpha=self.alpha, beta_start=self.beta_start, beta_frames=self.beta_frames, n_step=self.n_step, gamma=self.gamma)

        Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))
        self.Transition = Transition


        if (self.feature_extractor_name == "vgg19"):
            state_dim = 65536
        elif (self.feature_extractor_name == "resnet50"):
            state_dim = 262144
        elif (self.feature_extractor_name == "mesonet"):
            state_dim = 64
        else:

            state_dim = 1024




        num_attrs = max(1, len(getattr(self, 'target_attributes', {})))
        total_steps_est = int(self.max_steps_per_episode) * int(self.training_image_num) * int(num_attrs)
        initial_ratio_steps = max(0, total_steps_est // 2)

        action_dim = self.action_dim
        self.rl_agent = RainbowDQNAgent(state_dim, action_dim, self.agent_lr, self.gamma, self.epsilon_start, self.epsilon_end, self.epsilon_decay, self.target_update_interval,
                                        v_min=self.v_min, v_max=self.v_max, atom_size=self.atom_size,
                                        beta_start=self.beta_start, beta_frames=self.beta_frames, prior_eps=self.prior_eps,
                                        n_step=self.n_step, initial_ratio_steps=initial_ratio_steps)
        self.rl_agent.dqn.to(self.device)
        self.rl_agent.dqn_target.to(self.device)

    def load_model_weights(self, model, path):
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = model.state_dict()


        pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'preprocessing' not in k}

        model_dict.update(pretrained_dict)

        model.load_state_dict(pretrained_dict, strict=False)

    def denorm(self, x):
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def label2onehot(self, labels, dim):
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out

    def create_labels(self, c_org, c_dim=5, dataset=None, selected_attrs=None):

        if dataset in ('CelebA', 'MAADFace'):
            hair_color_indices = []
            for i, attr_name in enumerate(selected_attrs):
                if attr_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
                    hair_color_indices.append(i)

        c_trg_list = []
        for i in range(c_dim):
            if dataset in ('CelebA', 'MAADFace'):
                c_trg = c_org.clone()
                if i in hair_color_indices:
                    c_trg[:, i] = 1
                    for j in hair_color_indices:
                        if j != i:
                            c_trg[:, j] = 0
                else:
                    c_trg[:, i] = (c_trg[:, i] == 0)

            c_trg_list.append(c_trg.to(self.device))
        return c_trg_list



    """
        Reward calculation function
        LPIPS: A value between [0, 1]. The lower the value, the more similar the two images are.
        Generally, an LPIPS value greater than 0.5 is considered to indicate a human-observable difference between the two images.
    """
    def calculate_reward(self, original_gen_image, perturbed_gen_image, x_real, perturbed_image, attr_name):

        transformed_x_real, transformed_perturbed_image = apply_random_transform(x_real, perturbed_image)
        transformed_original = self.diffusionclip_wrapper.forward_edit(transformed_x_real.detach(), attr_name, t_0=self.t_0, n_inv_step=self.n_inv_step, n_test_step=self.n_test_step, require_grad=False,)
        transformed_perturbed = self.diffusionclip_wrapper.forward_edit(transformed_perturbed_image.detach(), attr_name, t_0=self.t_0, n_inv_step=self.n_inv_step, n_test_step=self.n_test_step, require_grad=False,)

        defense_l1_loss = F.l1_loss(transformed_original, transformed_perturbed)
        defense_l2_loss = F.mse_loss(transformed_original, transformed_perturbed)

        transformed_original = transformed_original.clamp(-1.0, 1.0)
        transformed_perturbed = transformed_perturbed.clamp(-1.0, 1.0)

        defense_lpips = self.lpips_loss(transformed_original, transformed_perturbed).mean()


        reward_defense = ((defense_l1_loss / 10) + (defense_l2_loss / 5) + defense_lpips) * 5



        x_real_np = x_real.squeeze().cpu().numpy()
        perturbed_image_np = perturbed_image.squeeze().cpu().numpy()
        if np.array_equal(x_real_np, perturbed_image_np):
            invisibility_psnr = 100.0
        else:
            invisibility_psnr = psnr(x_real.squeeze().cpu().numpy(), perturbed_image.squeeze().cpu().numpy(), data_range=2.0)
        invisibility_ssim = ssim(x_real.squeeze().cpu().numpy(), perturbed_image.squeeze().cpu().numpy(), data_range=2.0, win_size=3, channel_axis=0, multichannel=True)
        invisibility_lpips = self.lpips_loss(perturbed_image, x_real).mean()


        reward_invisibility = (0.01 * invisibility_psnr) + invisibility_ssim + (1 - invisibility_lpips)


        w_defense = self.reward_weight
        w_invisibility = 1.0 - w_defense
        total_reward = w_defense * reward_defense + w_invisibility * reward_invisibility
        print(f"[Attack Metrics] L2_err={defense_l2_loss:.4f}, LPIPS={invisibility_lpips:.4f}, SSIM={invisibility_ssim:.4f}, L1_err={defense_l1_loss:.4f}, Reward={total_reward:.4f}")
        return total_reward, defense_l1_loss, defense_l2_loss, defense_lpips, invisibility_ssim, invisibility_psnr, invisibility_lpips


    """
        Extracts features from images using VGG-19/ResNet50/MesoNet and returns a state vector.

        Args:
            x_real: Original image, shape=[1, 3, 256, 256], range=[-1, 1]
            perturbed_image: Image with added noise, shape=[1, 3, 256, 256], range=[-1, 1]
            original_gen_image: Image generated from the original image, shape=[1, 3, 256, 256], range=[-1, 1]
            perturbed_gen_image: Transformed generated image, shape=[1, 3, 256, 256], range=[-1, 1]

        Returns:
            A state vector combining the features of the four images.

        1. Using VGG-19 model
        Output dimension: 256x256 input image -> [1, 512, 8, 8] feature map -> [1, 32768] flattened
        Final output when combining 2 images: [1, 65536]

        2. Using ResNet50 model
        Output dimension: 256x256 input image -> [1, 2048, 8, 8] feature map -> [1, 131072] flattened
        Final output when combining 2 images: [1, 262144]

        3. Using MesoNet model
        Final output when combining 2 images: [1, 64]

        4. Using GhostFaceNets / EdgeFace model
        Output dimension: 256x256 input image -> [1, 512] embedding vector
        Final output when combining 2 images: [1, 1024]
    """
    def get_state(self, perturbed_image, perturbed_gen_image):
        if self.feature_extractor_name == "mesonet":

            with torch.no_grad():
                meso4_features_perturbed = self.meso4_extractor.extract_features(perturbed_image)
                meso4_features_perturbed_gen = self.meso4_extractor.extract_features(perturbed_gen_image)

                meso4_inception_features_perturbed = self.meso4_inception_extractor.extract_features(perturbed_image)
                meso4_inception_features_perturbed_gen = self.meso4_inception_extractor.extract_features(perturbed_gen_image)


            combined_features = torch.cat([meso4_features_perturbed, meso4_features_perturbed_gen, meso4_inception_features_perturbed, meso4_inception_features_perturbed_gen], dim=1)

            return combined_features

        if self.feature_extractor_name == "edgeface" or self.feature_extractor_name == "ghostfacenets":


            perturbed_image_norm = perturbed_image
            perturbed_gen_image_norm = perturbed_gen_image
        else:


            perturbed_image_norm = (perturbed_image + 1) / 2
            perturbed_gen_image_norm = (perturbed_gen_image + 1) / 2


            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
            perturbed_image_norm = normalize(perturbed_image_norm)
            perturbed_gen_image_norm = normalize(perturbed_gen_image_norm)


        with torch.no_grad():
            perturbed_features = self.feature_extractor(perturbed_image_norm)
            perturbed_gen_features = self.feature_extractor(perturbed_gen_image_norm)


        perturbed_features = perturbed_features.view(perturbed_features.size(0), -1)
        perturbed_gen_features = perturbed_gen_features.view(perturbed_gen_features.size(0), -1)


        combined_features = torch.cat([perturbed_features, perturbed_gen_features], dim=1)

        return combined_features


    """Performs the RLAB attack (Rainbow DQN Agent)"""
    def train_attack(self):
        torch.cuda.empty_cache()
        torch.autograd.set_detect_anomaly(True)
        gc.collect()

        total_perturbation_map = np.zeros((256, 256))
        total_remain_map = np.zeros((256, 256))


        checkpoint_path = os.path.join(self.model_save_dir, f'final_rainbow_dqn.pth')
        self.load_rainbow_dqn_checkpoint(checkpoint_path)

        data_loader = self.dataset_loader
        print(f"data_loader length: {len(data_loader)}")


        results = {
            "원본(변형없음)": {"l1_error": 0.0, "l2_error": 0.0, "defense_psnr": 0.0,
                        "defense_ssim": 0.0, "defense_lpips": 0.0, "attack_success": 0, "total_remain_map": np.zeros((256, 256))},
            "JPEG압축": {"l1_error": 0.0, "l2_error": 0.0, "defense_psnr": 0.0,
                    "defense_ssim": 0.0, "defense_lpips": 0.0, "attack_success": 0, "total_remain_map": np.zeros((256, 256))},
            "OpenCV디노이즈": {"l1_error": 0.0, "l2_error": 0.0, "defense_psnr": 0.0,
                        "defense_ssim": 0.0, "defense_lpips": 0.0, "attack_success": 0, "total_remain_map": np.zeros((256, 256))},
            "중간값스무딩": {"l1_error": 0.0, "l2_error": 0.0, "defense_psnr": 0.0,
                        "defense_ssim": 0.0, "defense_lpips": 0.0, "attack_success": 0, "total_remain_map": np.zeros((256, 256))},
            "크기조정패딩": {"l1_error": 0.0, "l2_error": 0.0, "defense_psnr": 0.0,
                    "defense_ssim": 0.0, "defense_lpips": 0.0, "attack_success": 0, "total_remain_map": np.zeros((256, 256))},
            "이미지변환": {"l1_error": 0.0, "l2_error": 0.0, "defense_psnr": 0.0,
                    "defense_ssim": 0.0, "defense_lpips": 0.0, "attack_success": 0, "total_remain_map": np.zeros((256, 256))}
        }


        total_invisible_psnr, total_invisible_ssim, total_invisible_lpips = 0.0, 0.0, 0.0
        episode = 0
        frame_idx = 0


        action_history = []
        image_indices = []
        attr_indices = []
        step_indices = []
        reward_per_episode = []


        for test_img_idx, (x_real, c_org, filename) in enumerate(data_loader):
            print('\n'*3)
            print(f"Processing image {test_img_idx+1}: {filename}")


            x_real = x_real.to(self.device)




            attack_func = diffusionclip_attacks.AttackFunction(config=self.config, diffusion_model=self.diffusionclip_wrapper, device=self.device)


            noattack_result_list = [x_real]
            jpeg_result_list = [x_real]
            opencv_result_list = [x_real]
            median_result_list = [x_real]
            padding_result_list = [x_real]
            transforms_result_list = [x_real]



            attr_keys = list(self.target_attributes.keys())

            for idx, attr_name in enumerate(attr_keys):
                print("=" * 100)
                print(f"Target Attribute: {attr_name} ({idx + 1}/{len(attr_keys)})")


                perturbed_image = x_real.clone().detach_() + torch.tensor(np.random.uniform(-self.noise_level, self.noise_level, x_real.shape).astype('float32')).to(self.device)
                perturbed_image = perturbed_image.clamp(-1.0, 1.0)




                with torch.no_grad():

                    original_gen_image = self.diffusionclip_wrapper.forward_edit(
                        x_real, attr_name, t_0=self.t_0, n_inv_step=self.n_inv_step, n_test_step=self.n_test_step
                    )


                    perturbed_gen_image = self.diffusionclip_wrapper.forward_edit(
                        perturbed_image, attr_name, t_0=self.t_0, n_inv_step=self.n_inv_step, n_test_step=self.n_test_step
                    )

                n_step_buffer_test_attack = deque(maxlen=self.n_step)
                total_reward_this_episode = 0

                for step in range(self.max_steps_per_episode):
                    frame_idx += 1


                    state = self.get_state(perturbed_image, perturbed_gen_image)


                    action = self.rl_agent.select_action(state)


                    action_history.append(action.item())
                    image_indices.append(test_img_idx)
                    attr_indices.append(idx)
                    step_indices.append(step)



                    if action == 0:



                        perturbed_image, _ = attack_func.Diff_PGD(
                            perturbed_image,
                            original_gen_image,
                            attr_name,
                            X_base=x_real,
                        )
                        print("Action selected this step: Diff_PGD (anchored to x_real)")
                    elif action in [1, 2, 3]:

                        freq_band = ['LOW', 'MID', 'HIGH'][action - 1]
                        perturbed_image, _ = attack_func.perturb_frequency_domain(
                            perturbed_image,
                            attr_name,
                            freq_band=freq_band,
                            X_base=x_real,
                        )
                    else:
                        raise ValueError("Invalid action index")



                    with torch.no_grad():
                        perturbed_gen_image = self.diffusionclip_wrapper.forward_edit(
                            perturbed_image, attr_name, t_0=self.t_0, n_inv_step=self.n_inv_step, n_test_step=self.n_test_step
                        )



                    reward, defense_l1_loss, defense_l2_loss, defense_lpips, invisibility_ssim, invisibility_psnr, invisibility_lpips = self.calculate_reward(original_gen_image, perturbed_gen_image, x_real, perturbed_image, attr_name)


                    if self.run is not None:
                        self.run["train/reward"].append(float(reward))
                        self.run["train/L1"].append(float(defense_l1_loss))
                        self.run["train/L2"].append(float(defense_l2_loss))
                        self.run["train/LPIPS"].append(float(invisibility_lpips))
                        self.run["train/SSIM"].append(float(invisibility_ssim))


                    if isinstance(reward, torch.Tensor):
                        total_reward_this_episode += reward.item()
                    else:
                        total_reward_this_episode += reward


                    reward_tensor = torch.tensor([reward], dtype=torch.float32).to(self.device)







                    next_state = self.get_state(perturbed_image, perturbed_gen_image)




                    n_step_buffer_test_attack.append((state, action, reward_tensor, next_state, torch.tensor([False])))


                    if len(n_step_buffer_test_attack) == self.n_step:
                        state_n_step, action_n_step, reward_n_step, next_state_n_step, done_n_step = self._get_n_step_transition(n_step_buffer_test_attack)
                        self.memory.push(state_n_step, action_n_step, reward_n_step.unsqueeze(0), next_state_n_step, done_n_step)


                    if len(self.memory) >= 5:

                        batch, weights, indices = self.memory.sample(self.batch_size, frame_idx)
                        loss_val, priorities = self.rl_agent.update_model(batch, weights, indices, self.batch_size, frame_idx)
                        self.memory.update_priorities(indices, priorities)


                    self.rl_agent.reset_noise()

                reward_per_episode.append(total_reward_this_episode)





                analyzed_perturbation_array = analyze_perturbation(perturbed_image - x_real)
                total_perturbation_map += analyzed_perturbation_array



                with torch.no_grad():
                    remain_perturb_array = analyze_perturbation(perturbed_image - x_real)
                    results["원본(변형없음)"]["total_remain_map"] += remain_perturb_array

                    perturbed_gen_image_orig = self.diffusionclip_wrapper.forward_edit(perturbed_image, attr_name)

                    noattack_result_list.append(perturbed_image)
                    noattack_result_list.append(original_gen_image)
                    noattack_result_list.append(perturbed_gen_image_orig)
                    orig_m = original_gen_image.clamp(-1.0, 1.0)
                    pert_m = perturbed_gen_image_orig.clamp(-1.0, 1.0)
                    results = calculate_and_save_metrics(orig_m, pert_m, "원본(변형없음)", results)


                x_adv_jpeg = compress_jpeg(perturbed_image, quality=75)
                with torch.no_grad():
                    remain_perturb_array = analyze_perturbation(x_adv_jpeg - x_real)
                    results["JPEG압축"]["total_remain_map"] += remain_perturb_array

                    perturbed_gen_image_jpeg = self.diffusionclip_wrapper.forward_edit(x_adv_jpeg, attr_name)

                    jpeg_result_list.append(x_adv_jpeg)
                    jpeg_result_list.append(original_gen_image)
                    jpeg_result_list.append(perturbed_gen_image_jpeg)
                    orig_m = original_gen_image.clamp(-1.0, 1.0)
                    pert_m = perturbed_gen_image_jpeg.clamp(-1.0, 1.0)
                    results = calculate_and_save_metrics(orig_m, pert_m, "JPEG압축", results)


                x_adv_denoise_opencv = denoise_opencv(perturbed_image)
                with torch.no_grad():
                    remain_perturb_array = analyze_perturbation(x_adv_denoise_opencv - x_real)
                    results["OpenCV디노이즈"]["total_remain_map"] += remain_perturb_array

                    perturbed_gen_image_opencv = self.diffusionclip_wrapper.forward_edit(x_adv_denoise_opencv, attr_name)

                    opencv_result_list.append(x_adv_denoise_opencv)
                    opencv_result_list.append(original_gen_image)
                    opencv_result_list.append(perturbed_gen_image_opencv)
                    orig_m = original_gen_image.clamp(-1.0, 1.0)
                    pert_m = perturbed_gen_image_opencv.clamp(-1.0, 1.0)
                    results = calculate_and_save_metrics(orig_m, pert_m, "OpenCV디노이즈", results)


                x_adv_median = denoise_scikit(perturbed_image)
                with torch.no_grad():
                    remain_perturb_array = analyze_perturbation(x_adv_median - x_real)
                    results["중간값스무딩"]["total_remain_map"] += remain_perturb_array

                    perturbed_gen_image_median = self.diffusionclip_wrapper.forward_edit(x_adv_median, attr_name)

                    median_result_list.append(x_adv_median)
                    median_result_list.append(original_gen_image)
                    median_result_list.append(perturbed_gen_image_median)
                    orig_m = original_gen_image.clamp(-1.0, 1.0)
                    pert_m = perturbed_gen_image_median.clamp(-1.0, 1.0)
                    results = calculate_and_save_metrics(orig_m, pert_m, "중간값스무딩", results)


                x_real_padding, x_adv_padding = random_resize_padding(x_real, perturbed_image)
                with torch.no_grad():
                    remain_perturb_array = analyze_perturbation(x_adv_padding - x_real_padding)
                    results["크기조정패딩"]["total_remain_map"] += remain_perturb_array

                    original_gen_image_padding = self.diffusionclip_wrapper.forward_edit(x_real_padding, attr_name)
                    perturbed_gen_image_padding = self.diffusionclip_wrapper.forward_edit(x_adv_padding, attr_name)

                    padding_result_list.append(x_adv_padding)
                    padding_result_list.append(original_gen_image_padding)
                    padding_result_list.append(perturbed_gen_image_padding)
                    orig_m = original_gen_image_padding.clamp(-1.0, 1.0)
                    pert_m = perturbed_gen_image_padding.clamp(-1.0, 1.0)
                    results = calculate_and_save_metrics(orig_m, pert_m, "크기조정패딩", results)


                x_real_transforms, x_adv_transforms = random_image_transforms(x_real, perturbed_image)
                with torch.no_grad():
                    remain_perturb_array = analyze_perturbation(x_adv_transforms - x_real_transforms)
                    results["이미지변환"]["total_remain_map"] += remain_perturb_array

                    original_gen_image_transforms = self.diffusionclip_wrapper.forward_edit(x_real_transforms, attr_name)
                    perturbed_gen_image_transforms= self.diffusionclip_wrapper.forward_edit(x_adv_transforms, attr_name)

                    transforms_result_list.append(x_adv_transforms)
                    transforms_result_list.append(original_gen_image_transforms)
                    transforms_result_list.append(perturbed_gen_image_transforms)
                    orig_m = original_gen_image_transforms.clamp(-1.0, 1.0)
                    pert_m = perturbed_gen_image_transforms.clamp(-1.0, 1.0)
                    results = calculate_and_save_metrics(orig_m, pert_m, "이미지변환", results)


                with torch.no_grad():

                    x_real_np = x_real.squeeze(0).permute(1, 2, 0).cpu().numpy()
                    perturbed_image_np = perturbed_image.squeeze(0).permute(1, 2, 0).cpu().numpy()

                    invisible_lpips_value = self.lpips_loss(x_real, perturbed_image).mean()

                    invisible_psnr_value = psnr(x_real_np, perturbed_image_np, data_range=2.0)
                    invisible_ssim_value = ssim(x_real_np, perturbed_image_np, data_range=2.0, win_size=3, channel_axis=2)

                    total_invisible_lpips += invisible_lpips_value
                    total_invisible_psnr += invisible_psnr_value
                    total_invisible_ssim += invisible_ssim_value

                    episode += 1



                if episode % self.target_update_interval == 0:

                    self.rl_agent.update_target_net()
                    self.rl_agent.reset_noise()



            all_result_lists = [noattack_result_list, jpeg_result_list, opencv_result_list, median_result_list, padding_result_list, transforms_result_list]
            row_images = []
            for result_list in all_result_lists:
                row_concat = torch.cat(result_list, dim=3)
                row_images.append(row_concat)






            spacing = 10
            blank_image = torch.ones_like(row_images[0][:, :, :spacing, :])
            blank_image = blank_image * 1.0


            vertical_concat_list = [row_images[0]]

            for i in range(1, len(row_images)):
                vertical_concat_list.append(blank_image)
                vertical_concat_list.append(row_images[i])

            x_concat = torch.cat(vertical_concat_list, dim=2)
            result_path = os.path.join(self.result_dir, '{}-images.jpg'.format(test_img_idx + 1))

            x_concat_safe = x_concat.detach().clamp(-1.0, 1.0).cpu()
            save_image(self.denorm(x_concat_safe), result_path, nrow=1, padding=0)




            try:
                from collections import Counter

                image_actions = action_history[image_action_start:]
                action_counts = Counter(int(a) for a in image_actions)
                total_actions = max(1, len(image_actions))
                ordered = [action_counts.get(i, 0) for i in range(int(self.action_dim))]
                ratios = [c / total_actions for c in ordered]
                print(
                    "[Action Distribution: Image] "
                    + ", ".join(
                        [f"a{i}={ordered[i]}({ratios[i]:.1%})" for i in range(int(self.action_dim))]
                    )
                )
            except Exception as e:
                print(f"[Action Distribution: Image] Failed to compute: {e}")



            checkpoint_path = os.path.join(self.model_save_dir, f'final_rainbow_dqn.pth')
            try:
                torch.save({
                'rainbow_dqn_state_dict': self.rl_agent.dqn.state_dict(),
                'optimizer_state_dict': self.rl_agent.optimizer.state_dict(),
                }, checkpoint_path)
                print(f"[*] Saved Rainbow DQN agent weights and optimizer weights (episode {episode}) -> {checkpoint_path}")
            except Exception as e:
                print(f"[!] Error saving Rainbow DQN weights and optimizer weights (episode {episode}): {e}")


            if test_img_idx >= (self.training_image_num - 1):
                break


        save_reward_moving_average_txt(reward_per_episode, window_size=25, save_path=os.path.join(self.result_dir, "reward_moving_avg.txt"))
        score = print_comprehensive_metrics(results, episode, total_invisible_psnr, total_invisible_ssim, total_invisible_lpips)
        visualize_actions(action_history, image_indices, attr_indices, step_indices)


        try:
            from collections import Counter

            action_counts = Counter(int(a) for a in action_history)
            total_actions = max(1, len(action_history))
            ordered = [action_counts.get(i, 0) for i in range(int(self.action_dim))]
            ratios = [c / total_actions for c in ordered]
            print(
                "[Action Distribution: Overall] "
                + ", ".join([f"a{i}={ordered[i]}({ratios[i]:.1%})" for i in range(int(self.action_dim))])
            )
        except Exception as e:
            print(f"[Action Distribution: Overall] Failed to compute: {e}")



        checkpoint_path = os.path.join(self.model_save_dir, f'final_rainbow_dqn.pth')
        try:
            torch.save({
                'rainbow_dqn_state_dict': self.rl_agent.dqn.state_dict(),
                'optimizer_state_dict': self.rl_agent.optimizer.state_dict(),
            }, checkpoint_path)
            print(f"[*] Saved Rainbow DQN agent weights and optimizer weights to: {checkpoint_path}")
        except Exception as e:
            print(f"[!] Error saving Rainbow DQN agent weights and optimizer weights: {e}")
        print(f"[INFO] Finished saving Rainbow DQN agent weights and optimizer weights: {checkpoint_path}")


        if self.run is not None:
            self.run.stop()

    """Helper function to create an N-step transition"""
    def _get_n_step_transition(self, n_step_buffer):
        state, action = n_step_buffer[0][:2]
        n_step_reward = 0
        next_state = n_step_buffer[-1][3]
        done = False

        for i in range(self.n_step):
            reward, s, d = n_step_buffer[i][2:]
            n_step_reward += reward * (self.gamma ** i)

            d_bool = bool(d)
            next_state, done = (s, d_bool) if d_bool else (next_state, done)
            if d_bool:
                break
        return state, action, n_step_reward, next_state, done

"""
Rainbow DQN Agent (Prioritized Experience Replay, Dueling DQN, Noisy Network, Categorical DQN, Double DQN, N-step Learning)
-> Uses only N-step Loss, applies N-step Transition sampling
"""
class RainbowDQNAgent:
    def __init__(self, state_dim, action_dim, agent_lr, gamma, epsilon_start, epsilon_end, epsilon_decay, target_update_interval,
                v_min, v_max, atom_size, beta_start, beta_frames, prior_eps, n_step, initial_ratio_steps):
        self.action_dim = action_dim
        self.atom_size = atom_size
        self.support = torch.linspace(v_min, v_max, atom_size).to(device)
        self.delta_z = float(v_max - v_min) / (atom_size - 1)

        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.prior_eps = prior_eps
        self.lr = agent_lr
        self.gamma = gamma
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.n_step = n_step
        self.target_update_interval = target_update_interval
        self.frame_idx = 0
        self.v_min = v_min
        self.v_max = v_max


        self.steps_done = 0
        self.initial_ratio_steps = initial_ratio_steps


        self.dqn = RainbowDQNNet(state_dim, action_dim, atom_size, self.support)
        self.dqn_target = RainbowDQNNet(state_dim, action_dim, atom_size, self.support)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.optimizer = Adam(self.dqn.parameters(), lr=self.lr, eps=0.01/32)


    """
    NoisyNet does not use Epsilon-greedy, uses Value-based action selection
    """
    def select_action(self, state):

        return torch.tensor([[1]], device=device)



        self.steps_done += 1
        if self.steps_done <= self.initial_ratio_steps:

            rand_num = random.randint(0, 9)


            if rand_num <= 3:
                return torch.tensor([[0]], device=device)
            elif rand_num == 4:
                return torch.tensor([[1]], device=device)
            elif rand_num == 5:
                return torch.tensor([[2]], device=device)
            else:
                return torch.tensor([[3]], device=device)


        with torch.no_grad():
            self.dqn.reset_noise()
            q_values = self.dqn(state)
            return q_values.argmax(dim=1).view(1, 1)

    """Receives weight, index, and batch data required for PER -> N-step Transition"""
    def update_model(self, batch, weights, indices, batch_size, frame_idx):
        weights = weights.to(device)
        indices = indices.to(device)

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)
        done_mask = torch.cat(batch.done).float()


        loss, elementwise_loss = self._compute_rainbow_dqn_loss(batch, weights, batch_size, frame_idx)


        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.dqn.parameters(), 10.0)
        self.optimizer.step()


        prios = elementwise_loss.sum(dim=1).detach().cpu().numpy() + self.prior_eps
        return loss.item(), prios


    """Added frame_idx, Calculate Categorical DQN Loss (with Double DQN & N-step Loss & N-step Transition)"""
    def _compute_rainbow_dqn_loss(self, batch, weights, batch_size, frame_idx):
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)
        done_mask = torch.cat(batch.done).float()


        reward_batch = reward_batch.to(device)
        done_mask = done_mask.to(device)


        current_dist = self.dqn.get_distribution(state_batch)
        log_p = torch.log(current_dist[range(batch_size), action_batch.squeeze(1)])

        with torch.no_grad():
            self.dqn_target.reset_noise()
            next_action = self.select_action(next_state_batch)
            next_dist = self.dqn_target.get_distribution(next_state_batch)
            next_dist = next_dist[range(batch_size), next_action.squeeze(1)]


            n_step = self.n_step
            gamma = self.gamma
            target_dist = torch.zeros((batch_size, self.atom_size), device=device)

            t_z = reward_batch.reshape(-1, 1) + (1.0 - done_mask.reshape(-1, 1)) * (gamma**n_step) * self.support.unsqueeze(0)
            t_z = t_z.clamp(min=self.v_min, max=self.v_max)

            b = (t_z - self.v_min) / self.delta_z
            l = b.floor().long().clamp(min=0, max=self.atom_size - 1)
            u = b.ceil().long().clamp(min=0, max=self.atom_size - 1)

            offset = (torch.arange(batch_size) * self.atom_size).long().to(device).unsqueeze(1)

            proj_dist = torch.zeros(next_dist.size(), device=device)
            proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
            proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))


        elementwise_loss = -proj_dist * log_p
        loss = torch.mean(weights * elementwise_loss.sum(dim=1))

        return loss, elementwise_loss

    """Target Network Update (Hard Update)"""
    def update_target_net(self):
        self.dqn_target.load_state_dict(self.dqn.state_dict())

    """Policy Network Noise Reset (every step)"""
    def reset_noise(self):
        self.dqn.reset_noise()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))
