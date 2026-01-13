import neptune
import os
import argparse
from dotenv import load_dotenv
from sympy import im
from torch.backends import cudnn

from diffusionclip_solver import SolverRainbow
from stargan_data_loader import get_loader

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

def str2bool(v):
    return v.lower() in ('true')

def main(config):
    # .env 파일 활성화
    load_dotenv()

    # For fast training
    cudnn.benchmark = True
    
    run = neptune.init_run(
        project=os.getenv("NEPTUNE_PROJECT"),
        api_token=os.getenv("NEPTUNE_API_TOKEN"),
    )


    # Create directories if not exist
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)

    # Data loader
    dataset_loader = None

    if config.dataset == 'CelebA':
        dataset_loader = get_loader(config.images_dir, config.attr_path, config.selected_attrs,
                                   config.celeba_crop_size, config.image_size, config.batch_size,
                                   'CelebA', config.mode, config.num_workers, config.start_index)
    elif config.dataset == 'MAADFace':
        dataset_loader = get_loader(config.images_dir, config.attr_path, config.selected_attrs,
                                config.celeba_crop_size, config.image_size, config.batch_size,
                                'MAADFace', config.mode, config.num_workers, config.start_index)

    solver = SolverRainbow(dataset_loader, config, run = run)

    if config.mode == 'train':
        solver.train_attack()

    elif config.mode == 'inference':
        # checkpoint_path = os.path.join(config.model_save_dir, f'rainbow_dqn_final_{config.test_iters}.pth')
        checkpoint_path = os.path.join(config.model_save_dir, f'final_rainbow_dqn.pth') # rainbow_dqn_agent.ckpt
        solver.load_rainbow_dqn_checkpoint(checkpoint_path)

        # # Load StarGAN model (required)
        # solver.restore_model(config.test_iters)

        # # Perform inference
        # solver.inference_rainbow_dqn(dataset_loader, result_dir=config.result_dir)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Existing parameter settings...
    # Model configuration
    parser.add_argument('--c_dim', type=int, default=5, help='dimension of domain labels (1st dataset)')
    parser.add_argument('--c2_dim', type=int, default=8, help='dimension of domain labels (2nd dataset)')
    parser.add_argument('--celeba_crop_size', type=int, default=178, help='crop size for the CelebA dataset')
    parser.add_argument('--image_size', type=int, default=256, help='image resolution')
    parser.add_argument('--g_conv_dim', type=int, default=64, help='number of conv filters in the first layer of G')
    parser.add_argument('--d_conv_dim', type=int, default=64, help='number of conv filters in the first layer of D')
    parser.add_argument('--g_repeat_num', type=int, default=6, help='number of residual blocks in G')
    parser.add_argument('--d_repeat_num', type=int, default=6, help='number of strided conv layers in D')
    parser.add_argument('--lambda_cls', type=float, default=1, help='weight for domain classification loss')
    parser.add_argument('--lambda_rec', type=float, default=10, help='weight for reconstruction loss')
    parser.add_argument('--lambda_gp', type=float, default=10, help='weight for gradient penalty')
    parser.add_argument('--selected_attrs', '--list', nargs='+', help='selected attributes for the CelebA dataset',
                        default=['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young'])
    
    # Training configuration
    parser.add_argument('--dataset', type=str, default='CelebA', choices=['CelebA', 'Both', 'MAADFace'])
    parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for D')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')

    parser.add_argument('--training_image_num', type=int, default=5, help='Number of images used to train Rainbow DQN')
    parser.add_argument('--inference_image_num', type=int, default=5, help='Number of images used to inference Rainbow DQN')
    # Starting index to resume training from the point of interruption when training with the MAAD-FACE dataset
    parser.add_argument('--start_index', type=int, default=0, help='Data index to start training from')
    parser.add_argument('--reward_weight', type=float, default=0.5, help='Reward weight (Deepfake defense: reward_weight, Imperceptibility: 1 - reward_weight)')


    # Test configuration
    parser.add_argument('--test_iters', type=int, default=200000, help='test model from this step')

    # Miscellaneous settings
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'inference']) # Changed mode to train

    # Directory settings
    parser.add_argument('--images_dir', type=str, default='data/celeba/images')
    parser.add_argument('--attr_path', type=str, default='data/celeba/list_attr_celeba.txt')
    parser.add_argument('--model_save_dir', type=str, default='checkpoints/models/diffusionclip')
    parser.add_argument('--result_dir', type=str, default='stargan/result_test') # Changed result_dir

    parser.add_argument('--epsilon_start', type=float, default=0.95, help='epsilon start value for exploration') # Currently not in use
    parser.add_argument('--epsilon_end', type=float, default=0.04, help='epsilon end value for exploration') # Currently not in use
    parser.add_argument('--epsilon_decay', type=int, default=1000, help='epsilon decay steps') # Currently not in use

    # Rainbow DQN Hyperparameters (PER, Categorical DQN, N-step Learning)
    parser.add_argument('--batch_size', type=int, default=1, help='How many images to process at once')
    parser.add_argument('--agent_lr', type=float, default=0.0001, help='learning rate for Agent')
    parser.add_argument('--gamma', type=float, default=0.96, help='discount factor for RL')
    parser.add_argument('--target_update_interval', type=int, default=5, help='target network update interval')
    parser.add_argument('--memory_capacity', type=int, default=512, help='replay memory capacity')
    parser.add_argument('--max_steps_per_episode', type=int, default=20, help='max steps per episode')
    # Action dim: 0=Diff-PGD, 1=DCT-LOW, 2=DCT-MID, 3=DCT-HIGH, 4=Inversion-Attack, 5=Score-Matching
    parser.add_argument('--action_dim', type=int, default=6, help='max action dimension (6 = includes score matching attack)')
    parser.add_argument('--noise_level', type=float, default=0.008, help='noise level for RLAB perturbation (increased)')
    parser.add_argument('--feature_extractor_name', type=str, default="edgeface", help='Image feature extraction for State (mesonet, resnet50, vgg19, ghostfacenets, edgeface)')
    
    # Attack Parameters (Enhanced based on Mist/PhotoGuard/DiffAttack papers)
    parser.add_argument('--inv_attack_iter', type=int, default=6, help='DDIM inversion attack iterations')
    parser.add_argument('--score_attack_iter', type=int, default=10, help='Score matching attack iterations')
    parser.add_argument('--diffusion_t0', type=int, default=400, help='Maximum timestep for diffusion attacks')


    parser.add_argument('--alpha', type=float, default=0.8, help='PER alpha parameter')
    parser.add_argument('--beta_start', type=float, default=0.35, help='PER beta start parameter')
    parser.add_argument('--beta_frames', type=int, default=4000, help='PER beta frames parameter')
    parser.add_argument('--prior_eps', type=float, default=1e-6, help='PER prior epsilon parameter')
    parser.add_argument('--v_min', type=int, default=-5, help='Categorical DQN v_min value') 
    parser.add_argument('--v_max', type=int, default=5, help='Categorical DQN v_max value') 
    parser.add_argument('--atom_size', type=int, default=11, help='Categorical DQN atom size')
    parser.add_argument('--n_step', type=int, default=5, help='N-step Learning step size') 

    parser.add_argument('--pgd_iter', type=int, default=6, help='Action 0, number of PGD iterations')
    parser.add_argument('--dct_iter', type=int, default=4, help='Action 1~3, number of frequency noise insertion iterations')
    parser.add_argument('--dct_coefficent', type=int, default=10, help='DCT noise coefficient')
    parser.add_argument('--dct_clamp', type=float, default=0.40, help='DCT coefficient clamp (smaller -> more invisible)')

    # DiffusionCLIP attack strength
    parser.add_argument('--attack_epsilon', type=float, default=0.35, help='Max Linf perturbation in pixel space [-1,1] (higher = more attack)')
    parser.add_argument('--attack_alpha', type=float, default=0.08, help='Step size for PGD/Diff-PGD/DCT updates (higher = faster change)'
)


    # 여기서부터 Diffusionclip 관련 인자들 추가
    # Mode
    parser.add_argument('--clip_finetune', action='store_true')
    parser.add_argument('--clip_latent_optim', action='store_true')
    parser.add_argument('--edit_images_from_dataset', action='store_true')
    parser.add_argument('--edit_one_image', action='store_true')
    parser.add_argument('--unseen2unseen', action='store_true')
    parser.add_argument('--clip_finetune_eff', action='store_true')
    parser.add_argument('--edit_one_image_eff', action='store_true')
    # Default
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--exp', type=str, default='./runs/', help='Path for saving running related data.')
    parser.add_argument('--comment', type=str, default='', help='A string for experiment comment')
    parser.add_argument('--verbose', type=str, default='info', help='Verbose level: info | debug | warning | critical')
    parser.add_argument('--ni', type=int, default=1,  help="No interaction. Suitable for Slurm Job launcher")
    parser.add_argument('--align_face', type=int, default=1, help='align face or not')
    # Text
    parser.add_argument('--edit_attr', type=str, default=None, help='Attribute to edit defiend in ./utils/text_dic.py')
    parser.add_argument('--src_txts', type=str, action='append', help='Source text e.g. Face')
    parser.add_argument('--trg_txts', type=str, action='append', help='Target text e.g. Angry Face')
    parser.add_argument('--target_class_num', type=str, default=None)
    # Sampling
    parser.add_argument('--t_0', type=int, default=400, help='Return step in [0, 1000)')
    parser.add_argument('--n_inv_step', type=int, default=40, help='# of steps during generative pross for inversion')
    parser.add_argument('--n_train_step', type=int, default=6, help='# of steps during generative pross for train')
    parser.add_argument('--n_test_step', type=int, default=40, help='# of steps during generative pross for test')
    parser.add_argument('--sample_type', type=str, default='ddim', help='ddpm for Markovian sampling, ddim for non-Markovian sampling')
    parser.add_argument('--eta', type=float, default=0.0, help='Controls of varaince of the generative process')
    # Train & Test
    parser.add_argument('--do_train', type=int, default=1, help='Whether to train or not during CLIP finetuning')
    parser.add_argument('--do_test', type=int, default=1, help='Whether to test or not during CLIP finetuning')
    parser.add_argument('--save_train_image', type=int, default=1, help='Wheter to save training results during CLIP fineuning')
    parser.add_argument('--bs_train', type=int, default=1, help='Training batch size during CLIP fineuning')
    parser.add_argument('--bs_test', type=int, default=1, help='Test batch size during CLIP fineuning')
    parser.add_argument('--n_precomp_img', type=int, default=100, help='# of images to precompute latents')
    parser.add_argument('--n_train_img', type=int, default=50, help='# of training images')
    parser.add_argument('--n_test_img', type=int, default=10, help='# of test images')
    parser.add_argument('--model_path', type=str, default=None, help='Test model path')
    parser.add_argument('--img_path', type=str, default=None, help='Image path to test')
    parser.add_argument('--deterministic_inv', type=int, default=1, help='Whether to use deterministic inversion during inference')
    parser.add_argument('--hybrid_noise', type=int, default=0, help='Whether to change multiple attributes by mixing multiple models')
    parser.add_argument('--model_ratio', type=float, default=1, help='Degree of change, noise ratio from original and finetuned model.')
    # Loss & Optimization
    parser.add_argument('--clip_loss_w', type=int, default=3, help='Weights of CLIP loss')
    parser.add_argument('--l1_loss_w', type=float, default=0, help='Weights of L1 loss')
    parser.add_argument('--id_loss_w', type=float, default=0, help='Weights of ID loss')
    parser.add_argument('--clip_model_name', type=str, default='ViT-B/16', help='ViT-B/16, ViT-B/32, RN50x16 etc')
    parser.add_argument('--lr_clip_finetune', type=float, default=2e-6, help='Initial learning rate for finetuning')
    parser.add_argument('--lr_clip_lat_opt', type=float, default=2e-2, help='Initial learning rate for latent optim')
    parser.add_argument('--n_iter', type=int, default=1, help='# of iterations of a generative process with `n_train_img` images')
    parser.add_argument('--scheduler', type=int, default=1, help='Whether to increase the learning rate')
    parser.add_argument('--sch_gamma', type=float, default=1.3, help='Scheduler gamma')
    config = parser.parse_args()


    print(config)
    main(config)