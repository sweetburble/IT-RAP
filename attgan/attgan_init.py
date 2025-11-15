import argparse
import torch
from .attgan import AttGAN

# --- 1. Set parameters that determine the model architecture ---
# Set to match the selected weights "256_shortcut1_inject1_none"
parser = argparse.ArgumentParser()
# (Important) Parameters directly related to the model structure
parser.add_argument('--img_size', dest='img_size', type=int, default=256) # 256
parser.add_argument('--shortcut_layers', dest='shortcut_layers', type=int, default=1) # shortcut1
parser.add_argument('--inject_layers', dest='inject_layers', type=int, default=1) # inject1

# (Reference) Other model structure parameters (Keep default values as they are not specified in the README)
parser.add_argument('--enc_dim', dest='enc_dim', type=int, default=64)
parser.add_argument('--dec_dim', dest='dec_dim', type=int, default=64)
parser.add_argument('--dis_dim', dest='dis_dim', type=int, default=64)
parser.add_argument('--dis_fc_dim', dest='dis_fc_dim', type=int, default=1024)
parser.add_argument('--enc_layers', dest='enc_layers', type=int, default=5)
parser.add_argument('--dec_layers', dest='dec_layers', type=int, default=5)
parser.add_argument('--dis_layers', dest='dis_layers', type=int, default=5)
parser.add_argument('--enc_norm', dest='enc_norm', type=str, default='batchnorm')
parser.add_argument('--dec_norm', dest='dec_norm', type=str, default='batchnorm')
parser.add_argument('--dis_norm', dest='dis_norm', type=str, default='instancenorm')
parser.add_argument('--dis_fc_norm', dest='dis_fc_norm', type=str, default='none')
parser.add_argument('--enc_acti', dest='enc_acti', type=str, default='lrelu')
parser.add_argument('--dec_acti', dest='dec_acti', type=str, default='relu')
parser.add_argument('--dis_acti', dest='dis_acti', type=str, default='lrelu')
parser.add_argument('--dis_fc_acti', dest='dis_fc_acti', type=str, default='relu')

# (Reference) Training-related parameters not used during inference
parser.add_argument('--lambda_1', dest='lambda_1', type=float, default=100.0)
parser.add_argument('--lambda_2', dest='lambda_2', type=float, default=10.0)
parser.add_argument('--lambda_3', dest='lambda_3', type=float, default=1.0)
parser.add_argument('--lambda_gp', dest='lambda_gp', type=float, default=10.0)
parser.add_argument('--mode', dest='mode', default='wgan', choices=['wgan', 'lsgan', 'dcgan'])
parser.add_argument('--lr', dest='lr', type=float, default=0.0002)
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5)
parser.add_argument('--beta2', dest='beta2', type=float, default=0.999)
parser.add_argument('--gpu', action='store_true')

# Parse args
args = parser.parse_args(args=[]) # Pass an empty list when running in environments like Jupyter Notebook
args.n_attrs = 13 # The number of attributes used by AttGAN in the CelebA dataset is 13
args.betas = (args.beta1, args.beta2)
args.gpu = torch.cuda.is_available()

# --- 2. Create AttGAN model and load weights ---
attgan_model = AttGAN(args)

# Path to the downloaded checkpoint file
checkpoint_path = r'.\attgan\256_shortcut1_inject1_none_hq\checkpoint\weights.199.pth'
attgan_model.load(checkpoint_path)
attgan_model.eval() # Switch to inference mode

print("AttGAN model loaded successfully!")

# Now, you can perform image deepfake using attgan_model.G.
# e.g., generated_image = attgan_model.G(input_image, target_attribute, mode='enc-dec')
