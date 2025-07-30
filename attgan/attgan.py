"""AttGAN, generator, and discriminator."""

import torch
import torch.nn as nn
from .nn import LinearBlock, Conv2dBlock, ConvTranspose2dBlock
import torch.autograd as autograd
import torch.nn.functional as F
import torch.optim as optim
import argparse


# This architecture is for images of 128x128
# In the original AttGAN, slim.conv2d uses padding 'same'
MAX_DIM = 64 * 16 # 1024

class Generator(nn.Module):
    """
    Generator class.
    It has an encoder-decoder structure, extracts the latent representation of an image, 
    and generates a new image with transformed attributes based on it.
    """
    def __init__(self, enc_dim=64, enc_layers=5, enc_norm_fn='batchnorm', enc_acti_fn='lrelu',
                 dec_dim=64, dec_layers=5, dec_norm_fn='batchnorm', dec_acti_fn='relu',
                 n_attrs=13, shortcut_layers=1, inject_layers=0, img_size=128):
        super(Generator, self).__init__()
        # Set the number of shortcut and attribute injection layers.
        self.shortcut_layers = min(shortcut_layers, dec_layers - 1)
        self.inject_layers = min(inject_layers, dec_layers - 1)
        self.f_size = img_size // 2**enc_layers  # Feature map size after encoding (e.g., 4x4 for a 128x128 image).
        
        # Encoder layer configuration.
        layers = []
        n_in = 3  # Input image is 3-channel (RGB).
        for i in range(enc_layers):
            n_out = min(enc_dim * 2**i, MAX_DIM)
            layers += [Conv2dBlock(
                n_in, n_out, (4, 4), stride=2, padding=1, norm_fn=enc_norm_fn, acti_fn=enc_acti_fn
            )]
            n_in = n_out
        self.enc_layers = nn.ModuleList(layers)
        
        # Decoder layer configuration.
        layers = []
        n_in = n_in + n_attrs  # Concatenate the attribute vector (n_attrs) to the final output of the encoder.
        for i in range(dec_layers):
            if i < dec_layers - 1:
                n_out = min(dec_dim * 2**(dec_layers-i-1), MAX_DIM)
                layers += [ConvTranspose2dBlock(
                    n_in, n_out, (4, 4), stride=2, padding=1, norm_fn=dec_norm_fn, acti_fn=dec_acti_fn
                )]
                n_in = n_out
                # Shortcut connection: Concatenate the intermediate feature map from the encoder to the output of the decoder.
                if self.shortcut_layers > i:
                    n_in = n_in + n_in//2 
                # Attribute injection: Additionally inject the attribute vector into the intermediate layers of the decoder.
                if self.inject_layers > i:
                    n_in = n_in + n_attrs
            else:
                # The last layer generates a 3-channel (RGB) image and uses tanh as the activation function (range -1 to 1).
                layers += [ConvTranspose2dBlock(
                    n_in, 3, (4, 4), stride=2, padding=1, norm_fn='none', acti_fn='tanh'
                )]
        self.dec_layers = nn.ModuleList(layers)
    
    def encode(self, x):
        """Encoder: Compresses the image into a latent representation (z)."""
        z = x
        zs = [] # Store intermediate feature maps for shortcut connections.
        for layer in self.enc_layers:
            z = layer(z)
            zs.append(z)
        return zs
    
    def decode(self, zs, a):
        """Decoder: Generates an image from the latent representations (zs) and attribute vector (a)."""
        # Transform the dimension of the attribute vector 'a' to match the feature map.
        a_tile = a.view(a.size(0), -1, 1, 1).repeat(1, 1, self.f_size, self.f_size)
        # Combine the deepest latent representation (zs[-1]) with the attribute vector.
        z = torch.cat([zs[-1], a_tile], dim=1)
        
        for i, layer in enumerate(self.dec_layers):
            z = layer(z)
            # Apply shortcut connection: Combine the current decoder output 'z' with the corresponding feature map from the encoder.
            if self.shortcut_layers > i:
                z = torch.cat([z, zs[len(self.dec_layers) - 2 - i]], dim=1)
            # Apply attribute injection: Inject the attribute vector into the intermediate feature maps as well.
            if self.inject_layers > i:
                a_tile = a.view(a.size(0), -1, 1, 1) \
                          .repeat(1, 1, self.f_size * 2**(i+1), self.f_size * 2**(i+1))
                z = torch.cat([z, a_tile], dim=1)
        return z
    
    def forward(self, x, a=None, mode='enc-dec'):
        """
        Forward pass of the generator. Performs encoding, decoding, or the full process depending on the mode.
        """
        if mode == 'enc-dec':
            assert a is not None, 'Attribute vector (a) is not provided.'
            return self.decode(self.encode(x), a)
        if mode == 'enc':
            return self.encode(x)
        if mode == 'dec':
            assert a is not None, 'Attribute vector (a) is not provided.'
            return self.decode(x, a)
        raise Exception('Unknown mode: ' + mode)

class Discriminators(nn.Module):
    """
    Discriminators class.
    Performs two tasks: determines if the input image is real or fake (adversarial), 
    and classifies its attributes (classification).
    """
    def __init__(self, dim=64, norm_fn='instancenorm', acti_fn='lrelu',
                 fc_dim=1024, fc_norm_fn='none', fc_acti_fn='lrelu', n_layers=5, img_size=128):
        super(Discriminators, self).__init__()
        self.f_size = img_size // 2**n_layers
        
        # Convolutional layers for feature extraction from the image.
        layers = []
        n_in = 3
        for i in range(n_layers):
            n_out = min(dim * 2**i, MAX_DIM)
            layers += [Conv2dBlock(
                n_in, n_out, (4, 4), stride=2, padding=1, norm_fn=norm_fn, acti_fn=acti_fn
            )]
            n_in = n_out
        self.conv = nn.Sequential(*layers)
        
        # 1. FC layers for real/fake discrimination (Adversarial head).
        self.fc_adv = nn.Sequential(
            LinearBlock(1024 * self.f_size * self.f_size, fc_dim, fc_norm_fn, fc_acti_fn),
            LinearBlock(fc_dim, 1, 'none', 'none') # Output: 1 (real/fake score).
        )
        # 2. FC layers for attribute classification (Classification head).
        self.fc_cls = nn.Sequential(
            LinearBlock(1024 * self.f_size * self.f_size, fc_dim, fc_norm_fn, fc_acti_fn),
            LinearBlock(fc_dim, 13, 'none', 'none') # Output: 13 (number of attributes).
        )
    
    def forward(self, x):
        """Forward pass of the discriminator. Returns both the real/fake discrimination result and the attribute classification result."""
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        return self.fc_adv(h), self.fc_cls(h)


class AttGAN():
    """
    The main class that manages the entire AttGAN model and oversees the training process.
    """
    def __init__(self, args):
        self.mode = args.mode
        self.gpu = args.gpu
        self.multi_gpu = args.multi_gpu if 'multi_gpu' in args else False
        # Weight parameters for the loss functions.
        self.lambda_1 = args.lambda_1  # Reconstruction loss weight
        self.lambda_2 = args.lambda_2  # Generator's classification loss weight
        self.lambda_3 = args.lambda_3  # Discriminator's classification loss weight
        self.lambda_gp = args.lambda_gp # Gradient penalty weight
        
        # Initialize the Generator (G) and Discriminator (D) models.
        self.G = Generator(
            args.enc_dim, args.enc_layers, args.enc_norm, args.enc_acti,
            args.dec_dim, args.dec_layers, args.dec_norm, args.dec_acti,
            args.n_attrs, args.shortcut_layers, args.inject_layers, args.img_size
        )
        self.G.train()
        if self.gpu: self.G.cuda()
        
        self.D = Discriminators(
            args.dis_dim, args.dis_norm, args.dis_acti,
            args.dis_fc_dim, args.dis_fc_norm, args.dis_fc_acti, args.dis_layers, args.img_size
        )
        self.D.train()
        if self.gpu: self.D.cuda()
        
        if self.multi_gpu:
            self.G = nn.DataParallel(self.G)
            self.D = nn.DataParallel(self.D)
        
        # Optimizer setup.
        self.optim_G = optim.Adam(self.G.parameters(), lr=args.lr, betas=args.betas)
        self.optim_D = optim.Adam(self.D.parameters(), lr=args.lr, betas=args.betas)
    
    def set_lr(self, lr):
        """Dynamically change the learning rate."""
        for g in self.optim_G.param_groups:
            g['lr'] = lr
        for g in self.optim_D.param_groups:
            g['lr'] = lr
    
    def trainG(self, img_a, att_a, att_a_, att_b, att_b_):
        """Train the Generator (G) for one step."""
        # Fix the parameters of the Discriminator (D) so they are not updated.
        for p in self.D.parameters():
            p.requires_grad = False
        
        # 1. Image generation.
        zs_a = self.G(img_a, mode='enc')          # Encode the original image.
        img_fake = self.G(zs_a, att_b_, mode='dec') # Generate a fake image with the target attributes (att_b).
        img_recon = self.G(zs_a, att_a_, mode='dec') # Reconstruct the image with the original attributes (att_a).
        
        # 2. Pass the generated fake image through the discriminator.
        d_fake, dc_fake = self.D(img_fake)
        
        # 3. Loss calculation.
        # 3-1. Adversarial Loss: Train the generator to make the fake image look real (to fool the discriminator).
        if self.mode == 'wgan':
            gf_loss = -d_fake.mean()
        elif self.mode == 'lsgan':
            gf_loss = F.mse_loss(d_fake, torch.ones_like(d_fake))
        elif self.mode == 'dcgan':
            gf_loss = F.binary_cross_entropy_with_logits(d_fake, torch.ones_like(d_fake))
        
        # 3-2. Attribute Classification Loss: Train the generator to ensure the fake image has the target attributes (att_b).
        gc_loss = F.binary_cross_entropy_with_logits(dc_fake, att_b)
        
        # 3-3. Reconstruction Loss: Train to make the reconstructed image identical to the original image (identity preservation).
        gr_loss = F.l1_loss(img_recon, img_a)
        
        # 3-4. Final Generator Loss: Weighted sum of the three losses.
        g_loss = gf_loss + self.lambda_2 * gc_loss + self.lambda_1 * gr_loss
        
        # 4. Backpropagation and parameter update.
        self.optim_G.zero_grad()
        g_loss.backward()
        self.optim_G.step()
        
        errG = {
            'g_loss': g_loss.item(), 'gf_loss': gf_loss.item(),
            'gc_loss': gc_loss.item(), 'gr_loss': gr_loss.item()
        }
        return errG
    
    def trainD(self, img_a, att_a, att_a_, att_b, att_b_):
        """Train the Discriminator (D) for one step."""
        # Set the parameters of the Discriminator (D) to be updated.
        for p in self.D.parameters():
            p.requires_grad = True
        
        # 1. Generate a fake image (no gradient calculation needed).
        img_fake = self.G(img_a, att_b_).detach()
        
        # 2. Pass the real and fake images through the discriminator.
        d_real, dc_real = self.D(img_a)
        d_fake, dc_fake = self.D(img_fake)
        
        # Gradient Penalty calculation function (for WGAN-GP training stabilization).
        def gradient_penalty(f, real, fake=None):
            # Create interpolation points between real and fake data.
            def interpolate(a, b=None):
                if b is None:
                    beta = torch.rand_like(a)
                    b = a + 0.5 * a.var().sqrt() * beta
                alpha = torch.rand(a.size(0), 1, 1, 1).cuda() if self.gpu else torch.rand(a.size(0), 1, 1, 1)
                inter = a + alpha * (b - a)
                return inter
            
            x = interpolate(real, fake).requires_grad_(True)
            pred = f(x)
            if isinstance(pred, tuple):
                pred = pred[0]
            
            # Calculate the gradient at the interpolated points.
            grad = autograd.grad(
                outputs=pred, inputs=x,
                grad_outputs=torch.ones_like(pred),
                create_graph=True, retain_graph=True, only_inputs=True
            )[0]
            
            # Calculate the loss to make the norm of the gradient close to 1.
            grad = grad.view(grad.size(0), -1)
            norm = grad.norm(2, dim=1)
            gp = ((norm - 1.0) ** 2).mean()
            return gp
        
        # 3. Loss calculation.
        # 3-1. Adversarial Loss: Train the discriminator to distinguish real from fake.
        if self.mode == 'wgan':
            wd = d_real.mean() - d_fake.mean()
            df_loss = -wd
            df_gp = gradient_penalty(self.D, img_a, img_fake)
        elif self.mode == 'lsgan':
            df_loss = F.mse_loss(d_real, torch.ones_like(d_fake)) + F.mse_loss(d_fake, torch.zeros_like(d_fake))
            df_gp = gradient_penalty(self.D, img_a)
        elif self.mode == 'dcgan':
            df_loss = F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real)) + F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake))
            df_gp = gradient_penalty(self.D, img_a)
        
        # 3-2. Attribute Classification Loss: Train the discriminator to correctly classify the attributes (att_a) of the real image.
        dc_loss = F.binary_cross_entropy_with_logits(dc_real, att_a)
        
        # 3-3. Final Discriminator Loss: Weighted sum of the three losses (adversarial, gradient penalty, classification).
        d_loss = df_loss + self.lambda_gp * df_gp + self.lambda_3 * dc_loss
        
        # 4. Backpropagation and parameter update.
        self.optim_D.zero_grad()
        d_loss.backward()
        self.optim_D.step()
        
        errD = {
            'd_loss': d_loss.item(), 'df_loss': df_loss.item(), 
            'df_gp': df_gp.item(), 'dc_loss': dc_loss.item()
        }
        return errD
    
    # Functions for changing, saving, and loading model states.
    def train(self):
        self.G.train()
        self.D.train()
    
    def eval(self):
        self.G.eval()
        self.D.eval()
    
    def save(self, path):
        """Save the model and optimizer states."""
        states = {
            'G': self.G.state_dict(),
            'D': self.D.state_dict(),
            'optim_G': self.optim_G.state_dict(),
            'optim_D': self.optim_D.state_dict()
        }
        torch.save(states, path)
    
    def load(self, path):
        """Load the saved model and optimizer states."""
        states = torch.load(path, map_location=lambda storage, loc: storage)
        if 'G' in states:
            self.G.load_state_dict(states['G'])
        if 'D' in states:
            self.D.load_state_dict(states['D'])
        if 'optim_G' in states:
            self.optim_G.load_state_dict(states['optim_G'])
        if 'optim_D' in states:
            self.optim_D.load_state_dict(states['optim_D'])
    
    def saveG(self, path):
        """Save only the weights of the Generator (G)."""
        states = {
            'G': self.G.state_dict()
        }
        torch.save(states, path)

if __name__ == '__main__':
    # When this script is executed directly, this section parses command-line arguments
    # to configure and initialize the AttGAN model.
    # Used for model testing or standalone execution.
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_size', dest='img_size', type=int, default=256)
    parser.add_argument('--shortcut_layers', dest='shortcut_layers', type=int, default=1)
    parser.add_argument('--inject_layers', dest='inject_layers', type=int, default=1)
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
    parser.add_argument('--lambda_1', dest='lambda_1', type=float, default=100.0)
    parser.add_argument('--lambda_2', dest='lambda_2', type=float, default=10.0)
    parser.add_argument('--lambda_3', dest='lambda_3', type=float, default=1.0)
    parser.add_argument('--lambda_gp', dest='lambda_gp', type=float, default=10.0)
    parser.add_argument('--mode', dest='mode', default='wgan', choices=['wgan', 'lsgan', 'dcgan'])
    parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--beta1', dest='beta1', type=float, default=0.5)
    parser.add_argument('--beta2', dest='beta2', type=float, default=0.999)
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()
    args.n_attrs = 13
    args.betas = (args.beta1, args.beta2)
    attgan = AttGAN(args)