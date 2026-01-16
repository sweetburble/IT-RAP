import torch
import torch.nn as nn

import torch_dct as dct

"""
Role: A class that integrally implements FGSM, I-FGSM, and PGD attacks.

Initialization:
epsilon: The maximum allowed noise size for the attack (based on L∞ norm).
a: The noise update size at each step.
"""
class AttackFunction(object):
    def __init__(self, config, model, device=None, epsilon=0.05, a=0.01):
        self.model = model
        self.epsilon = epsilon
        self.a = a
        self.loss_fn = nn.MSELoss().to(device)
        self.device = device

        self.config = config


        self.rand = True


        self.freq_mask_all = self.create_frequency_masks([1, 3, 256, 256], "ALL")
        self.freq_mask_low = self.create_frequency_masks([1, 3, 256, 256], "LOW")
        self.freq_mask_mid = self.create_frequency_masks([1, 3, 256, 256], "MID")
        self.freq_mask_high = self.create_frequency_masks([1, 3, 256, 256], "HIGH")

    """
    Role: Performs a basic I-FGSM attack.

    Operation:
    1. Uses the original image.
    2. Calculates gradients and updates the noise.
    3. At each step, limits the noise size to within epsilon and clips the image pixel values to [-1, 1].
    """
    def PGD(self, X_nat, y, c_trg):
        X = X_nat.clone().detach_()

        iter = self.config.pgd_iter

        for i in range(iter):
            X.requires_grad = True


            output = self.model(X, c_trg, mode='enc-dec')

            self.model.zero_grad()


            loss = self.loss_fn(output, y)
            loss.backward()
            grad = X.grad


            X_adv = X + self.a * grad.sign()


            eta = torch.clamp(X_adv - X_nat, min=-self.epsilon, max=self.epsilon)

            X = torch.clamp(X_nat + eta, min=-1, max=1).detach_()

        self.model.zero_grad()


        return X, X - X_nat


    def perturb_frequency_domain(self, X_nat, y, c_trg, freq_band='ALL', iter=1):

        iter = self.config.dct_iter
        dct_coef = self.config.dct_coefficent
        dct_clamp = self.config.dct_clamp


        X_nat_dct = torch.zeros_like(X_nat)
        for b in range(X_nat.shape[0]):
            for c in range(X_nat.shape[1]):
                X_nat_dct[b, c] = dct.dct_2d(X_nat[b, c])


        if freq_band == 'ALL':
            freq_mask = self.freq_mask_all
        elif freq_band == 'LOW':
            freq_mask = self.freq_mask_low
        elif freq_band == 'MID':
            freq_mask = self.freq_mask_mid
        elif freq_band == 'HIGH':
            freq_mask = self.freq_mask_high
        else:
            raise ValueError(f"Unsupported frequency band: {freq_band}")


        eta_dct = torch.zeros_like(X_nat_dct)


        eta_dct = eta_dct * freq_mask

        for i in range(iter):
            eta_dct.requires_grad = True


            X_dct = X_nat_dct + eta_dct


            X = torch.zeros_like(X_nat)
            for b in range(X_nat.shape[0]):
                for c in range(X_nat.shape[1]):
                    X[b, c] = dct.idct_2d(X_dct[b, c])


            output = self.model(X, c_trg, mode='enc-dec')

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
        masks = torch.zeros(shape).to(self.device)


        if freq_band == 'ALL':
            return torch.ones(shape).to(self.device)


        for b in range(B):
            for c in range(C):


                i_coords = torch.arange(H).reshape(-1, 1).repeat(1, W).to(self.device)
                j_coords = torch.arange(W).reshape(1, -1).repeat(H, 1).to(self.device)


                frequency_map = torch.sqrt(i_coords**2 + j_coords**2)


                max_freq = torch.sqrt(torch.tensor((H-1)**2 + (W-1)**2)).to(self.device)
                frequency_map = frequency_map / max_freq


                if freq_band == 'LOW':
                    masks[b, c] = (frequency_map <= 1/3).float()
                elif freq_band == 'MID':
                    masks[b, c] = ((frequency_map > 1/3) & (frequency_map <= 2/3)).float()
                elif freq_band == 'HIGH':
                    masks[b, c] = (frequency_map > 2/3).float()

        return masks
