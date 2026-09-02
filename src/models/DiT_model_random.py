
# Import required libraries, models, utilities, and loss functions
import random
from src.models.modules.OpenAI_Unet import UNetModel as OpenAI_UNet
from src.models.modules.OpenAI_Unet_without_skip import UNetModel as OpenAI_UNet_without_skip
from PIL import Image
import math
import copy
import torch
from torch import nn
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
from PIL import Image
from einops import rearrange, reduce
from src.utils.generate_noise import gen_noise
import numpy as np
from pytorch_lightning.core.lightning import LightningModule
import torch.optim as optim
from typing import Any, List
from src.utils.patch_sampling import BoxSampler
from src.utils.generate_noise import gen_noise
import os 
import json
# import cv2
from src.utils.ssim_train import SSIM
ssim_loss = SSIM()
# CORRECT
from src.models.modules.DiT import DiT_models
import lpips
from src.models.modules.guassian import GaussianDiffusion
lpips_loss = lpips.LPIPS(net='vgg', verbose=False).eval()
for param in lpips_loss.parameters():
    param.requires_grad = False
class DiT_class(LightningModule):
    def __init__(self,cfg,prefix=None):
        super().__init__()
        self.loss_history = []
        
        self.cfg = cfg
 
        print("num_res_blocks", cfg.get('num_res_blocks',3))
        
        self.dit = DiT_models['DiT-B/2'](
            input_size=64, 
            in_channels=256, 
            num_classes=1  # <--- CHANGE THIS from 4 to 1
        )
        # self.dit.convert_to_fp16()
        
        # Define modality-specific 4 UNets without skip connections for encoder and decoders
        self.model_1 = OpenAI_UNet_without_skip(
                            image_size =  (int(cfg.imageDim[0] / cfg.rescaleFactor),int(cfg.imageDim[1] / cfg.rescaleFactor)),
                            in_channels = 4,
                            model_channels = 64,
                            out_channels = 1,
                            num_res_blocks = 5,
                            attention_resolutions = (int(0),int(0)),
                            dropout=cfg.get('dropout_unet',0.1), 
                            channel_mult=[1, 2, 4],
                            conv_resample=True,
                            dims=2,
                            use_checkpoint=True,
                            use_fp16=True,
                            num_heads=cfg.get('num_heads',4),
                            num_head_channels=64,
                            num_heads_upsample=-1,
                            use_scale_shift_norm=True,
                            resblock_updown=True,
                            use_new_attention_order=True,
                            use_spatial_transformer=True,    
                            transformer_depth=1,
                            context_dim=256,              
                            )
        self.model_1.convert_to_fp16()
        
        self.model_2 = OpenAI_UNet_without_skip(
                            image_size =  (int(cfg.imageDim[0] / cfg.rescaleFactor),int(cfg.imageDim[1] / cfg.rescaleFactor)),
                            in_channels = 1,
                            model_channels = 64,
                            out_channels = 1,
                            num_res_blocks = 5,
                            attention_resolutions = (int(0),int(0)),
                            dropout=cfg.get('dropout_unet',0.1), 
                            channel_mult=[1, 2, 4],
                            conv_resample=True,
                            dims=2,
                            use_checkpoint=True,
                            use_fp16=True,
                            num_heads=cfg.get('num_heads',4),
                            num_head_channels=64,
                            num_heads_upsample=-1,
                            use_scale_shift_norm=True,
                            resblock_updown=True,
                            use_new_attention_order=True,
                            use_spatial_transformer=True,    
                            transformer_depth=1,
                            context_dim=256,              
                            )
        self.model_2.convert_to_fp16()
        
        self.model_3 = OpenAI_UNet_without_skip(
                            image_size =  (int(cfg.imageDim[0] / cfg.rescaleFactor),int(cfg.imageDim[1] / cfg.rescaleFactor)),
                            in_channels = 1,
                            model_channels = 64,
                            out_channels = 1,
                            num_res_blocks = 5,
                            attention_resolutions = (int(0),int(0)),
                            dropout=cfg.get('dropout_unet',0.1), 
                            channel_mult=[1, 2, 4],
                            conv_resample=True,
                            dims=2,
                            use_checkpoint=True,
                            use_fp16=True,
                            num_heads=cfg.get('num_heads',4),
                            num_head_channels=64,
                            num_heads_upsample=-1,
                            use_scale_shift_norm=True,
                            resblock_updown=True,
                            use_new_attention_order=True,
                            use_spatial_transformer=True,    
                            transformer_depth=1,
                            context_dim=256,              
                            )
        self.model_3.convert_to_fp16()
        self.model_4 = OpenAI_UNet_without_skip(
                            image_size =  (int(cfg.imageDim[0] / cfg.rescaleFactor),int(cfg.imageDim[1] / cfg.rescaleFactor)),
                            in_channels = 1,
                            model_channels = 64,
                            out_channels = 1,
                            num_res_blocks = 5,
                            attention_resolutions = (int(0),int(0)),
                            dropout=cfg.get('dropout_unet',0.1), 
                            channel_mult=[1, 2, 4],
                            conv_resample=True,
                            dims=2,
                            use_checkpoint=True,
                            use_fp16=True,
                            num_heads=cfg.get('num_heads',4),
                            num_head_channels=64,
                            num_heads_upsample=-1,
                            use_scale_shift_norm=True,
                            resblock_updown=True,
                            use_new_attention_order=True,
                            use_spatial_transformer=True,    
                            transformer_depth=1,
                            context_dim=256,              
                            )
        self.model_4.convert_to_fp16()
        
        # 
        self.num_timesteps = 1000
        self.test_timesteps = cfg.get('test_timesteps',150) 
        sampling_timesteps = cfg.get('sampling_timesteps',self.test_timesteps)
        self.image_size = (int(cfg.imageDim[0] / cfg.rescaleFactor),int(cfg.imageDim[1] / cfg.rescaleFactor))
        
        # Gaussian Diffusion process
        self.diffusion = GaussianDiffusion(
        self.dit,
        image_size = (int(cfg.imageDim[0] / cfg.rescaleFactor),int(cfg.imageDim[1] / cfg.rescaleFactor)), 
        timesteps = self.num_timesteps,   
        sampling_timesteps = sampling_timesteps,
        objective = cfg.get('objective','pred_x0'), 
        channels = 256,
        loss_type = cfg.get('loss','l1'),  
        p2_loss_weight_gamma = cfg.get('p2_gamma',0),
        inpaint = cfg.get('inpaint',False),
        cfg=cfg
        )     

        self.prefix = prefix 
        self.save_hyperparameters()
    
    def norm_lpips(self,x):
        return x/5.0
    
    @torch.no_grad()
    def inf_onestep(self, batch, specific_modalities=[0], T_start=500, ddim_steps=25):
                # --- 1. Selection Logic ---
                images = batch['vol']['data'] if isinstance(batch, dict) else batch
                if images.ndim == 5: images = images.squeeze(-1)
                images_res = F.interpolate(images[:, :4], size=(256, 256), mode='bilinear')
                B, device = images_res.shape[0], images_res.device
                
                X_stacked = images_res
                # X_all_modalities = X_stacked[torch.arange(B), index]
                batch_size, _, H, W = X_stacked.shape
                device = X_stacked.device
                
                encoder_input = torch.zeros((batch_size, 4, H, W), device=device)
                
                for i in range(batch_size):
                    
                    # --- MODIFIED: Use the specific modalities passed to the function ---
                    available_indices = torch.tensor(specific_modalities, device=device)
                    k = len(available_indices)
                    
                    indices_to_gather = available_indices.tolist()
                    
                    # --- KEPT: Random repetition to fill up to 4 channels ---
                    while len(indices_to_gather) < 4:
                        random_choice = available_indices[torch.randint(0, k, (1,)).item()].item()
                        indices_to_gather.append(random_choice)
                        
                    indices_tensor = torch.tensor(indices_to_gather, device=device) # Shape (4,)
                    
                    # --- KEPT: Random shuffle of those chosen indices ---
                    # This ensures the model doesn't overfit to a specific channel order
                    shuffled_indices = indices_tensor[torch.randperm(4)]
                    
                    # E. Construct the input for this batch item
                    encoder_input[i] = X_stacked[i, shuffled_indices, :, :]

                t_0 = torch.zeros((B,), device=device).long()
                feat = self.model_1.Encoder(encoder_input, t_0)
                x_latent = self.diffusion.q_sample(feat, torch.full((B,), T_start, device=device).long())

                # --- 2. DDIM Sampling (Eta=0) ---
                # Define the sub-sequence of times
                # times = torch.linspace(-1, T_start - 1, steps=ddim_steps + 1, device=device).long()
                # alphas_cumprod = self.diffusion.alphas_cumprod
                y_class = torch.zeros((B,), device=device).long()

                # for i in reversed(range(1, len(times))):
                #     t_idx, prev_idx = times[i], times[i-1]
                    
                #     # Predict x0 from the current latent
                pred_x0 = self.dit(x_latent, torch.full((B,), T_start, device=device), y_class)
                    


                return self._decode_all(x_latent, t_0)
    
    @torch.no_grad()
    def inf_ddim(self, batch, specific_modalities=[0], T_start=500, ddim_steps=25):
                # --- 1. Selection Logic ---
                images = batch['vol']['data'] if isinstance(batch, dict) else batch
                if images.ndim == 5: images = images.squeeze(-1)
                images_res = F.interpolate(images[:, :4], size=(256, 256), mode='bilinear')
                B, device = images_res.shape[0], images_res.device
                
                X_stacked = images_res
                # X_all_modalities = X_stacked[torch.arange(B), index]
                batch_size, _, H, W = X_stacked.shape
                device = X_stacked.device
                
                encoder_input = torch.zeros((batch_size, 4, H, W), device=device)
                
                for i in range(batch_size):
                    
                    # --- MODIFIED: Use the specific modalities passed to the function ---
                    available_indices = torch.tensor(specific_modalities, device=device)
                    k = len(available_indices)
                    
                    indices_to_gather = available_indices.tolist()
                    
                    # --- KEPT: Random repetition to fill up to 4 channels ---
                    while len(indices_to_gather) < 4:
                        random_choice = available_indices[torch.randint(0, k, (1,)).item()].item()
                        indices_to_gather.append(random_choice)
                        
                    indices_tensor = torch.tensor(indices_to_gather, device=device) # Shape (4,)
                    
                    # --- KEPT: Random shuffle of those chosen indices ---
                    # This ensures the model doesn't overfit to a specific channel order
                    shuffled_indices = indices_tensor[torch.randperm(4)]
                    
                    # E. Construct the input for this batch item
                    encoder_input[i] = X_stacked[i, shuffled_indices, :, :]

                t_0 = torch.zeros((B,), device=device).long()
                feat = self.model_1.Encoder(encoder_input, t_0)
                # feat = 2*(feat - feat.min()) / (feat.max() - feat.min() + 1e-8) - 1
                x_latent = self.diffusion.q_sample(feat, torch.full((B,), T_start, device=device).long())

                # --- 2. DDIM Sampling (Eta=0) ---
                # Define the sub-sequence of times
                times = torch.linspace(-1, T_start - 1, steps=ddim_steps + 1, device=device).long()
                alphas_cumprod = self.diffusion.alphas_cumprod
                y_class = torch.zeros((B,), device=device).long()

                for i in reversed(range(1, len(times))):
                    t_idx, prev_idx = times[i], times[i-1]
                    
                    # Predict x0 from the current latent
                    pred_x0 = self.dit(x_latent, torch.full((B,), t_idx, device=device), y_class)
                    
                    a_t = alphas_cumprod[t_idx]
                    a_prev = alphas_cumprod[prev_idx] if prev_idx >= 0 else torch.tensor(1.0, device=device)
                    
                    # Predict noise direction (eps)
                    # Formula: eps = (x_t - sqrt(alpha_t) * pred_x0) / sqrt(1 - alpha_t)
                    eps = (x_latent - torch.sqrt(a_t) * pred_x0) / torch.sqrt(1 - a_t)
                    
                    # Compute x_prev (Deterministic update)
                    # Formula: x_{t-1} = sqrt(alpha_{prev}) * pred_x0 + sqrt(1 - alpha_{prev}) * eps
                    x_latent = torch.sqrt(a_prev) * pred_x0 + torch.sqrt(1 - a_prev) * eps

                return self._decode_all(x_latent, t_0)
        
    @torch.no_grad()
    def inf_ddim_test(self, batch, specific_modalities=[0], T_start=500, ddim_steps=25):
                # --- 1. Selection Logic ---
                images = batch['vol']['data'] if isinstance(batch, dict) else batch
                if images.ndim == 5: images = images.squeeze(-1)
                images_res = F.interpolate(images[:, :4], size=(256, 256), mode='bilinear')
                B, device = images_res.shape[0], images_res.device
                
                X_stacked = images_res
                # X_all_modalities = X_stacked[torch.arange(B), index]
                batch_size, _, H, W = X_stacked.shape
                device = X_stacked.device
                
                encoder_input = torch.zeros((batch_size, 4, H, W), device=device)
                
                for i in range(batch_size):
                    
                    # --- MODIFIED: Use the specific modalities passed to the function ---
                    available_indices = torch.tensor(specific_modalities, device=device)
                    k = len(available_indices)
                    
                    indices_to_gather = available_indices.tolist()
                    
                    # --- KEPT: Random repetition to fill up to 4 channels ---
                    while len(indices_to_gather) < 4:
                        random_choice = available_indices[torch.randint(0, k, (1,)).item()].item()
                        indices_to_gather.append(random_choice)
                        
                    indices_tensor = torch.tensor(indices_to_gather, device=device) # Shape (4,)
                    
                    
                    # E. Construct the input for this batch item
                    encoder_input[i] = X_stacked[i, indices_tensor , :, :]

                t_0 = torch.zeros((B,), device=device).long()
                feat = self.model_1.Encoder(encoder_input, t_0)
                # feat = 2*(feat - feat.min()) / (feat.max() - feat.min() + 1e-8) - 1
                x_latent = self.diffusion.q_sample(feat, torch.full((B,), T_start, device=device).long())

                # --- 2. DDIM Sampling (Eta=0) ---
                # Define the sub-sequence of times
                times = torch.linspace(-1, T_start - 1, steps=ddim_steps + 1, device=device).long()
                alphas_cumprod = self.diffusion.alphas_cumprod
                y_class = torch.zeros((B,), device=device).long()

                for i in reversed(range(1, len(times))):
                    t_idx, prev_idx = times[i], times[i-1]
                    
                    # Predict x0 from the current latent
                    pred_x0 = self.dit(x_latent, torch.full((B,), t_idx, device=device), y_class)
                    
                    a_t = alphas_cumprod[t_idx]
                    a_prev = alphas_cumprod[prev_idx] if prev_idx >= 0 else torch.tensor(1.0, device=device)
                    
                    # Predict noise direction (eps)
                    # Formula: eps = (x_t - sqrt(alpha_t) * pred_x0) / sqrt(1 - alpha_t)
                    eps = (x_latent - torch.sqrt(a_t) * pred_x0) / torch.sqrt(1 - a_t)
                    
                    # Compute x_prev (Deterministic update)
                    # Formula: x_{t-1} = sqrt(alpha_{prev}) * pred_x0 + sqrt(1 - alpha_{prev}) * eps
                    x_latent = torch.sqrt(a_prev) * pred_x0 + torch.sqrt(1 - a_prev) * eps

                return self._decode_all(x_latent, t_0)
        

    @torch.no_grad()
    def inf_ddpm(self, batch, index, T_start=500):
        # --- 1. Modality Selection (Your logic) ---
        images = batch['vol']['data'] if isinstance(batch, dict) else batch
        if images.ndim == 5: images = images.squeeze(-1) # Handle (B,C,H,W,1)
        
        images_res = F.interpolate(images[:, :4], size=(256, 256), mode='bilinear')
        B, device = images_res.shape[0], images_res.device
        
        # Select single modality [B, 1, 256, 256]
        X_stacked = torch.stack([images_res[:, i:i+1] for i in range(4)], dim=1)
        modality_selected = X_stacked[torch.arange(B), index]
        
        # --- 2. Latent Encoding & Noise ---
        t_0 = torch.zeros((B,), device=device).long()
        feat = self.model_1.Encoder(modality_selected, t_0)
        x_latent = self.diffusion.q_sample(feat, torch.full((B,), T_start, device=device).long())

        # --- 3. Deterministic Sampling (No Noise Term) ---
        y_class = torch.zeros((B,), device=device).long()
        for t in reversed(range(0, T_start + 1)):
            t_batch = torch.full((B,), t, device=device)
            pred_x0 = self.dit(x_latent, t_batch, y_class)
            
            if t > 0:
                alpha_bar_t = self.diffusion.alphas_cumprod[t]
                alpha_bar_prev = self.diffusion.alphas_cumprod[t-1]
                beta_t = self.diffusion.betas[t]
                alpha_t = alpha_bar_t / alpha_bar_prev

                # Posterior Mean Coefficients
                c0 = (torch.sqrt(alpha_bar_prev) * beta_t) / (1. - alpha_bar_t)
                ct = (torch.sqrt(alpha_t) * (1. - alpha_bar_prev)) / (1. - alpha_bar_t)
                
                # Update x_latent (Purely deterministic: no + sigma*z)
                x_latent = c0 * pred_x0 + ct * x_latent
            else:
                x_latent = pred_x0

        return self._decode_all(x_latent, t_0)

    def _decode_all(self, x_latent, t_0):
        decoders = [self.model_1, self.model_2, self.model_3, self.model_4]
        return [F.interpolate(d.Decoder(x_latent, t_0).cpu(), size=(160, 200), mode='bicubic') for d in decoders]



        
    def training_step(self, batch, batch_idx: int):
        # --- 1. Unpack Batch ---
        if isinstance(batch, dict):
            images = batch['vol']['data']
        else:
            images = batch
        
        # Ensure 4D tensor (B, C, H, W) before separating
        if images.ndim == 5:
            images = images.squeeze(2)

        # --- 2. Resize & Separate ---
        # Resize to (256, 256)
        images_resized = F.interpolate(images.squeeze(-1), size=(256, 256), mode='bilinear', align_corners=False)

        # Extract individual modalities (B, 1, H, W)
        X_t1c   = images_resized[:, 0:1, :, :]
        X_t1n   = images_resized[:, 1:2, :, :]
        X_t2    = images_resized[:, 2:3, :, :]
        X_flair = images_resized[:, 3:4, :, :]
        seg_org = images_resized[:, 4:5, :, :] 

        # --- 3. Modality Mixing Logic ---
        # Concatenate to (B, 4, H, W) representing [T1c, T1n, T2, Flair]
        # We use cat instead of stack so we don't add an extra dimension
        X_all_modalities = torch.cat([X_t1c, X_t1n, X_t2, X_flair], dim=1)
        
        batch_size, _, H, W = X_all_modalities.shape
        device = X_all_modalities.device

        # Create a placeholder for the final 4-channel encoder input
        encoder_input = torch.zeros((batch_size, 4, H, W), device=device)

        # Iterate per sample to allow different random selections per image in the batch
        for i in range(batch_size):

            k = torch.randint(low=1, high=3, size=(1,)).item() 

            available_indices = torch.randperm(4, device=device)[:k]

            indices_to_gather = list(available_indices)
            while len(indices_to_gather) < 4:
                random_choice = available_indices[torch.randint(0, k, (1,)).item()]
                indices_to_gather.append(random_choice)
            indices_tensor = torch.stack(indices_to_gather) # Shape (4,)
            shuffled_indices = indices_tensor[torch.randperm(4)]
            
            # E. Construct the input for this batch item
            encoder_input[i] = X_all_modalities[i, shuffled_indices, :, :]

        t_0 = torch.full((batch_size,), 0, device=device).long()
        
        # Now passing a full (B, 4, H, W) tensor
        feature_selected = self.model_1.Encoder(encoder_input, t_0)
        # feature_selected = 2*(feature_selected - feature_selected.min()) / (feature_selected.max() - feature_selected.min()) - 1
    
        # --- 5. Diffusion Process ---
        # Setup Noise
        if self.cfg.get('noisetype') is not None:
            noise = gen_noise(self.cfg, feature_selected.shape).to(self.device)
        else:
            noise = torch.randn_like(feature_selected) # Standard Gaussian noise

        t = torch.randint(0, self.num_timesteps, (batch_size,), device=device).long()

        # q_sample: Add noise to the latent representation
        # We use 'feature_selected' directly (Raw values)
        x = self.diffusion.q_sample(x_start=feature_selected, t=t, noise=noise)
        
        # Denoise using UNet
        # model_out, _ = self.unet(x, t, context=None)
        y_class = torch.zeros((batch_size,), device=device, dtype=torch.long)
        model_out = self.dit(x, t, y_class)
        
        # --- 6. Decoding (Reconstruction) ---
        # Note: If your Decoder ends in Sigmoid, your output is 0..1.
        # But your X_t1 target is -3..3.
        # Ensure your Decoder output has NO ACTIVATION (Linear) if targets are Z-Score.
        output_1 = self.model_1.Decoder(model_out, t_0)
        output_2 = self.model_2.Decoder(model_out, t_0)
        output_3 = self.model_3.Decoder(model_out, t_0)
        output_4 = self.model_4.Decoder(model_out, t_0)
        
        target_1 = X_t1c
        target_2 = X_t1n
        target_3 = X_t2
        target_4 = X_flair
        def normalize_for_lpips(img):
            # Maps approx -3..3 range to 0..1 safely
            _min = img.min()
            _max = img.max()
            # Avoid div by zero
            if _max == _min: return torch.zeros_like(img) 
            return 2 * (img - _min) / (_max - _min) - 1
        lpips_loss.to(device)
        p_loss_1 = lpips_loss(normalize_for_lpips(output_1), normalize_for_lpips(target_1))
        p_loss_2 = lpips_loss(normalize_for_lpips(output_2), normalize_for_lpips(target_2))
        p_loss_3 = lpips_loss(normalize_for_lpips(output_3), normalize_for_lpips(target_3))
        p_loss_4 = lpips_loss(normalize_for_lpips(output_4), normalize_for_lpips(target_4))
        
        total_lpips = (p_loss_1 + p_loss_2 + p_loss_3 + p_loss_4).mean()

        # --- 7. Loss Computation ---

        # Latent Loss: Compare 'model_out' (Prediction) vs 'feature_selected' (Truth)
        # FIX: Ensure we compare against the SAME feature we added noise to
        loss_latent = F.mse_loss(model_out, feature_selected, reduction='none')
        loss_latent = reduce(loss_latent, 'b ... -> b (...)', 'mean')
        loss_latent = loss_latent * extract(self.diffusion.p2_loss_weight, t, loss_latent.shape)

        # Reconstruction Loss (Weighted by Seg)
        # Helper to calculate weighted L1 loss
        def calc_recon_loss(pred, target, mask):
            loss_l1 = F.l1_loss(pred, target, reduction='none')
            # Weight lesion areas 5x (1 + 4)
            mask = (mask > 0).float() # Binary mask of lesion areas    
            loss = loss_l1 * (1 + mask * 4)
            return reduce(loss, 'b ... -> b (...)', 'mean')


        loss_model_1 = calc_recon_loss(output_1, target_1, seg_org)
        loss_model_2 = calc_recon_loss(output_2, target_2, seg_org)
        loss_model_3 = calc_recon_loss(output_3, target_3, seg_org)
        loss_model_4 = calc_recon_loss(output_4, target_4, seg_org)
        loss_recon = loss_model_1.mean() + loss_model_2.mean() + loss_model_3.mean() + loss_model_4.mean()

        # SSIM Loss (CRITICAL FIX 2: Normalize to 0-1 for SSIM calculation ONLY)
        def normalize_for_ssim(img):
            # Maps approx -3..3 range to 0..1 safely
            _min = img.min()
            _max = img.max()
            # Avoid div by zero
            if _max == _min: return torch.zeros_like(img) 
            return (img - _min) / (_max - _min)

        # We detach SSIM inputs so gradients don't get messy, or keep attached if you want to optimize for SSIM
        # Usually SSIM expects 0-1
        ssim_1 = 1 - ssim_loss(normalize_for_ssim(output_1), normalize_for_ssim(target_1))
        ssim_2 = 1 - ssim_loss(normalize_for_ssim(output_2), normalize_for_ssim(target_2))
        ssim_3 = 1 - ssim_loss(normalize_for_ssim(output_3), normalize_for_ssim(target_3))
        ssim_4 = 1 - ssim_loss(normalize_for_ssim(output_4), normalize_for_ssim(target_4))
        
        total_ssim = ssim_1 + ssim_2 + ssim_3 + ssim_4

        # Final Loss
        # Note: If loss_latent is 0.05 (small) and recon is 10 (huge), 
        # the model ignores the diffusion part. You might need to balance weights.
        loss = (loss_latent.mean() * 1.0) + \
           0.1 * (loss_recon + total_ssim + total_lpips)

        # Logging
        self.log(f'{self.prefix}train/Loss', loss, prog_bar=True, on_epoch=True, batch_size=batch_size, sync_dist=True)
        
        # Print mainly for debugging (remove later)
        # print(loss_latent.mean().item(), loss_model_1.mean().item())
        
        return {"loss": loss}



    def validation_step(self, batch, batch_idx: int):
            def normalize_for_lpips(img):
                # Maps approx -3..3 range to 0..1 safely
                _min = img.min()
                _max = img.max()
                # Avoid div by zero
                if _max == _min: return torch.zeros_like(img) 
                return 2 * (img - _min) / (_max - _min) - 1
            # --- 1. Unpack Batch ---
            if isinstance(batch, dict):
                images = batch['vol']['data']
            else:
                images = batch
            if images.ndim == 5:
                images = images.squeeze(2) 

            # --- 2. Resize & Separate ---
            images_resized = F.interpolate(images.squeeze(-1), size=(256, 256), mode='bilinear', align_corners=False)

            X_t1c   = images_resized[:, 0:1, :, :]
            X_t1n   = images_resized[:, 1:2, :, :]
            X_t2    = images_resized[:, 2:3, :, :]
            X_flair = images_resized[:, 3:4, :, :]
            seg_org = images_resized[:, 4:5, :, :] 
            
            # Ground Truths (Targets)
            targets = [X_t1c, X_t1n, X_t2, X_flair]
            modality_names = ['T1c', 'T1n', 'T2', 'Flair']

            (b_0, c, h, w) = X_t1c.shape
            device = X_t1c.device
            t_0 = torch.full((b_0,), 0, device=device).long()
            
            total_val_loss = 0
            target_1 = X_t1c
            target_2 = X_t1n
            target_3 = X_t2
            target_4 = X_flair

            for i in range(4):
                source_modality = targets[i] # The single modality we "have"
                
 
                encoder_input = source_modality.repeat(1, 4, 1, 1)

                # --- 4. Encoding ---
                # Pass the constructed 4-channel input to the shared encoder
                feature_selected = self.model_1.Encoder(encoder_input, t_0)
                feature_selected = 2*(feature_selected - feature_selected.min()) / (feature_selected.max() - feature_selected.min()) - 1

                # --- 5. Diffusion Process ---
                # Setup Noise
                if self.cfg.get('noisetype') is not None:
                    noise = gen_noise(self.cfg, feature_selected.shape).to(device)
                else:
                    noise = torch.randn_like(feature_selected)

                t = torch.randint(0, 1000, (b_0,), device=device).long()

                # Add noise
                x = self.diffusion.q_sample(x_start=feature_selected, t=t, noise=noise)
                
                # Denoise
                y_class = torch.zeros((b_0,), device=device, dtype=torch.long)
                model_out = self.dit(x, t, y_class)
                
                # --- 6. Decoding (Reconstruction) ---
                # Generate all 4 modalities from the latent representation
                output_1 = self.model_1.Decoder(model_out, t_0) # Pred T1c
                output_2 = self.model_2.Decoder(model_out, t_0) # Pred T1n
                output_3 = self.model_3.Decoder(model_out, t_0) # Pred T2
                output_4 = self.model_4.Decoder(model_out, t_0) # Pred Flair
                
                outputs = [output_1, output_2, output_3, output_4]

                # --- 7. Loss Computation ---
                # We calculate how well we reconstructed ALL targets from the CURRENT source
                
                # Helper: LPIPS
                curr_lpips = 0
                lpips_loss.to(device)
                for j in range(4):
                    curr_lpips += lpips_loss(normalize_for_lpips(outputs[j]), normalize_for_lpips(targets[j]))
                

                # Helper: Latent Loss
                loss_latent = F.mse_loss(model_out, feature_selected, reduction='none')
                loss_latent = reduce(loss_latent, 'b ... -> b (...)', 'mean')
                loss_latent = loss_latent * extract(self.diffusion.p2_loss_weight, t, loss_latent.shape)

                # Helper: Recon Loss (Weighted by Seg)
                def calc_recon_loss(pred, target, mask):
                    loss_l1 = F.l1_loss(pred, target, reduction='none')
                    # Weight lesion areas 5x (1 + 4)
                    mask = (mask > 0).float() # Binary mask of lesion areas    
                    loss = loss_l1 * (1 + mask * 4)
                    return reduce(loss, 'b ... -> b (...)', 'mean')

                loss_model_1 = calc_recon_loss(output_1, target_1, seg_org)
                loss_model_2 = calc_recon_loss(output_2, target_2, seg_org)
                loss_model_3 = calc_recon_loss(output_3, target_3, seg_org)
                loss_model_4 = calc_recon_loss(output_4, target_4, seg_org)
                loss_recon = loss_model_1.mean() + loss_model_2.mean() + loss_model_3.mean() + loss_model_4.mean()
                # Helper: SSIM Loss
                # Define normalize inside loop or globally
                def normalize_for_ssim(img):
                    _min, _max = img.min(), img.max()
                    if _max == _min: return torch.zeros_like(img)
                    return (img - _min) / (_max - _min)

                curr_ssim = 0
                for j in range(4):
                    curr_ssim += (1 - ssim_loss(normalize_for_ssim(outputs[j]), normalize_for_ssim(targets[j])))
                curr_ssim_loss = curr_ssim  # Sum of 4 SSIM losses

                # Combined Loss for this scenario
                scenario_loss = (loss_latent.mean() * 1.0) + \
                                (0.1 * loss_recon) + \
                                (0.1 * curr_ssim_loss) + \
                                (0.1 * curr_lpips.mean())
                
                total_val_loss += scenario_loss

            final_loss = total_val_loss / 4.0
            
            self.log(f'{self.prefix}val/Loss_comb', final_loss, prog_bar=False, on_step=False, on_epoch=True, batch_size=b_0, sync_dist=True)
            
            return {"loss": final_loss}

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.cfg.lr)
    
    def update_prefix(self, prefix):
        self.prefix = prefix 

    @property
    def loss_fn(self):
        if self.cfg.get('loss','l1') == 'l1':
            # print(1)
            return F.l1_loss
        elif self.cfg.get('loss','l1') == 'l2':
            # print(0)
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type')


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)

class Bridge_network(nn.Module):
    def __init__(self,n_channel):
        super().__init__()

        # take fully noisy image and convert to multichannel
        self.B_vartheta_ = nn.Sequential(
            ResnetBlock(2, n_channel//2),
            ResnetBlock(n_channel//2, n_channel)
        )
        #take multiple channel image and convert into original
        self.B_xi_ = nn.Sequential(
            ResnetBlock(n_channel, n_channel//2),
            ResnetBlock(n_channel//2, 1, groups=1)
        )

    def B_vartheta(self, x):
        # print(x.shape,"x_shape")
        return self.B_vartheta_(x)

    def B_xi(self, x):
        return self.B_xi_(x)


def normalize_to_neg_one_to_one(img):
    return img * 2 - 1
def normalize_to_zero_to_one(img):
    return (img-img.min()) / (img.max()-img.min())

        
        
def exists(x):
    return x is not None
    
def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)
    
def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def crop_and_resize(image, target_size=(256, 256), threshold=2,seg=None):
    image_array = np.array(image)
    original_shape = image_array.shape
    crop_coords = None
    if seg is not None:
        seg = np.array(seg)        
    if np.sum(image_array) <1:
        resized_image = Image.fromarray(image_array).resize(target_size, Image.BILINEAR)
        if seg is not None:
            seg = Image.fromarray(seg).resize(target_size, Image.BILINEAR)
        crop_coords = (0, 1, 0, 1) 
        return resized_image, seg, crop_coords, original_shape  
    non_zero_rows = np.any(image_array, axis=1)
    non_zero_cols = np.any(image_array, axis=0)  
    row_indices = np.where(non_zero_rows)[0]
    col_indices = np.where(non_zero_cols)[0]
    if len(row_indices) >= threshold and len(col_indices) >= threshold:
        min_r, max_r = min(row_indices), max(row_indices)
        min_c, max_c = min(col_indices), max(col_indices)
        crop_coords = (min_r, max_r, min_c, max_c)
        cropped_image = image_array[min(row_indices):max(row_indices), min(col_indices):max(col_indices)]
        if seg is not None:
            seg = seg[min(row_indices):max(row_indices), min(col_indices):max(col_indices)]
    elif row_indices is not None and col_indices is not None:
        top = 32-((max(row_indices) - min(row_indices)) // 2)
        left = 32-((max(col_indices) - min(col_indices)) // 2)
        cropped_image = image_array[min(row_indices)-top:max(row_indices)+top, min(col_indices)-left:max(col_indices)+left]
        crop_coords = (min(row_indices)-top,max(row_indices)+top, min(col_indices)-left,max(col_indices)+left)
        if seg is not None:
            seg = seg[min(row_indices)-top:max(row_indices)+top, min(col_indices)-left:max(col_indices)+left]
    
    else:
        cropped_image = image_array
        crop_coords = (0,239,0,239)
    
    cropped_image_256 = Image.fromarray(cropped_image).resize(target_size, Image.BILINEAR)
    if seg is not None:
        seg = Image.fromarray(seg).resize(target_size, Image.BILINEAR)
    
    return cropped_image_256,seg,crop_coords, original_shape


def restore_to_original(resized_tensor, crop_coords, original_shape):

    min_r, max_r, min_c, max_c = crop_coords
    H_orig, W_orig = original_shape
    h, w = max_r - min_r, max_c - min_c

    resized_back = F.interpolate(resized_tensor, size=(int(h),int(w)), mode='bilinear', align_corners=False)

    restored = torch.full((1, H_orig, W_orig), 0, dtype=resized_tensor.dtype, device=resized_tensor.device)
    restored[:,min_r:max_r, min_c:max_c] = resized_back
    return restored


def read_file_to_list(file_path):
    with open(file_path, 'r') as f:
        lines = f.read().splitlines()
    return lines    