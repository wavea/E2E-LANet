from SAM.mlla import MLLA
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import cv2
from SAM.image_encoder import ImageEncoderViT
from SAM.mask_decoder import MaskDecoder
from SAM.prompt_encoder import PromptEncoder
from SAM.transformer import TwoWayTransformer
from SAM.common import LayerNorm2d
from typing import List, Tuple, Type, Optional
from transforms import ResizeLongestSide
import os
from tqdm import tqdm
from functools import partial


class MLLAMed(nn.Module):
    def __init__(self, dim=192, img_size=224):
        super(MLLAMed, self).__init__()

        self.img_size = img_size
        self.pt = ResizeLongestSide(img_size)

        self.image_encoder = MLLA(embed_dim=dim, img_size=img_size)

        self.prompt_encoder = PromptEncoder(
            embed_dim=256,
            image_embedding_size=(img_size // 16, img_size // 16), # 1024 // 16
            input_image_size=(img_size, img_size),
            mask_in_chans=16,
            )
        
        self.mask_decoder = MaskDecoder(
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=256,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=256,
            )
        
        self.neck = nn.Sequential(
            nn.Conv2d(
                dim,
                256,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(256),
            nn.Conv2d(
                256,
                256,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(256),
        )


  
    def forward(self, x, mask=None, domain_seq=None, img_id=None):

        b = x.shape[0]

        image_embeddings = self.image_encoder(x)

        B, _, C = image_embeddings.size()
        image_embeddings = image_embeddings.view(B, self.img_size // 16, self.img_size // 16, C)
        image_embeddings = image_embeddings.permute(0, 3, 1, 2)
        image_embeddings = self.neck(image_embeddings)
        
        outputs_mask = []

        for idx in range(b): # for each batch 

            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
            )
    
            low_res_masks = self.mask_decoder(
                image_embeddings=image_embeddings[idx].unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )

            masks = F.interpolate(low_res_masks[0], (self.img_size, self.img_size), mode="bilinear", align_corners=False)

            outputs_mask.append(masks.squeeze(0))


        return torch.stack(outputs_mask, dim=0)
    
 


        
    

