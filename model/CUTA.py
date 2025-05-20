import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw, ImageChops
from mmseg.models.builder import BACKBONES
from timm.models.layers import trunc_normal_
from torch.nn.init import normal_
from torchvision import transforms

from model.adapter_modules import SpatialPriorModule, InteractionBlock, deform_inputs
from model.base.vit import TIMMVisionTransformer

_logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@BACKBONES.register_module()
class CUTA(TIMMVisionTransformer):
    def __init__(self, split, pretrain_size=224, num_heads=12, conv_inplane=64, n_points=4,
                 deform_num_heads=6, init_values=0., interaction_indexes=[[0, 2], [3, 5], [6, 8], [9, 11]],
                 with_cffn=True, cffn_ratio=0.25, deform_ratio=1.0, add_vit_feature=True,
                 pretrained='pretrained_ckpt/jx_vit_base_patch16_224_in21k-e5005f0a.pth',
                 use_extra_extractor=True, with_cp=False, *args, **kwargs):

        super().__init__(num_heads=num_heads, pretrained=pretrained,
                         with_cp=with_cp, *args, **kwargs)

        self.pretrain_size = (pretrain_size, pretrain_size)
        self.interaction_indexes = interaction_indexes
        self.add_vit_feature = add_vit_feature
        embed_dim = self.embed_dim

        self.level_embed = nn.Parameter(torch.zeros(3, embed_dim))
        self.spm = SpatialPriorModule(inplanes=conv_inplane, embed_dim=embed_dim, with_cp=False)
        self.interactions = nn.Sequential(*[
            InteractionBlock(dim=embed_dim, num_heads=deform_num_heads, n_points=n_points,
                             init_values=init_values, drop_path=self.drop_path_rate,
                             norm_layer=self.norm_layer, with_cffn=with_cffn,
                             cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                             extra_extractor=((True if i == len(interaction_indexes) - 1
                                               else False) and use_extra_extractor),
                             with_cp=with_cp)
            for i in range(len(interaction_indexes))
        ])
        self.up = nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2)
        self.norm4 = nn.BatchNorm2d(embed_dim)
        self.params = {
            'location': {
                'size': 19,
                'test_size': 14,
                'channels': [64, 128, 256, 728, 728, 728, 768],
                'mid_channel': 512,
                'test_channel': 768,
            },
            'cls_size': 10,
            'Fusion': {
                'hidden_dim': 384,
                'output_dim': 768,
            }
        }

        self.cls_header = nn.Sequential(
            nn.BatchNorm2d(self.params['Fusion']['output_dim']),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(self.params['Fusion']['output_dim'], 2),
        )

        self.up.apply(self._init_weights)
        self.spm.apply(self._init_weights)
        self.interactions.apply(self._init_weights)
        normal_(self.level_embed)

        self.split = split

        for i in range(12):
            for param in self.blocks[i].attn.parameters():
                param.requires_grad = False
            for param in self.blocks[i].mlp.parameters():
                param.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _get_pos_embed(self, pos_embed, H, W):
        pos_embed = pos_embed.reshape(
            1, self.pretrain_size[0] // 16, self.pretrain_size[1] // 16, -1).permute(0, 3, 1, 2)
        pos_embed = F.interpolate(pos_embed, size=(H, W), mode='bicubic', align_corners=False). \
            reshape(1, -1, H * W).permute(0, 2, 1)
        return pos_embed

    def _add_level_embed(self, c2, c3, c4):
        c2 = c2 + self.level_embed[0]
        c3 = c3 + self.level_embed[1]
        c4 = c4 + self.level_embed[2]
        return c2, c3, c4

    def forward(self, x):
        deform_inputs1, deform_inputs2 = deform_inputs(x)

        # Mask generator
        tensor_images_list = []
        generator = FrequencyDomainMasking(ratio=0.15, band='high', split=self.split)
        for i in range(0, x.shape[0]):
            new_image = x[i]
            mask_image = generator.transform(new_image.cpu(), i)
            transform = transforms.ToTensor()
            tensor_image = transform(mask_image).unsqueeze(0).to(device)
            tensor_images_list.append(tensor_image)
        final_tensor_images = torch.cat(tensor_images_list, dim=0)
        x_mask, H_mask, W_mask = self.patch_embed(final_tensor_images)

        c1, c2, c3, c4 = self.spm(x)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)

        x, H, W = self.patch_embed(x)
        bs, n, dim = x.shape
        pos_embed = self._get_pos_embed(self.pos_embed[:, 1:], H, W)
        x = self.pos_drop(x + pos_embed + x_mask)

        outs = list()
        for i, layer in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]
            x, c = layer(x, c, self.blocks[indexes[0]:indexes[-1] + 1],
                         deform_inputs1, deform_inputs2, H, W)
            outs.append(x.transpose(1, 2).view(bs, dim, H, W).contiguous())

        c2 = c[:, 0:c2.size(1), :]
        c3 = c[:, c2.size(1):c2.size(1) + c3.size(1), :]
        c4 = c[:, c2.size(1) + c3.size(1):, :]
        c4 = c4.transpose(1, 2).view(bs, dim, H // 2, W // 2).contiguous()

        if self.add_vit_feature:
            x1, x2, x3, x4 = outs
            x4 = F.interpolate(x4, scale_factor=0.5, mode='bilinear', align_corners=False)
            c4 = c4 + x4

        f4 = self.norm4(c4)
        cls_preds = self.cls_header(f4)

        return cls_preds


class FrequencyDomainMasking:
    def __init__(self, ratio: float = 0.15, band: str = 'all', split='train') -> None:
        self.ratio = ratio
        self.band = band  # 'low', 'mid', 'high', 'all'
        self.split = split

    def transform(self, image: Image.Image, i) -> Image.Image:
        image_array = np.array(image).astype(np.complex64)
        freq_image = np.fft.fftn(image_array, axes=(0, 1))

        _, height, width = image_array.shape

        if self.split == 'train':
            mask = self._create_balanced_mask(height, width).transpose(2, 0, 1)
            self.masked_freq_image = freq_image * mask
            masked_image_array = np.fft.ifftn(self.masked_freq_image, axes=(0, 1)).real.transpose(1, 2, 0)
            masked_image = Image.fromarray((masked_image_array * 255).astype(np.uint8))
            return masked_image
        else:
            masked_image_array = np.fft.ifftn(freq_image, axes=(0, 1)).real.transpose(1, 2, 0)
            freq_image = Image.fromarray(masked_image_array.astype(np.uint8))
            return freq_image

    def _create_balanced_mask(self, height, width):
        mask = np.ones((height, width, 3), dtype=np.complex64)

        if self.band == 'low':
            y_start, y_end = 0, height // 4
            x_start, x_end = 0, width // 4
        elif self.band == 'mid':
            y_start, y_end = height // 4, 3 * height // 4
            x_start, x_end = width // 4, 3 * width // 4
        elif self.band == 'high':
            y_start, y_end = 3 * height // 4, height
            x_start, x_end = 3 * width // 4, width
        elif self.band == 'all':
            y_start, y_end = 0, height
            x_start, x_end = 0, width
        else:
            raise ValueError(f"Invalid band: {self.band}")

        num_frequencies = int(np.ceil((y_end - y_start) * (x_end - x_start) * self.ratio))
        mask_frequencies_indices = np.random.permutation((y_end - y_start) * (x_end - x_start))[:num_frequencies]
        y_indices = mask_frequencies_indices // (x_end - x_start) + y_start
        x_indices = mask_frequencies_indices % (x_end - x_start) + x_start

        mask[y_indices, x_indices, :] = 0
        return mask
