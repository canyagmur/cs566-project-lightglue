import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

import gluefactory.multipoint.utils as utils
from .cbam import * #attention!
from .SimpleViT import SimpleViT
from .Swinv2 import SwinTransformerV2
from .Swinv2pretrained import SwinTransformerV2Pretrained
from .ScuNet import SCUNet
class XPoint(nn.Module):
    default_config = {
        'multispectral': True,
        'descriptor_head': True,
        "homography_regression_head":False, #added by me
        'intepolation_mode': 'bilinear',
        'descriptor_size': 256,
        'normalize_descriptors': True,
        'final_batchnorm': True,
        'reflection_pad': True,
        'bn_first': False,
        'double_convolution': True,
        'channel_version': 0,
        'verbose': False,
        'mixed_precision': False,
        'force_return_logits': False,
        'use_attention' : 
            {
            'check' : False,
            'type' : 'SimpleViT',
            'height' : 256,
            'width' : 256,
            'pretrained' :
                 {
                    'check' : True,
                    'type_dir' : "model_weights/swinv2-imagenet/base256"
                }
            },
    }



    def __init__(self, config = None):
        super(XPoint, self).__init__()

        if config:
            self.config = utils.dict_update(copy.deepcopy(self.default_config), config)
        else:
            self.config = self.default_config

        if self.config['reflection_pad']:
            self.pad_method = nn.ReflectionPad2d
        else:
            self.pad_method = nn.ZeroPad2d

        if self.config['channel_version'] == 0:
            self.n_channels = [1, 64, 64, 128, 128]
            self.head_channels = 256 #256#

        elif self.config['channel_version'] == 1:
            self.n_channels = [1, 32, 64, 96, 128]
            self.head_channels = self.config['descriptor_size']

        elif self.config['channel_version'] == 2:
            self.n_channels = [1, 8, 16, 32, 64]
            self.head_channels = self.config['descriptor_size']

        else:
            print('Unknown channel_version: ', self.config['channel_version'])
            self.n_channels = [1, 64, 64, 128, 128]
            self.head_channels = 256

        if self.config['multispectral']:
            self.encoder_thermal = self.generate_encoder()
            self.encoder_optical = self.generate_encoder()
        else:
            self.encoder = self.generate_encoder()

        # detector head
        self.detector_head_convolutions = [
            self.pad_method(1),
            nn.Conv2d(self.n_channels[4], self.head_channels, 3),
            *self.getNonlinearity(self.head_channels),
            nn.Conv2d(self.head_channels, 65, 1),
        ]

        if self.config['final_batchnorm']:
            self.detector_head_convolutions.append(nn.BatchNorm2d(65))

        self.detector_head_convolutions = nn.Sequential(*self.detector_head_convolutions)

        self.softmax = nn.Softmax2d()
        self.shuffle = nn.PixelShuffle(8)

        if self.config['descriptor_head']:
            self.descriptor_head_convolutions = [
                self.pad_method(1),
                nn.Conv2d(self.n_channels[4], self.head_channels, 3),
                *self.getNonlinearity(self.head_channels),
                nn.Conv2d(self.head_channels, self.config['descriptor_size'], 1),
            ]

            if self.config['final_batchnorm']:
                self.descriptor_head_convolutions.append(nn.BatchNorm2d(self.config['descriptor_size']))

            self.descriptor_head_convolutions = nn.Sequential(*self.descriptor_head_convolutions)
        
        if self.config['homography_regression_head']:
            self.homography_regression_head = [
                self.pad_method(1),
                nn.Conv2d(self.n_channels[4], self.head_channels, 3),
                *self.getNonlinearity(self.head_channels),
                nn.Conv2d(self.head_channels, self.config['descriptor_size'], 1),
            ]

        if self.config['verbose']:
            print('MultiPoint number of trainable parameter: ' + str(sum([p.numel() for p in self.parameters()])))

    def set_force_return_logits(self, value):
        if not isinstance(value, bool):
            raise ValueError('set_force_return_logits: The input value needs to be a bool')

        self.config['force_return_logits']= value

    def forward(self, data):
        #print(data["image"].shape)
        if self.config['mixed_precision']:
            with torch.cuda.amp.autocast():
                return self.forward_impl(data)
        else:
            return self.forward_impl(data)

    def disp_heatmap(self,original,x,original2,x2):
            ##try to vis
            import ntpath
            import os
            import pathlib
            import warnings

            import GPUtil
            import cv2
            import matplotlib.image as mpimg
            import matplotlib.pyplot as plt
            import numpy as np
            import torch
            import torch.nn as nn

            #swin_output = swin_output.permute(0, 2, 3, 1)
            print(x.shape)
            #x = x.permute(0,2,3,1).squeeze(0).detach().cpu().numpy()
            x = x.permute(0,2,1).squeeze(0).detach().cpu().numpy()
            print(x.shape)
            
            
            emb1_attn = np.mean(x.reshape(32*32, 64), axis=1)
            indices = emb1_attn.argsort()[:int(-0.9 * emb1_attn.shape[0])]
            #print( emb1_attn.argsort())
            emb1_attn[indices] = 0
            emb1_attn = emb1_attn.reshape(32, 32)
            emb1_attn = 255 * (emb1_attn - emb1_attn.min()) / (emb1_attn.max() - emb1_attn.min())
            emb1_attn = np.uint8(emb1_attn)
            emb1_attn = cv2.resize(emb1_attn, (256, 256), interpolation=cv2.INTER_CUBIC)

            x2 = x2.permute(0,2,1).squeeze(0).detach().cpu().numpy()
            emb2_attn = np.mean(x2.reshape(32*32,64), axis=1)
            indices = emb2_attn.argsort()[:int(-0.9 * emb2_attn.shape[0])]
            emb2_attn[indices] = 0
            emb2_attn = emb2_attn.reshape(32, 32)
            emb2_attn = 255 * (emb2_attn - emb2_attn.min()) / (emb2_attn.max() - emb2_attn.min())
            emb2_attn = np.uint8(emb2_attn)
            emb2_attn = cv2.resize(emb2_attn, (256, 256), interpolation=cv2.INTER_CUBIC)

            def display(rgb, attn1, nir_orig, attn2):
                fig, ax = plt.subplots(nrows=2, ncols=2)
                ax[0, 0].imshow(rgb,cmap="gray")
                ax[0, 0].axis('off')
                ax[0, 0].set_title('Input Image')

                ax[0, 1].imshow(rgb)
                ax[0, 1].imshow(attn1, alpha=0.25, cmap='jet')
                ax[0, 1].axis('off')
                ax[0, 1].set_title('Attention')

                ax[1, 0].imshow(nir_orig, cmap="gray")
                ax[1, 0].axis('off')
                ax[1, 0].set_title('Input Image')

                ax[1, 1].imshow(nir_orig, cmap="gray")
                ax[1, 1].imshow(attn2, alpha=0.25, cmap='jet')
                ax[1, 1].axis('off')
                ax[1, 1].set_title('Attention')

                plt.show()
            orig = original[0].permute(1,2,0).detach().cpu().numpy()
            orig2 = original2[0].permute(1,2,0).detach().cpu().numpy()
            display(orig/255, emb1_attn,orig2/255, emb2_attn)
            plt.close()

    def forward_impl(self, data):
        if self.config['multispectral']:
            # create a tensor with the output shape of the encoder
            shape = data['image'].shape
            
            tensor_dtype = torch.float
            if self.config['mixed_precision']:
                tensor_dtype = torch.half

            x = torch.zeros((shape[0], self.n_channels[4], int(shape[2]/8), int(shape[3]/8)), dtype = tensor_dtype).to(data['image'].device)

            # check if there is at least one optical image in the batch
            if len(data['is_optical'].shape) ==1: #added by me
                data['is_optical']=data['is_optical'].unsqueeze(dim=0) #added by me
            
            num_optical = data['is_optical'][:,0].sum()
            num_thermal = shape[0] - num_optical
            if num_optical > 0.0:
                x[data['is_optical'][:,0],:] = self.encoder_optical(data['image'][data['is_optical'][:,0]])

            if num_thermal > 0.0:
                x[~data['is_optical'][:,0],:] = self.encoder_thermal(data['image'][~data['is_optical'][:,0],:])
           
            encoder_output = x.clone().detach() # NEW added
        
        else:
            x = self.encoder(data['image'])
            encoder_output = x.clone().detach()


        prob, logits = self.detector_head(x)
        out = {'prob': prob,
               'logits': logits,
        }

        if self.config['descriptor_head']:
            desc = self.descriptor_head(x)
            out['desc'] = desc
        
        print(x.shape)
        if self.config["homography_regression_head"]:
            homography_parameters = self.homography_regression_head(x)
            out["homography_parameters"] = homography_parameters
        
        out['encoder_output']  = encoder_output #added by me for cos sim

        #print("desc shape : ",desc.shape)
        # cv2.imshow("prob image",prob[0][0].cpu().numpy()*255)
        # cv2.imshow("desc image",desc[0][0].cpu().numpy()*255)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return out

    def getNonlinearity(self, N):
        if self.config['bn_first']:
            return nn.BatchNorm2d(N), nn.ReLU(True)
        else:
            return nn.ReLU(True), nn.BatchNorm2d(N)

    def getConvolutionBlock(self, N_in, N_out):
        if self.config['double_convolution']:
            return (self.pad_method(1), nn.Conv2d(N_in, N_out, 3), *self.getNonlinearity(N_out),
                    self.pad_method(1), nn.Conv2d(N_out, N_out, 3), *self.getNonlinearity(N_out))
        else:
            return (self.pad_method(1), nn.Conv2d(N_in, N_out, 3), *self.getNonlinearity(N_out))

    def detector_head(self, x):
        logits = self.detector_head_convolutions(x).to(torch.float)
        #print("logits : ",logits.shape)
        if self.training or self.config['force_return_logits']:
            #print(logits.shape)
            #logits = F.interpolate(logits,size=(512,512),mode="bicubic",align_corners=True)
            return None, logits
        else:
            prob = self.softmax(logits)
            prob = self.shuffle(prob[:,:-1])
            #prob = F.interpolate(prob, size=(512, 640), mode='nearest')
            #prob = F.interpolate(prob,size=(512,512),mode="bicubic",align_corners=True)
            return prob, None

    def descriptor_head(self, x):
        x = self.descriptor_head_convolutions(x).to(torch.float)

        if self.config['normalize_descriptors']:
            x = torch.nn.functional.normalize(x, p=2, dim=1)

        #print("desc",x.shape)
        #x = F.interpolate(x,size=(64,80),mode="bicubic",align_corners=True)
        #print("desc after",x.shape)
        return x
    def homography_regression_head(self, x):
        x = self.descriptor_head_convolutions(x).to(torch.float)

        return x

    

    #self.n_channels (0) = [1, 64, 64, 128, 128]
    #self.head_channels = 256
    def generate_encoder(self):
        
        if self.config["use_attention"]["check"]:
            if  self.config["use_attention"]["type"] =="conv":
                encoder = nn.Sequential(
                    *self.getConvolutionBlock(self.n_channels[0], self.n_channels[1]),
                    #CBAM(self.n_channels[1],16),

                    nn.MaxPool2d(2,2),

                    *self.getConvolutionBlock(self.n_channels[1], self.n_channels[2]),
                    #CBAM(self.n_channels[2],16),

                    nn.MaxPool2d(2,2),

                    *self.getConvolutionBlock(self.n_channels[2], self.n_channels[3]),
                    #CBAM(self.n_channels[3],16),

                    nn.MaxPool2d(2,2),

                    *self.getConvolutionBlock(self.n_channels[3], self.n_channels[4]),
                    CBAM(self.n_channels[4],16)
                )
            elif self.config["use_attention"]["type"] =="SimpleViT":
                image_size = self.config["use_attention"]["height"],self.config["use_attention"]["width"]
                dim = int(image_size[0]/8 * image_size[1]/8)
                encoder = SimpleViT(image_size=image_size, patch_size=16, dim=dim, depth=12, heads=8,channels=1, mlp_dim=2048, num_classes=1000)
            elif self.config["use_attention"]["type"] =="Swinv2":
                if self.config['use_attention']['pretrained']['check']:
                    swin_config_object = utils.get_yaml_object(self.config['use_attention']['pretrained']['type_dir'] )

                    encoder = SwinTransformerV2Pretrained(img_size=swin_config_object.DATA.IMG_SIZE,
                            #patch_size=swin_config_object.MODEL.SWINV2.PATCH_SIZE,
                            #in_chans=swin_config_object.MODEL.SWINV2.IN_CHANS,
                            #num_classes=swin_config_object.MODEL.NUM_CLASSES,
                            embed_dim=swin_config_object.MODEL.SWINV2.EMBED_DIM,
                            depths=swin_config_object.MODEL.SWINV2.DEPTHS,
                            num_heads=swin_config_object.MODEL.SWINV2.NUM_HEADS,
                            window_size=swin_config_object.MODEL.SWINV2.WINDOW_SIZE,
                            #mlp_ratio=swin_config_object.MODEL.SWINV2.MLP_RATIO,
                            #qkv_bias=swin_config_object.MODEL.SWINV2.QKV_BIAS,
                            #drop_rate=swin_config_object.MODEL.DROP_RATE,
                            drop_path_rate=swin_config_object.MODEL.DROP_PATH_RATE,
                            #ape=swin_config_object.MODEL.SWINV2.APE,
                            #patch_norm=swin_config_object.MODEL.SWINV2.PATCH_NORM,
                            #use_checkpoint=swin_config_object.TRAIN.USE_CHECKPOINT,
                            pretrained_window_sizes=swin_config_object.MODEL.SWINV2.PRETRAINED_WINDOW_SIZES,
                            original_image_size=self.config["use_attention"]["height"])
                    #print(swin_config_object.MODEL.SWINV2.PRETRAINED_WINDOW_SIZES)
                    self.n_channels[4] = int(swin_config_object.MODEL.SWINV2.EMBED_DIM // 2) #not sure if this general case ?  


                else :
                    image_size = self.config["use_attention"]["height"],self.config["use_attention"]["width"]
                    encoder = SwinTransformerV2(
                                img_size=image_size, patch_size=4, in_chans=1, 
                                embed_dim=96, depths=[4,8], num_heads=[3, 6, 12, 24], #depths=[2, 2, 6, 2]
                                window_size=8, mlp_ratio=4., qkv_bias=True,
                                drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                                norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                                use_checkpoint=False, pretrained_window_sizes=[0, 0, 0, 0]
                            )
            elif self.config["use_attention"]["type"] =="ScuNet":
                encoder = SCUNet(in_nc=1,config=[4,4,4,4,4,4,4],dim=64)
                self.n_channels[4] = 512

        else:
                encoder = nn.Sequential(
                *self.getConvolutionBlock(self.n_channels[0], self.n_channels[1]),

                nn.MaxPool2d(2,2),

                *self.getConvolutionBlock(self.n_channels[1], self.n_channels[2]),

                nn.MaxPool2d(2,2),

                *self.getConvolutionBlock(self.n_channels[2], self.n_channels[3]),

                nn.MaxPool2d(2,2),

                *self.getConvolutionBlock(self.n_channels[3], self.n_channels[4]),
            )
            

        return encoder
