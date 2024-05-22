import torch
import torch.nn as nn
import timm

from botnet import BoTStack


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class Swin_b_384_in22k(torch.nn.Module):
    def __init__(self, global_pool):
        
        super(Swin_b_384_in22k, self).__init__()
        swin_b = timm.create_model('swin_base_patch4_window12_384_in22k', pretrained=True, global_pool=global_pool)
        swin_b.head = Identity()

        self.feature_extraction = swin_b
        self.quality = self.quality_regression(1024+256, 128,1)

    def quality_regression(self,in_channels, middle_channels, out_channels):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, middle_channels),
            nn.Linear(middle_channels, out_channels),          
        )

        return regression_block

    def forward(self, x):
        
        x = self.feature_extraction(x)


        x = self.quality(x)

            
        return x



class RQ_VQA(torch.nn.Module):
    def __init__(self, pretrained_path):
        
        super(RQ_VQA, self).__init__()
        model = Swin_b_384_in22k(global_pool='')
        if pretrained_path!= None:
            model.load_state_dict(torch.load(pretrained_path))
        model.quality = Identity()
        swin_b = model

        self.feature_extraction = swin_b
        self.bot4 = BoTStack(dim=1024, dim_out=1024, num_layers=3, fmap_size=(12, 12), stride=1, rel_pos_emb=True)
        self.quality = self.quality_regression(1024+256+4096+495+768, 128,1)

    def quality_regression(self,in_channels, middle_channels, out_channels):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, middle_channels),
            nn.Linear(middle_channels, out_channels),          
        )

        return regression_block

    def forward(self, x, x_3D_features, x_LLM, x_LIQE, x_SlowFast):
        # input dimension: batch x frames x 3 x height x width
        x_size = x.shape

        # x_3D: batch x frames x 2048
        x_3D_features_size = x_3D_features.shape
        x_LLM_size = x_LLM.shape
        x_LIQE_size = x_LIQE.shape
        x_SlowFast_size = x_SlowFast.shape
        # x: batch * frames x 3 x height x width
        x = x.view(-1, x_size[2], x_size[3], x_size[4])
        # x_3D: batch * frames x 2048
        x_3D_features = x_3D_features.view(-1, x_3D_features_size[2])
        x_LLM = x_LLM.view(-1, x_LLM_size[2])
        x_LIQE = x_LIQE.view(-1, x_LIQE_size[2])
        x_SlowFast = x_SlowFast.view(-1, x_SlowFast_size[2])
        
        x = self.feature_extraction(x)
        x = x.permute(0, 2, 1)
        x = x.view(-1, 1024, 12, 12)
        x = self.bot4(x)
        x = x.view(-1, 1024, 144)
        x = x.permute(0, 1, 2)
        x = x.mean(dim=2)

        x = torch.cat((x, x_3D_features, x_LLM, x_LIQE, x_SlowFast), dim = 1)
        # print(x.shape)
        x = self.quality(x)
        # x: batch x frames
        x = x.view(x_size[0],x_size[1])
        # x: batch x 1
        x = torch.mean(x, dim = 1)
            
        return x

class RQ_VQA_base_model(torch.nn.Module):
    def __init__(self, pretrained_path):
        
        super(RQ_VQA_base_model, self).__init__()
        model = Swin_b_384_in22k(global_pool='avg')
        if pretrained_path!= None:
            model.load_state_dict(torch.load(pretrained_path))
        model.quality = Identity()
        swin_b = model

        self.feature_extraction = swin_b
        self.quality = self.quality_regression(1024+256, 128,1)

    def quality_regression(self,in_channels, middle_channels, out_channels):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, middle_channels),
            nn.Linear(middle_channels, out_channels),          
        )

        return regression_block

    def forward(self, x, x_3D_features):
        # input dimension: batch x frames x 3 x height x width
        x_size = x.shape

        # x_3D: batch x frames x 2048
        x_3D_features_size = x_3D_features.shape
        # x: batch * frames x 3 x height x width
        x = x.view(-1, x_size[2], x_size[3], x_size[4])
        # x_3D: batch * frames x 2048
        x_3D_features = x_3D_features.view(-1, x_3D_features_size[2])
        
        x = self.feature_extraction(x)

        x = torch.cat((x, x_3D_features), dim = 1)
        # print(x.shape)
        x = self.quality(x)
        # x: batch x frames
        x = x.view(x_size[0],x_size[1])
        # x: batch x 1
        x = torch.mean(x, dim = 1)
            
        return x

