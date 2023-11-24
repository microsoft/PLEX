import torch
import torch.nn as nn
# import torchvision

from robomimic.models.base_nets import ResNet18Conv

EPSILON = 1e-8


class R3M(nn.Module):
    def __init__(self, lr, l2weight=1.0, l1weight=1.0,
                 tcnweight=1.0, dist='l2'):
        super().__init__()

        assert dist in {'l2', 'cosine'}

        self.use_tb = False
        self.l2weight = l2weight
        self.l1weight = l1weight
        self.tcnweight = tcnweight ## Weight on TCN loss (states closer in same clip closer in embedding)
        self.dist = dist ## Use -l2 or cosine sim
        self.num_negatives = 3

        ## Distances and Metrics
        self.cs = nn.CosineSimilarity(1)
        self.bce = nn.BCELoss(reduce=False)
        self.sigm = nn.Sigmoid()

        # Determine PyTorch compute device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        ## Visual Encoder
        # if size == 18:
        #     self.outdim = 512
        #     self.convnet = torchvision.models.resnet18(pretrained=False)
        # elif size == 34:
        #     self.outdim = 512
        #     self.convnet = torchvision.models.resnet34(pretrained=False)
        # elif size == 50:
        #     self.outdim = 2048
        #     self.convnet = torchvision.models.resnet50(pretrained=False)
        # elif size == 0:
        #     from transformers import AutoConfig, AutoModel
        #     self.outdim = 768
        #     self.convnet = AutoModel.from_config(
        #         config=AutoConfig.from_pretrained('google/vit-base-patch32-224-in21k')
        #     )

        self.convnet = ResNet18Conv()
        self.convnet.fc = nn.Identity()
        self.convnet.train()
        self.convnet.to(self.device)

        ## Optimizer
        self.encoder_opt = torch.optim.Adam(self.convnet.parameters(), lr=lr)

    def sim(self, tensor1, tensor2):
        if self.dist == 'l2':
            d = -torch.linalg.norm(tensor1 - tensor2, dim=-1)
        else:
            d = self.cs(tensor1, tensor2)
        return d

    def update(self, images):
        metrics = dict()

        images = images.to(self.device)

        ## Encode Start and End Frames
        bs = images.shape[0]
        im_shape = images.shape[-3:]
        b_im_r = images.reshape(bs*5, *im_shape)
        alles = self.convnet(b_im_r)
        alle = alles.reshape(bs, 5, -1)
        e0 = alle[:, 0]
        eg = alle[:, 1]
        es0 = alle[:, 2]
        es1 = alle[:, 3]
        es2 = alle[:, 4]

        full_loss = 0

        ## LP Loss
        l2loss = torch.linalg.norm(alles, ord=2, dim=-1).mean()
        l1loss = torch.linalg.norm(alles, ord=1, dim=-1).mean()
        l0loss = torch.linalg.norm(alles, ord=0, dim=-1).mean()
        metrics['l2loss'] = l2loss.item()
        metrics['l1loss'] = l1loss.item()
        metrics['l0loss'] = l0loss.item()
        full_loss += self.l2weight * l2loss
        full_loss += self.l1weight * l1loss

        ## Within Video TCN Loss
        if self.tcnweight > 0:
            ## Number of negative video examples to use
            num_neg_v = self.num_negatives

            ## Computing distance from t0-t2, t1-t2, t1-t0
            sim_0_2 = self.sim(es2, es0)
            sim_1_2 = self.sim(es2, es1)
            sim_0_1 = self.sim(es1, es0)

            ## For the specified number of negatives from other videos
            ## Add it as a negative
            neg2 = []
            neg0 = []
            for _ in range(num_neg_v):
                es0_shuf = es0[torch.randperm(es0.size()[0])]
                es2_shuf = es2[torch.randperm(es2.size()[0])]
                neg0.append(self.sim(es0, es0_shuf))
                neg2.append(self.sim(es2, es2_shuf))
            neg0 = torch.stack(neg0, -1)
            neg2 = torch.stack(neg2, -1)

            ## TCN Loss
            smoothloss1 = -torch.log(EPSILON + (torch.exp(sim_1_2) / (EPSILON + torch.exp(sim_0_2) + torch.exp(sim_1_2) + torch.exp(neg2).sum(-1))))
            smoothloss2 = -torch.log(EPSILON + (torch.exp(sim_0_1) / (EPSILON + torch.exp(sim_0_1) + torch.exp(sim_0_2) + torch.exp(neg0).sum(-1))))
            smoothloss = ((smoothloss1 + smoothloss2) / 2.0).mean()
            a_state = ((1.0 * (sim_0_2 < sim_1_2)) * (1.0 * (sim_0_1 > sim_0_2))).mean()
            metrics['tcnloss'] = smoothloss.item()
            metrics['aligned'] = a_state.item()
            full_loss += self.tcnweight * smoothloss

        metrics['full_loss'] = full_loss.item()

        self.encoder_opt.zero_grad()
        full_loss.backward()
        self.encoder_opt.step()

        return metrics