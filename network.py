import os.path

import torch.nn as nn
from torchvision import models
import torch


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention block."""
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, max(channels // reduction, 4)),
            nn.ReLU(inplace=True),
            nn.Linear(max(channels // reduction, 4), channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c = x.size(0), x.size(1)
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class AlexNet(nn.Module):
    def __init__(self, hash_bit, pretrained=True):
        super(AlexNet, self).__init__()

        model_alexnet = models.alexnet(pretrained=pretrained)
        self.features = model_alexnet.features
        cl1 = nn.Linear(256 * 6 * 6, 4096)
        cl1.weight = model_alexnet.classifier[1].weight
        cl1.bias = model_alexnet.classifier[1].bias

        cl2 = nn.Linear(4096, 4096)
        cl2.weight = model_alexnet.classifier[4].weight
        cl2.bias = model_alexnet.classifier[4].bias

        self.hash_layer = nn.Sequential(
            nn.Dropout(),
            cl1,
            nn.ReLU(inplace=True),
            nn.Dropout(),
            cl2,
            nn.ReLU(inplace=True),
            nn.Linear(4096, hash_bit),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.hash_layer(x)
        return x


resnet_dict = {"ResNet18": models.resnet18, "ResNet34": models.resnet34, "ResNet50": models.resnet50,
               "ResNet101": models.resnet101, "ResNet152": models.resnet152}


class ResNet(nn.Module):
    """ResNet backbone with multi-scale SE-attention feature extraction (Innovation 1).

    Extracts features from three stages of ResNet34:
      - layer2: 128 channels  (28x28 for 224-input)
      - layer3: 256 channels  (14x14)
      - layer4: 512 channels  (7x7)
    Each scale is recalibrated by a Squeeze-and-Excitation block, then
    globally average-pooled and concatenated (128+256+512=896-d) before
    being projected to the hash code.
    """
    def __init__(self, hash_bit, res_model="ResNet34"):
        super(ResNet, self).__init__()
        if os.path.exists('./save/resnet34-b627a593.pth'):
            model_resnet = resnet_dict[res_model](pretrained=False)
            pre = torch.load('./models_ckpt/resnet34-b627a593.pth')
            model_resnet.load_state_dict(pre)
        else:
            model_resnet = resnet_dict[res_model](pretrained=True)

        self.conv1   = model_resnet.conv1
        self.bn1     = model_resnet.bn1
        self.relu    = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1  = model_resnet.layer1
        self.layer2  = model_resnet.layer2
        self.layer3  = model_resnet.layer3
        self.layer4  = model_resnet.layer4
        self.gap     = nn.AdaptiveAvgPool2d(1)

        # SE channel-attention at each scale (ResNet34 channels: 128 / 256 / 512)
        self.se2 = SEBlock(128)
        self.se3 = SEBlock(256)
        self.se4 = SEBlock(512)

        # Fusion projection: 128+256+512=896 → 512 → hash_bit
        self.hash_layer = nn.Sequential(
            nn.Linear(128 + 256 + 512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, hash_bit),
        )
        self.hash_layer[-1].weight.data.normal_(0, 0.01)
        self.hash_layer[-1].bias.data.fill_(0.0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)            # [B,  64, 56, 56]

        f2 = self.layer2(x)           # [B, 128, 28, 28]
        f3 = self.layer3(f2)          # [B, 256, 14, 14]
        f4 = self.layer4(f3)          # [B, 512,  7,  7]

        f2 = self.gap(self.se2(f2)).view(f2.size(0), -1)   # [B, 128]
        f3 = self.gap(self.se3(f3)).view(f3.size(0), -1)   # [B, 256]
        f4 = self.gap(self.se4(f4)).view(f4.size(0), -1)   # [B, 512]

        feat = torch.cat([f2, f3, f4], dim=1)              # [B, 896]
        return self.hash_layer(feat)


class NewNet(nn.Module):
    def __init__(self, hash_bit, pretrained=True):
        super(NewNet, self).__init__()
        self.m = 0.9
        self.encoder_q = ResNet(hash_bit)
        self.encoder_k = ResNet(hash_bit)
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    def forward(self, x):
        encode_x = self.encoder_q(x)
        with torch.no_grad():
            self._momentum_update_key_encoder()
            encode_x2 = self.encoder_k(x)
        return encode_x, encode_x2


class ClassifyNet(nn.Module):
    def __init__(self, n_class, res_model="ResNet34"):
        super(ClassifyNet, self).__init__()
        if os.path.exists('./save/resnet34-b627a593.pth'):
            model_resnet = resnet_dict[res_model](pretrained=False)
            pre = torch.load('./models_ckpt/resnet34-b627a593.pth')
            model_resnet.load_state_dict(pre)
        else:
            model_resnet = resnet_dict[res_model](pretrained=True)

        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool,
                                            self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)

        self.classify_layer = nn.Linear(model_resnet.fc.in_features, n_class)
        self.softmax = nn.Softmax(dim=1)
        self.classify_layer.weight.data.normal_(0, 0.01)
        self.classify_layer.bias.data.fill_(0.0)

    def forward(self, x):
        x = self.feature_layers(x)
        x = x.view(x.size(0), -1)
        x = self.classify_layer(x)
        y = self.softmax(x)
        return x, y


class LTHNet(nn.Module):
    def __init__(self, origin_model, feature_dim=2000, code_length=64, num_classes=100, num_prototypes=100):
        super(LTHNet, self).__init__()
        self.feature_dim = feature_dim
        self.code_length = code_length
        self.num_classes = num_classes
        self.num_prototypes = num_prototypes

        self.features = nn.Sequential(*list(origin_model.children())[:-1])
        self.fc = nn.Linear(512, feature_dim)

        self.fc_hallucinator = nn.Linear(feature_dim, num_prototypes)
        self.fc_selector = nn.Linear(feature_dim, feature_dim)
        self.attention = nn.Softmax(dim=1)

        self.hash_layer = nn.Linear(feature_dim, code_length)
        self.classifier = nn.Linear(code_length, num_classes)
        self.assignments = nn.Softmax(dim=1)

    def forward(self, x, dynamic_meta_embedding, prototypes):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = nn.ReLU()(x)
        x = self.fc(x)
        x = nn.ReLU()(x)

        direct_feature = x

        if dynamic_meta_embedding:
            if prototypes.size(0) != self.num_prototypes or prototypes.size(1) != self.feature_dim:
                print('prototypes error')
                return

            attention = self.fc_hallucinator(x)
            attention = self.attention(attention)
            memory_feature = torch.matmul(attention, prototypes)

            concept_selector = self.fc_selector(x)
            concept_selector = nn.Tanh()(concept_selector)

            x_meta = direct_feature + concept_selector * memory_feature
            x = self.hash_layer(x_meta)
            hash_codes = nn.Tanh()(x)
            assignments = self.classifier(hash_codes)
            assignments = self.assignments(assignments)
        else:
            x_meta = direct_feature
            x = self.hash_layer(x_meta)
            hash_codes = nn.Tanh()(x)
            assignments = self.classifier(hash_codes)
            assignments = self.assignments(assignments)

        return hash_codes, assignments, direct_feature


class orthohashNet(nn.Module):
    def __init__(self, hash_bit, nclass, res_model="ResNet34"):
        super(orthohashNet, self).__init__()
        if os.path.exists('./save/resnet34-b627a593.pth'):
            model_resnet = resnet_dict[res_model](pretrained=False)
            pre = torch.load('./models_ckpt/resnet34-b627a593.pth')
            model_resnet.load_state_dict(pre)
        else:
            model_resnet = resnet_dict[res_model](pretrained=True)

        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool,
                                            self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)

        self.hash_layer = nn.Linear(model_resnet.fc.in_features, hash_bit)
        self.hash_layer.weight.data.normal_(0, 0.01)
        self.hash_layer.bias.data.fill_(0.0)

        self.ce_fc = nn.Linear(hash_bit, nclass)

    def forward(self, x):
        x = self.feature_layers(x)
        x = x.view(x.size(0), -1)
        x = self.hash_layer(x)
        y = self.ce_fc(x)
        return y, x

    def get_hash_params(self):
        return list(self.ce_fc.parameters()) + list(self.hash_layer.parameters())

    def get_backbone_params(self):
        return list(self.feature_layers.parameters())
