import torch
import torch.nn as nn

class GatingMechanism(nn.Module):
    def __init__(self, args):
        super().__init__()
        # self.x_linear = Linear(, 1)
        # self.x_linear = Linear(batch_len, 1)

        self.fc = nn.Linear(512 * 2, 1)

        # self.fc_img_x = Linear(args.gating_dim, 128)

    def forward(self, x, y):
        # x = torch.mean(x, dim=0, keepdim=True)
        # region_x_features = torch.cat([region_img_features, x.repeat(region_img_features.size(0), 1, 1)], dim=-1)
        # region_linear_x = self.fc_img(region_x_features)
        #
        # region_sigmoid_x = torch.sigmoid(region_linear_x)  # max_len * batch * 1
        # region_img_features = torch.mul(region_sigmoid_x, region_img_features)
        # return region_img_features, region_sigmoid_x

        y = torch.mean(y, dim=0, keepdim=True)  ##
        t, b, c = x.shape
        y = y.expand(t, b, c)
        merge = torch.cat([x, y], dim=-1)

        gate = torch.sigmoid(self.fc(merge))  #
        #        gate = torch.tanh(self.fc_img(merge))
        #        gate = torch.relu(self.fc_img(merge))
        #        gate = torch.softplus(self.fc_img(merge))
        # gate = F.softmax(gate, dim=0)
        # out_features = torch.mul(gate, x)
        gated_text_feature = torch.mul(gate, x)
        gated_image_feature = torch.mul((1-gate), y)
        # return out_features, gate
        return gated_text_feature, gated_image_feature




import torch
import torch.nn as nn

class MMGatedFusion(nn.Module):
    def __init__(self, input_dim, gate_dim):
        super(MMGatedFusion, self).__init__()

        self.gate = nn.Sequential(
            nn.Linear(input_dim, gate_dim),
            nn.Tanh(),
            nn.Linear(gate_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, text_feature, image_feature):
        # text_feature shape: batch_size x text_feature_dim
        # image_feature shape: batch_size x image_feature_dim

        # concatenate text and image features
        combined_feature = torch.cat((text_feature, image_feature), dim=1) # shape: batch_size x (text_feature_dim + image_feature_dim)

        # calculate the gate value
        gate_value = self.gate(combined_feature) # shape: batch_size x 1

        # multiply text and image features with the gate
        gated_text_feature = gate_value * text_feature # shape: batch_size x text_feature_dim
        gated_image_feature = (1 - gate_value) * image_feature # shape: batch_size x image_feature_dim

        # concatenate gated text and image features
        fused_feature = torch.cat((gated_text_feature, gated_image_feature), dim=1) # shape: batch_size x (text_feature_dim + image_feature_dim)

        return fused_feature






import torch
import torch.nn as nn

class MMGatedFusion(nn.Module):
    def __init__(self, feature_dim, gate_dim, kernel_size=3, stride=1, padding=1):
        super(MMGatedFusion, self).__init__()

        self.conv_text = nn.Conv1d(in_channels=feature_dim, out_channels=gate_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv_image = nn.Conv1d(in_channels=feature_dim, out_channels=gate_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        self.gate = nn.Sequential(
            nn.Linear(gate_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, text_feature, image_feature):
        # text_feature shape: batch_size x length x feature_dim
        # image_feature shape: batch_size x length x feature_dim

        # permute the dimensions to match the input shape of 1D convolutional layers
        text_feature = text_feature.permute(0, 2, 1) # shape: batch_size x feature_dim x length
        image_feature = image_feature.permute(0, 2, 1) # shape: batch_size x feature_dim x length

        # apply 1D convolutional layers to text and image features
        conv_text_feature = self.conv_text(text_feature) # shape: batch_size x gate_dim x length
        conv_image_feature = self.conv_image(image_feature) # shape: batch_size x gate_dim x length

        # apply activation function to convolutional features
        conv_text_feature = torch.tanh(conv_text_feature)
        conv_image_feature = torch.tanh(conv_image_feature)

        # calculate the gate value
        gate_value = self.gate(conv_text_feature + conv_image_feature) # shape: batch_size x 1 x length

        # multiply text and image features with the gate
        gated_text_feature = gate_value * text_feature # shape: batch_size x feature_dim x length
        gated_image_feature = (1 - gate_value) * image_feature # shape: batch_size x feature_dim x length

        # concatenate gated text and image features
        fused_feature = torch.cat((gated_text_feature, gated_image_feature), dim=1) # shape: batch_size x (2 * feature_dim) x length

        return fused_feature



