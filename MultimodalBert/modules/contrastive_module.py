
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50


def sim_sorce(x, y ):
        x = torch.mean(x, dim=1)
        y = torch.mean(y, dim=1)
        x = F.normalize(x)
        y = F.normalize(y)
        similarity_matrix = torch.mm(x, y.t())
        similarity_score = similarity_matrix.diagonal()
        similarity_score = similarity_score.view(1, -1)
        similarity_score = torch.mean(similarity_score).item()


        return (similarity_score)

tau : float = 0.5
def sim(z1: torch.Tensor, z2: torch.Tensor):
        z1 = torch.mean(z1, dim=1)
        z2 = torch.mean(z2, dim=1)
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())


def semi_loss(z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / tau)
        refl_sim = f(sim(z1, z1))
        between_sim = f(sim(z1, z2))
        loss = -torch.log(between_sim.diag()/ (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))
        return loss


def get_loss(z1: torch.Tensor, z2: torch.Tensor):
        l1 = semi_loss(z1, z2)
        l2 = semi_loss(z2, z1)
        lc = (l1 + l2) * 0.5
        lc = lc.mean()

        return lc
#
#
#
#
class ContrastiveLossELI5(nn.Module):
        def __init__(self, args, temperature=0.5, verbose=True):
                super().__init__()
                self.batch_size = args.batch_size
                self.register_buffer("temperature", torch.tensor(temperature))
                self.verbose = verbose

        def forward(self, emb_i, emb_j):
                """
                emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
                z_i, z_j as per SimCLR paper
                """
                emb_i = torch.mean(emb_i, dim=1)
                emb_j = torch.mean(emb_j, dim=1)
                z_i = F.normalize(emb_i, dim=1)
                z_j = F.normalize(emb_j, dim=1)

                representations = torch.cat([z_i, z_j], dim=0)  ###2*b,len,dim
                similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0),
                                                        dim=2)
                # if self.verbose: print("Similarity matrix\n", similarity_matrix, "\n")

                def l_ij(i, j):
                        z_i_, z_j_ = representations[i], representations[j]
                        sim_i_j = similarity_matrix[i, j]
                        # if self.verbose: print(f"sim({i}, {j})={sim_i_j}")

                        numerator = torch.exp(sim_i_j / self.temperature)
                        one_for_not_i = torch.ones((2 * self.batch_size,)).to(emb_i.device).scatter_(0, torch.tensor([i],device='cuda'), 0.0)
                        # if self.verbose: print(f"1{{k!={i}}}", one_for_not_i)

                        denominator = torch.sum(
                                one_for_not_i * torch.exp(similarity_matrix[i, :] / self.temperature)
                        )
                        # if self.verbose: print("Denominator", denominator)

                        loss_ij = -torch.log(numerator / denominator)
                        # if self.verbose: print(f"loss({i},{j})={loss_ij}\n")

                        return loss_ij.squeeze(0)

                N = self.batch_size
                loss = 0.0
                for k in range(0, N):
                        loss += l_ij(k, k + N) + l_ij(k + N, k)
                return 1.0 / (2 * N) * loss


class ContrastiveLoss(nn.Module):
        def __init__(self, args, temperature=0.5):
                super().__init__()
                self.batch_size = args.batch_size
                self.register_buffer("temperature", torch.tensor(temperature))
                self.register_buffer("negatives_mask", (~torch.eye(args.batch_size * 2, args.batch_size * 2, dtype=bool,device="cuda")).float())

        def forward(self, emb_i, emb_j):
                """
                emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
                z_i, z_j as per SimCLR paper
                """
                emb_i = torch.mean(emb_i, dim=1)
                emb_j = torch.mean(emb_j, dim=1)
                z_i = F.normalize(emb_i, dim=1)
                z_j = F.normalize(emb_j, dim=1)

                representations = torch.cat([z_i, z_j], dim=0)####16*512
                similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0),
                                                        dim=-1)####16*16

                sim_ij = torch.diag(similarity_matrix, self.batch_size)
                sim_ji = torch.diag(similarity_matrix, -self.batch_size)
                positives = torch.cat([sim_ij, sim_ji], dim=0)    #16*1

                nominator = torch.exp(positives / self.temperature)
                denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)

                loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
                loss = torch.sum(loss_partial) / (2 * self.batch_size)
                return loss


# contrastive_loss = ContrastiveLoss(3, 1.0)
# # contrastive_loss(I, J).item() - ContrastiveLossELI5(I, J).item()
# # 0.0


############


# def NT_XentLoss(z1, z2, temperature=0.5):
#         z1 = torch.mean(z1, dim=1)
#         z2 = torch.mean(z2, dim=1)
#         z1 = F.normalize(z1, dim=1)
#         z2 = F.normalize(z2, dim=1)
#         N, Z = z1.shape
#         device = z1.device
#         representations = torch.cat([z1, z2], dim=0)   #2*B,dim
#         similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=-1)  ###2*B,#2*B
#         # similarity_matrix = torch.randn(16,16)
#         l_pos = torch.diag(similarity_matrix, N)  ##1*8
#         r_pos = torch.diag(similarity_matrix, -N)  ##1*8
#         positives = torch.cat([l_pos, r_pos]).view(2 * N, 1) ##16*1
#         diag = torch.eye(2 * N, dtype=torch.bool, device=device)
#         diag[N:, :N] = diag[:N, N:] = diag[:N, :N]
#
#         negatives = similarity_matrix[~diag].view(2 * N, -1)   #16*14
#
#         logits = torch.cat([positives, negatives], dim=1)   #16*15
#         logits /= temperature
#
#         labels = torch.zeros(2 * N, device=device, dtype=torch.int64)
#
#         loss = F.cross_entropy(logits, labels, reduction='sum')
#         return loss / (2 * N)
#
#
# class projection_MLP(nn.Module):
#         def __init__(self, args):
#                 super().__init__()
#                 hidden_dim = args.att_hidden_size
#                 self.layer1 = nn.Sequential(
#                         nn.Linear(args.att_hidden_size, args.att_hidden_size),
#                         nn.ReLU(inplace=True)
#                 )
#                 self.layer2 = nn.Linear(args.att_hidden_size, args.att_hidden_size)
#
#         def forward(self, x):
#                 x = self.layer1(x)
#                 x = self.layer2(x)
#
#                 return x
#
#
# class SimCLR(nn.Module):
#
#         def __init__(self, args):
#                 super().__init__()
#
#                 # self.backbone = backbone
#                 # self.projector = projection_MLP(args)
#                 # self.encoder = nn.Sequential(
#                         # self.backbone,
#                         # self.projector
#                 # )
#
#         def forward(self, x1, x2):
#                 # z1 = self.encoder(x1)
#                 # z2 = self.encoder(x2)
#
#                 loss = NT_XentLoss(x1, x2)
#                 # return {'loss': loss}
#                 return loss

##############

def D(p, z, version='simplified'):  # negative cosine similarity
        if version == 'original':
                z = z.detach()  # stop gradient
                z = torch.mean(z, dim=1)
                p = torch.mean(p, dim=1)
                p = F.normalize(p, dim=1)  # l2-normalize
                z = F.normalize(z, dim=1)  # l2-normalize
                return -(p * z).sum(dim=1).mean()

        elif version == 'simplified':  # same thing, much faster. Scroll down, speed test in __main__
                z = torch.mean(z, dim=1)
                p = torch.mean(p, dim=1)
                return - F.cosine_similarity(p, z.detach(), dim=-1).mean()
        else:
                raise Exception


# class projection_MLP(nn.Module):
#         def __init__(self, in_dim, hidden_dim=2048, out_dim=2048):
#                 super().__init__()
#                 ''' page 3 baseline setting
#                 Projection MLP. The projection MLP (in f) has BN ap-
#                 plied to each fully-connected (fc) layer, including its out-
#                 put fc. Its output fc has no ReLU. The hidden fc is 2048-d.
#                 This MLP has 3 layers.
#                 '''
#                 self.layer1 = nn.Sequential(
#                         nn.Linear(in_dim, hidden_dim),
#                         nn.BatchNorm1d(hidden_dim),
#                         nn.ReLU(inplace=True)
#                 )
#                 self.layer2 = nn.Sequential(
#                         nn.Linear(hidden_dim, hidden_dim),
#                         nn.BatchNorm1d(hidden_dim),
#                         nn.ReLU(inplace=True)
#                 )
#                 self.layer3 = nn.Sequential(
#                         nn.Linear(hidden_dim, out_dim),
#                         nn.BatchNorm1d(hidden_dim)
#                 )
#                 self.num_layers = 3
#
#         def set_layers(self, num_layers):
#                 self.num_layers = num_layers
#
#         def forward(self, x):
#                 if self.num_layers == 3:
#                         x = self.layer1(x)
#                         x = self.layer2(x)
#                         x = self.layer3(x)
#                 elif self.num_layers == 2:
#                         x = self.layer1(x)
#                         x = self.layer3(x)
#                 else:
#                         raise Exception
#                 return x
#
#
# class prediction_MLP(nn.Module):
#         def __init__(self, in_dim=2048, hidden_dim=512, out_dim=2048):  # bottleneck structure
#                 super().__init__()
#                 ''' page 3 baseline setting
#                 Prediction MLP. The prediction MLP (h) has BN applied
#                 to its hidden fc layers. Its output fc does not have BN
#                 (ablation in Sec. 4.4) or ReLU. This MLP has 2 layers.
#                 The dimension of h’s input and output (z and p) is d = 2048,
#                 and h’s hidden layer’s dimension is 512, making h a
#                 bottleneck structure (ablation in supplement).
#                 '''
#                 self.layer1 = nn.Sequential(
#                         nn.Linear(in_dim, hidden_dim),
#                         nn.BatchNorm1d(hidden_dim),
#                         nn.ReLU(inplace=True)
#                 )
#                 self.layer2 = nn.Linear(hidden_dim, out_dim)
#                 """
#                 Adding BN to the output of the prediction MLP h does not work
#                 well (Table 3d). We find that this is not about collapsing.
#                 The training is unstable and the loss oscillates.
#                 """
#
#         def forward(self, x):
#                 x = self.layer1(x)
#                 x = self.layer2(x)
#                 return x


class SimSiam(nn.Module):
        def __init__(self, backbone=resnet50()):
                super().__init__()

                # self.backbone = backbone
                # self.projector = projection_MLP(backbone.output_dim)
                #
                # self.encoder = nn.Sequential(  # f encoder
                #         self.backbone,
                #         self.projector
                # )
                # self.predictor = prediction_MLP()

        def forward(self, x1, x2):
                # f, h = self.encoder, self.predictor
                # z1, z2 = f(x1), f(x2)
                # p1, p2 = h(z1), h(z2)
                L = D(x1, x2) / 2 + D(x2, x1) / 2
                # return {'loss': L}
                return L

# if __name__ == "__main__":
#         model = SimSiam()
#         x1 = torch.randn((2, 3, 224, 224))
#         x2 = torch.randn_like(x1)
#
#         model.forward(x1, x2).backward()
#         print("forward backwork check")
#
#         z1 = torch.randn((200, 2560))
#         z2 = torch.randn_like(z1)
#         import time
#
#         tic = time.time()
#         print(D(z1, z2, version='original'))
#         toc = time.time()
#         print(toc - tic)
#         tic = time.time()
#         print(D(z1, z2, version='simplified'))
#         toc = time.time()
#         print(toc - tic)

