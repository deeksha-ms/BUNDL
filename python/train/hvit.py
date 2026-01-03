import torch
import torch.nn as nn
import torch.nn.functional as F



from cvnets.layers.linear_attention import LinearSelfAttention

    
class build_transformer_block(nn.Module):
    def __init__(self, embed_dim, heads=8, dropout=0.1):
        super(build_transformer_block, self).__init__()
        self.bn1  = nn.BatchNorm2d(embed_dim)
        self.smha = LinearSelfAttention(
            opts=None,
            embed_dim=embed_dim,
            attn_dropout=dropout
        )
        self.bn2 = nn.BatchNorm2d(embed_dim)
        self.bffn  = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(embed_dim)
        )

    def forward(self, x):
        residual = x
        x = self.bn1(x)
        x = self.smha(x)
        x = x + residual

        residual = x
        x = self.bn2(x)
        x = self.bffn(x)
        x = x + residual
        return x

class HVIT(nn.Module):
    def __init__(self, time_steps, channels, dropout=0.1, device='cpu'):
        super(HVIT, self).__init__()
        self.T = time_steps
        self.N = channels
        self.cnn_small = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=3, padding=(1,1), stride=(1,1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(10,1)),
            nn.LayerNorm([6, self.T//10, self.N])
        )
        self.cnn_large = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=17, padding=(8,8), stride=(10,1)),
            nn.LayerNorm([10, self.T//10, self.N]),
            nn.ReLU()
        )
        self.c2t1_cnn = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=3, padding=(1,1), stride=(2,2)), 
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=1, padding=(0,0), stride=(1,1)),
            nn.BatchNorm2d(32)
        )

        self.c2t2_cnn = nn.Sequential(
            nn.Conv2d(32, 128, kernel_size=3, padding=(1,1), stride=(1,1)), 
            nn.ReLU(),
            nn.Conv2d(128, 32, kernel_size=1, padding=(0,0), stride=(1,1)),
            nn.BatchNorm2d(32)
        )

        self.c2t1_transformer = build_transformer_block(32, heads=8, dropout=dropout)
        self.c2t2_transformer = build_transformer_block(32, heads=8, dropout=dropout)

        self.bottleneck_mu = nn.Sequential(            
            nn.Conv2d(32, 128, kernel_size=3, padding=(1,1), stride=(2,2)), 
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=1, padding=(0,0), stride=(1,1)),
            nn.BatchNorm2d(64), 
            nn.Conv2d(64, 64, kernel_size=(5, 3), padding=(2,1), stride=(1,1)),
            nn.BatchNorm2d(64),
            nn.ReLU(), 
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
        )
        self.bottleneck_logvar = nn.Sequential(
            nn.Conv2d(32, 128, kernel_size=3, padding=(1,1), stride=(2,2)), 
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=1, padding=(0,0), stride=(1,1)),
            nn.BatchNorm2d(64), 
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
        )
        self.classifier = nn.Linear(64, 2)  # Assuming binary classification

        self.hyperparam = nn.Parameter(torch.tensor(-5.00, requires_grad=True))
        self.bce = nn.CrossEntropyLoss(weight = torch.Tensor([0.20, 0.80]).double().to(device))

    def patch_embed(self, x):
        x1 = self.cnn_small(x)
        x2 = self.cnn_large(x) 
        #concatenate along the feature dimension
        x = torch.cat((x1, x2), dim=1)
        return x


    def forward(self, x, train=False):
        #x = x.view(???)
        h = self.patch_embed(x)
        h = self.c2t1_cnn(h)
        h = self.c2t1_transformer(h)


        h = self.c2t2_cnn(h)
        h = self.c2t2_transformer(h)


        mu = self.bottleneck_mu(h)
        logvar = self.bottleneck_logvar(h)
        if train:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu
        out = self.classifier(z)
        return out, mu, logvar

    def compute_kl_loss(self, mu, logvar):
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return kl_loss
    

    def compute_loss(self, logits, mu, logvar, targets, weighted =False):
  
        if weighted:
            ce_loss = self.bce(logits, targets)
        else:
            ce_loss = F.cross_entropy(logits, targets)
        
        kl_loss = self.compute_kl_loss(mu, logvar)
        weight = 1 / (1 + torch.exp(self.hyperparam))
        #weight = 0.5
        total_loss = ce_loss * weight +  kl_loss * (1 - weight)
        
        return total_loss, ce_loss, kl_loss
    
if __name__ == "__main__":
    model = HVIT(200, 18)
    model.double()
    x = torch.randn(5, 1, 200, 18).double()  # Example input tensor
    output, mu , logvar = model(x)
    loss, ce, kl = model.compute_loss(output, mu, logvar, torch.randint(0,2,(5,)).long())
    print(output.shape)
    print("Loss:", loss.item(), "CE:", ce.item(), "KL:", kl.item())
    

