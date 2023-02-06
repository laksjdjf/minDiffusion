#参考：https://data-analytics.fun/2022/01/22/pytorch-vae/

import torch
import torch.nn as nn
import torch.nn.functional as F

#UNetモジュールの使いまわし
class Conv3(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, is_res: bool = False
    ) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(),
        )

        self.is_res = is_res

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.main(x)
        if self.is_res:
            x = x + self.conv(x)
            return x / 1.414
        else:
            return self.conv(x)

#UNetモジュールの使いまわし
class VAEDown(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(VAEDown, self).__init__()
        layers = [Conv3(in_channels, out_channels), nn.MaxPool2d(2)]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.model(x)

#UNetモジュールの使いまわし、conv層を減らしてskip connectionをなくしている
class VAEUp(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(VAEUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            Conv3(out_channels, out_channels),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x

#エンコーダ
class VAEEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, n_feat:int = 64, num_downs:int = 2) -> None:
        super(VAEEncoder, self).__init__()
        self.in_channels = in_channels

        self.n_feat = n_feat
        
        self.init_conv = Conv3(in_channels, n_feat, is_res=True)
        
        #チャンネル2倍、解像度半分をnum_downs回繰り返す。
        self.downs = nn.Sequential(*[
            VAEDown(n_feat * (2 ** i), n_feat * (2 ** i)*2)
            for i in range(num_downs)
        ])
        
        #平均と分散
        self.conv_mu = nn.Conv2d(n_feat * (2 ** num_downs), self.in_channels, kernel_size=3, padding=1)
        self.conv_var = nn.Conv2d(n_feat * (2 ** num_downs), self.in_channels, kernel_size=3, padding=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.init_conv(x)
        x = self.downs(x)
        mu = self.conv_mu(x)
        log_var = self.conv_var(x)
        eps = torch.randn_like(torch.exp(log_var))
        z = mu + torch.exp(log_var / 2) * eps
        return mu, log_var, z
    
#デコータ
class VAEDecoder(nn.Module):
    def __init__(self, in_channels: int = 3, n_feat:int = 64, num_downs:int = 2) -> None:
        super(VAEDecoder, self).__init__()
        self.in_channels = in_channels

        self.n_feat = n_feat
        
        self.init_conv = Conv3(self.in_channels, n_feat * (2 ** num_downs), is_res=True)
        
        #チャンネル半分、解像度2倍をnum_ups回繰り返す。
        self.ups = nn.Sequential(*[
            VAEUp(n_feat * (2 ** i) * 2, n_feat * (2 ** i))
            for i in reversed(range(num_downs))
        ])
        
        self.conv_out = nn.Conv2d(n_feat , self.in_channels, kernel_size=3, padding=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.init_conv(x)
        x = self.ups(x)
        out = self.conv_out(x)
        return torch.sigmoid(out)
    
class VAE(nn.Module):
    def __init__(self, in_channels:int = 3, n_feat:int = 64, num_down:int = 2):
        super(VAE, self).__init__()
        self.encoder = VAEEncoder(in_channels, n_feat, num_down)
        self.decoder = VAEDecoder(in_channels, n_feat, num_down)
    
    def forward(self, x):
        mu, log_var, z = self.encoder(x) # エンコード
        x = self.decoder(z) # デコード
        return x, mu, log_var, z
    
    def loss_function(self, label, predict, mu, log_var):
        #再構成誤差
        reconstruction_loss = F.binary_cross_entropy(predict, label, reduction='mean')
        #KLダイバージェンス
        kl_loss = - 0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        vae_loss = reconstruction_loss + kl_loss
        return vae_loss, reconstruction_loss, kl_loss

if __name__ == "__main__":
    x = torch.randn(2,3,16,16)
    vae = VAE()
    pred, mu, log_var, z = vae(x)
    
    print(vae.loss_function(x, pred, mu, log_var))
