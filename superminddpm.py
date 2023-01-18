"""
Extremely Minimalistic Implementation of DDPM

https://arxiv.org/abs/2006.11239

Everything is self contained. (Except for pytorch and torchvision... of course)

run it with `python superminddpm.py`
"""

from typing import Dict, Tuple
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision.datasets import MNIST
from torchvision import transforms
from torchvision.utils import save_image, make_grid

#stepごとのノイズの平均分散を決める
def ddpm_schedules(beta1: float, beta2: float, T: int) -> Dict[str, torch.Tensor]:
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"
    
    # β1 ⇒ β2の線形スケジュール
    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    
    # √βt
    sqrt_beta_t = torch.sqrt(beta_t)
    
    # αt t-1⇒tのノイズ計算で使う
    alpha_t = 1 - beta_t
    
    # log(αt)　総乗を総和に置き換えるため対数の性質を利用する。
    log_alpha_t = torch.log(alpha_t)
    
    #  αt' = Παt = exp(logΠαt) = exp(Σlog(αt)) バーが再現できないのでアポストロフィで代用。
    # 0⇒tのノイズ計算で使う
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    # √αt'
    sqrtab = torch.sqrt(alphabar_t)
    
    # 1/√αt'
    oneover_sqrta = 1 / torch.sqrt(alpha_t)
    
    # √(1-αt') #ノイズの標準偏差（√しなければ分散）
    sqrtmab = torch.sqrt(1 - alphabar_t)
    
    # (1-αt)/√(1-αt') 逆拡散過程の平均に使う
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }

#icチャンネル⇒ocチャンネルへのconv層。何の略なんだろう？
blk = lambda ic, oc: nn.Sequential(
    nn.Conv2d(ic, oc, 7, padding=3),
    nn.BatchNorm2d(oc),
    nn.LeakyReLU(),
)

#blkを並べただけのモデル。UNet-likeと書いてあるがskip-connectionはない
#attentionもないし、条件付け（promptとか）をすることもできない。
#そもそもtime embもないんだが
class DummyEpsModel(nn.Module):
    """
    This should be unet-like, but let's don't think about the model too much :P
    Basically, any universal R^n -> R^n model should work.
    """

    def __init__(self, n_channel: int) -> None:
        super(DummyEpsModel, self).__init__()
        self.conv = nn.Sequential(  # with batchnorm
            blk(n_channel, 64),
            blk(64, 128),
            blk(128, 256),
            blk(256, 512),
            blk(512, 256),
            blk(256, 128),
            blk(128, 64),
            nn.Conv2d(64, n_channel, 3, padding=1),
        )
    
    #tこのコードでは使われていないが、実際の拡散モデルではモデルに何step目かを教えてあげる仕組みが必要
    def forward(self, x, t) -> torch.Tensor:
        # Lets think about using t later. In the paper, they used Tr-like positional embeddings.
        return self.conv(x)


class DDPM(nn.Module):
    def __init__(
        self,
        eps_model: nn.Module, #予測用のネットワーク
        betas: Tuple[float, float], #ノイズ量
        n_T: int, #サンプリングの層ステップ数
        criterion: nn.Module = nn.MSELoss(), #損失関数、diffusion modelの場合二乗誤差になる
    ) -> None:
        super(DDPM, self).__init__()
        self.eps_model = eps_model

        # register_buffer allows us to freely access these tensors by name. It helps device placement.
        # register_bufferはParameterにしたくないけど定数として取っておきたいときにつかうらしい。torch.nn.Moduleのメソッド。self.<key>で直接呼べるらしい。
        # スケジューラーがくれる定数を入れている。
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.criterion = criterion
    
    #ランダムなtimestepを選んで、モデルの損失を計算する。トレーニングにのみ使うメソッド。
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Makes forward diffusion x_t, and tries to guess epsilon value from x_t using eps_model.
        This implements Algorithm 1 in the paper.
        """
        
        # 0からn_Tまでの一様分布からtime stepをサンプリング
        _ts = torch.randint(1, self.n_T, (x.shape[0],)).to(
            x.device
        )  # t ~ Uniform(0, n_T)
        
        #　ノイズを平均0、分散1の正規分布からサンプリング
        eps = torch.randn_like(x)  # eps ~ N(0, 1)
        
        # 元の画像から_ts step目までの順方向過程を一気に計算して_ts時点でのノイズ付き画像x_tを計算する
        # x_t = √αt' * x_0 + √(1-αt') * ε
        x_t = (
            self.sqrtab[_ts, None, None, None] * x
            + self.sqrtmab[_ts, None, None, None] * eps
        )  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this x_t. Loss is what we return.
        
        # 実際のノイズとモデルの推定ノイズを計算する。
        return self.criterion(eps, self.eps_model(x_t, _ts / self.n_T))
    
    #逆拡散過程、ノイズをどんどん下げていくよ。n_sampleはbatch sizeで、sizeは画像のサイズかな。
    def sample(self, n_sample: int, size, device) -> torch.Tensor:
        
        #x_iとあるがこの時点では0step目の完全なノイズ状態であるx_0
        x_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1)
    
        # This samples accordingly to Algorithm 2. It is exactly the same logic.
        
        for i in range(self.n_T, 0, -1):
            
            # reverse_process、式を全然理解していないが論文通りなのだろう
            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0
            eps = self.eps_model(x_i, i / self.n_T)
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )

        return x_i

# mnistでtrainする。pytorchの標準的なトレーニングコードであり特筆すべきところはなさそう。
def train_mnist(n_epoch: int = 100, device="cuda:0") -> None:
    
    ddpm = DDPM(eps_model=DummyEpsModel(1), betas=(1e-4, 0.02), n_T=1000)
    ddpm.to(device)

    tf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (1.0))]
    )

    dataset = MNIST(
        "./data",
        train=True,
        download=True,
        transform=tf,
    )
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=20)
    optim = torch.optim.Adam(ddpm.parameters(), lr=2e-4)

    for i in range(n_epoch):
        ddpm.train()

        pbar = tqdm(dataloader)
        loss_ema = None
        for x, _ in pbar:
            optim.zero_grad()
            x = x.to(device)
            loss = ddpm(x)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.9 * loss_ema + 0.1 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            optim.step()

        ddpm.eval()
        with torch.no_grad():
            xh = ddpm.sample(16, (1, 28, 28), device)
            grid = make_grid(xh, nrow=4)
            save_image(grid, f"./contents/ddpm_sample_{i}.png")

            # save model
            torch.save(ddpm.state_dict(), f"./ddpm_mnist.pth")


if __name__ == "__main__":
    train_mnist()
