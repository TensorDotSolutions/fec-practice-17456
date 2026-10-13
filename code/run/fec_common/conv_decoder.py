"""PyTorch conv decoder for psi(R,Z) -- Colab/GPU only, not used by the local CPU pipeline.

Not imported by anything in fec_common/__init__.py (it needs torch, which the local venv
intentionally doesn't install -- see requirements-local.txt). Import it explicitly from a Colab
cell once torch is available there by default.

Why a decoder, not PCA + a flat regressor (what 02/02b/02c do): PCA linearizes psi's spatial
structure away before any model sees it, so a Ridge/MLP head is fitting 50 decorrelated
coefficients, not a 65x65 image. The one real local_score.json run showed R2_psi pointwise is
decent (0.87) but every *geometric* consistency scalar derived from the LCFS contour --
kappa, tri_top, tri_bot, volume -- has strongly negative R2. That pattern (fine on raw pixel
error, bad on shape) is exactly what you'd expect from a model that doesn't know psi is an
image: small per-pixel errors can still distort the contour topology the scorer extracts.
A conv decoder with transposed convolutions is structured to preserve that topology instead of
reconstructing it from a handful of linear PCA directions.

Architecture
------------
Input: the same 149-dim scaled feature vector 01/02 already build (magnetics + repaired
Thomson). Output: (65, 65) psi, plus q95/betaN from the same shared trunk (no separate scalar
model needed -- same rationale as the starter's ScalarTargetModels, just folded into one net).

    features (149)
      -> FC(149->512) -> BN -> ReLU -> Dropout(0.1)
      -> FC(512->3200) -> BN -> ReLU                 # 3200 = 128 * 5 * 5
      -> reshape (128, 5, 5)
      -> ConvTranspose2d(128->64, k3, s2, p1) -> BN -> ReLU    # 5x5  -> 9x9
      -> ConvTranspose2d(64->32,  k3, s2, p1) -> BN -> ReLU    # 9x9  -> 17x17
      -> ConvTranspose2d(32->16,  k3, s2, p1) -> BN -> ReLU    # 17x17-> 33x33
      -> ConvTranspose2d(16->1,   k3, s2, p1)                  # 33x33-> 65x65  (psi, linear out)

    (shared trunk's 512-dim layer also feeds two small heads: Linear(512->64)->ReLU->Linear(64->1)
    for q95, same shape for betaN -- cheap, and lets the net learn features useful for both psi
    and the scalars jointly instead of a separate scalar model seeing none of that signal.)

Each ConvTranspose2d uses kernel=3, stride=2, padding=1, output_padding=0, which maps an NxN
input to exactly (2N-1)x(2N-1) -- the 5 -> 9 -> 17 -> 33 -> 65 chain is exact, no cropping or
padding hacks needed to land on the required 65x65 output.

Loss: MSE on psi (the scorer's R2_psi term) + a small weight on q95/betaN MSE + a light Sobel
gradient-matching term, which directly penalizes the kind of smoothed-away curvature that wrecks
D_LCFS/Consistency without needing the (slow, non-differentiable) contour extraction itself in
the training loop.

Memory: with batch=256 and float32, peak activation memory is a few hundred MB -- fits in a
3.5GB GPU. If you hit OOM in Colab, drop --batch-size first (not the architecture).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PsiConvDecoder(nn.Module):
    def __init__(self, n_feat: int, dropout: float = 0.1):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(n_feat, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.to_grid = nn.Sequential(
            nn.Linear(512, 128 * 5 * 5), nn.BatchNorm1d(128 * 5 * 5), nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1),  # 5 -> 9
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1),   # 9 -> 17
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1),   # 17 -> 33
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1),    # 33 -> 65
        )
        self.q95_head = nn.Sequential(nn.Linear(512, 64), nn.ReLU(inplace=True), nn.Linear(64, 1))
        self.betan_head = nn.Sequential(nn.Linear(512, 64), nn.ReLU(inplace=True), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        grid = self.to_grid(h).view(-1, 128, 5, 5)
        psi = self.decoder(grid).squeeze(1)  # (B, 65, 65)
        q95 = self.q95_head(h).squeeze(-1)
        beta_n = self.betan_head(h).squeeze(-1)
        return psi, q95, beta_n


_SOBEL_X = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
_SOBEL_Y = _SOBEL_X.t()


def sobel_gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Penalize mismatched local curvature, not just pointwise error -- see module docstring."""
    device = pred.device
    kx = _SOBEL_X.to(device).view(1, 1, 3, 3)
    ky = _SOBEL_Y.to(device).view(1, 1, 3, 3)
    p, t = pred.unsqueeze(1), target.unsqueeze(1)
    gx_p, gy_p = F.conv2d(p, kx, padding=1), F.conv2d(p, ky, padding=1)
    gx_t, gy_t = F.conv2d(t, kx, padding=1), F.conv2d(t, ky, padding=1)
    return F.mse_loss(gx_p, gx_t) + F.mse_loss(gy_p, gy_t)


def psi_loss(psi_pred, psi_true, q95_pred, q95_true, betan_pred, betan_true,
            scalar_weight: float = 0.05, grad_weight: float = 0.1) -> torch.Tensor:
    mse = F.mse_loss(psi_pred, psi_true)
    grad = sobel_gradient_loss(psi_pred, psi_true)
    scal = F.mse_loss(q95_pred, q95_true) + F.mse_loss(betan_pred, betan_true)
    return mse + grad_weight * grad + scalar_weight * scal


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# V2 -- targets the Consistency term specifically (see local_score.json: psi_R2=0.99 but the
# LCFS-contour-derived shape scalars -- kappa, tri_top/bot, R_axis/Z_axis -- sit at 0.68-0.78).
# Those come from the SCORER re-deriving them from the predicted psi image itself (see
# fusion_scoring/derive.py) -- there is no auxiliary scalar head that can shortcut this, the
# fix has to be psi's own spatial fidelity. Three changes, each targeting a known failure mode
# of V1's plain ConvTranspose2d stack:
#
# 1. CoordConv (Liu et al. 2018): concat fixed normalized (R, Z) coordinate channels into the
#    decoder at every stage. Plain conv stacks have no way to know a pixel's absolute position
#    -- translation equivariance is the wrong prior here, since R_axis/Z_axis are defined by
#    absolute grid position, not a translatable pattern.
# 2. Upsample(bilinear) + Conv2d instead of ConvTranspose2d: strided transposed convs are the
#    textbook cause of checkerboard artifacts, which directly corrupt the LCFS contour
#    extraction (skimage traces a noisy, jagged boundary instead of a smooth one).
# 3. A residual block at each resolution stage before upsampling, for more capacity without
#    making optimization harder (identity shortcut keeps gradients flowing to early layers).
# ---------------------------------------------------------------------------

class _CoordConvResBlock(nn.Module):
    """Conv residual block with two fixed coordinate channels concatenated onto the input."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels + 2, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels + 2, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    @staticmethod
    def _coords(x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        ys = torch.linspace(-1, 1, h, device=x.device, dtype=x.dtype).view(1, 1, h, 1).expand(b, 1, h, w)
        xs = torch.linspace(-1, 1, w, device=x.device, dtype=x.dtype).view(1, 1, 1, w).expand(b, 1, h, w)
        return torch.cat([xs, ys], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = self._coords(x)
        h = F.relu(self.bn1(self.conv1(torch.cat([x, c], dim=1))), inplace=True)
        h = self.bn2(self.conv2(torch.cat([h, c], dim=1)))
        return F.relu(x + h, inplace=True)


class _UpBlock(nn.Module):
    """Bilinear upsample (no checkerboard artifacts) + conv to change channel count + a
    CoordConv residual block at the new resolution."""

    def __init__(self, in_ch: int, out_ch: int, out_size: int):
        super().__init__()
        self.out_size = out_size
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.res = _CoordConvResBlock(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=(self.out_size, self.out_size), mode="bilinear", align_corners=False)
        x = F.relu(self.bn(self.proj(x)), inplace=True)
        return self.res(x)


class PsiConvDecoderV2(nn.Module):
    """See the V2 module-level comment above for the rationale. Wider bottleneck (192 vs 128
    channels) than V1 to carry more global shape information into the spatial decoder."""

    def __init__(self, n_feat: int, dropout: float = 0.1):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(n_feat, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.to_grid = nn.Sequential(
            nn.Linear(512, 192 * 5 * 5), nn.BatchNorm1d(192 * 5 * 5), nn.ReLU(inplace=True),
        )
        self.up1 = _UpBlock(192, 96, out_size=9)
        self.up2 = _UpBlock(96, 48, out_size=17)
        self.up3 = _UpBlock(48, 24, out_size=33)
        self.up4 = _UpBlock(24, 16, out_size=65)
        self.out = nn.Conv2d(16, 1, kernel_size=3, padding=1)
        self.q95_head = nn.Sequential(nn.Linear(512, 64), nn.ReLU(inplace=True), nn.Linear(64, 1))
        self.betan_head = nn.Sequential(nn.Linear(512, 64), nn.ReLU(inplace=True), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        grid = self.to_grid(h).view(-1, 192, 5, 5)
        grid = self.up1(grid)
        grid = self.up2(grid)
        grid = self.up3(grid)
        grid = self.up4(grid)
        psi = self.out(grid).squeeze(1)  # (B, 65, 65)
        q95 = self.q95_head(h).squeeze(-1)
        beta_n = self.betan_head(h).squeeze(-1)
        return psi, q95, beta_n


def boundary_weighted_mse(pred: torch.Tensor, target: torch.Tensor, boost: float = 3.0) -> torch.Tensor:
    """Pointwise MSE, upweighted near the separatrix (steep |grad psi_true| region).

    The Consistency term's shape scalars (kappa, tri_top/bot, R_axis/Z_axis, volume) are all
    read off the LCFS contour -- a thin ridge in psi-gradient space. Plain MSE spends most of
    its gradient budget on the broad, easy, low-gradient interior/exterior; this reweights
    toward the boundary region that actually determines those scalars, without needing the
    (non-differentiable) contour extraction itself in the loss.
    """
    device = target.device
    kx = _SOBEL_X.to(device).view(1, 1, 3, 3)
    ky = _SOBEL_Y.to(device).view(1, 1, 3, 3)
    t = target.unsqueeze(1)
    gx, gy = F.conv2d(t, kx, padding=1), F.conv2d(t, ky, padding=1)
    grad_mag = torch.sqrt(gx ** 2 + gy ** 2).squeeze(1)
    w = 1.0 + boost * (grad_mag / (grad_mag.amax(dim=(1, 2), keepdim=True) + 1e-8))
    return (w * (pred - target) ** 2).mean()


def psi_loss_v2(psi_pred, psi_true, q95_pred, q95_true, betan_pred, betan_true,
                scalar_weight: float = 0.05, grad_weight: float = 0.1,
                boundary_weight: float = 0.3) -> torch.Tensor:
    mse = F.mse_loss(psi_pred, psi_true)
    grad = sobel_gradient_loss(psi_pred, psi_true)
    boundary = boundary_weighted_mse(psi_pred, psi_true)
    scal = F.mse_loss(q95_pred, q95_true) + F.mse_loss(betan_pred, betan_true)
    return mse + grad_weight * grad + boundary_weight * boundary + scalar_weight * scal


if __name__ == "__main__":
    # Smoke test only -- run this cell in Colab (needs torch) before training for real.
    m = PsiConvDecoder(n_feat=149)
    x = torch.randn(8, 149)
    psi, q95, beta_n = m(x)
    assert psi.shape == (8, 65, 65), psi.shape
    assert q95.shape == (8,) and beta_n.shape == (8,)
    print(f"V1 params: {count_params(m):,}  psi {psi.shape}  q95 {q95.shape}  betaN {beta_n.shape}")

    m2 = PsiConvDecoderV2(n_feat=149)
    psi2, q95_2, beta_n2 = m2(x)
    assert psi2.shape == (8, 65, 65), psi2.shape
    loss = psi_loss_v2(psi2, psi, q95_2, q95, beta_n2, beta_n)
    assert torch.isfinite(loss)
    print(f"V2 params: {count_params(m2):,}  psi {psi2.shape}  loss {loss.item():.4f}")
