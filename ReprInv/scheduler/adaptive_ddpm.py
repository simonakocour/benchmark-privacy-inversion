import torch
import torch.nn.functional as F


class AdaptiveDDPMScheduler:
    def __init__(self, base_scheduler, mode: str = 'later_for_dense', strength: float = 0.5, downsample_factor: int = 8, method: str = 'trim',
                 roi_blend: bool = True, roi_strength: float = 0.25, roi_delta_steps: int = 1, roi_sigma: float = 1.5):
        self._base = base_scheduler
        self.mode = mode
        self.strength = float(max(0.0, min(1.0, strength)))
        self.downsample_factor = int(max(1, downsample_factor))
        # method: 'resample' keeps length; 'trim' removes steps
        self.method = method
        # ROI blending configuration (location-aware)
        self.roi_blend = bool(roi_blend)
        self.roi_strength = float(max(0.0, min(1.0, roi_strength)))
        self.roi_delta_steps = int(max(1, roi_delta_steps))
        self.roi_sigma = float(max(0.5, roi_sigma))
        self._positions = None

    def __getattr__(self, name):
        # Delegate to base scheduler for all attributes/methods not overridden
        return getattr(self._base, name)

    def set_timesteps(self, num_inference_steps: int, positions=None, *args, **kwargs):
        self._base.set_timesteps(num_inference_steps, *args, **kwargs)
        self._positions = positions
        if positions is None or self.strength <= 0:
            return
        counts = []
        if isinstance(positions, list):
            for pos in positions:
                if pos is None:
                    counts.append(0)
                    continue
                if pos.ndim == 2 and pos.shape[1] >= 2:
                    valid = (pos[:, 0] != -1)
                    counts.append(int(valid.sum().item()))
                else:
                    counts.append(int(pos.shape[0]))
        else:
            # Tensor (B, N, 2)
            for b in range(positions.shape[0]):
                pos = positions[b]
                if pos.ndim == 2 and pos.shape[1] >= 2:
                    valid = (pos[:, 0] != -1)
                    counts.append(int(valid.sum().item()))
                else:
                    counts.append(int(pos.shape[0]))

        device = self._base.timesteps.device
        counts_t = torch.tensor(counts, device=device, dtype=torch.float32)
        if counts_t.max() > 0:
            score = (counts_t / counts_t.max().clamp(min=1.0)).mean()
        else:
            score = torch.tensor(0.0, device=device)

        timesteps = self._base.timesteps
        total = timesteps.shape[0]
        if total <= 2:
            return
        if self.method == 'trim':
            max_trim = max(1, total // 4)
            trim = int((self.strength * float(score)) * max_trim)
            if trim <= 0:
                return
            min_tail_keep = max(8, total // 5)
            min_head_keep = max(4, total // 10)
            if self.mode == 'later_for_dense':
                tail_trim = min(trim, total - min_head_keep - 1)
                tail_trim = max(0, tail_trim)
                new_timesteps = timesteps[:-tail_trim] if tail_trim > 0 else timesteps
            else:
                head_trim = min(trim, total - min_tail_keep - 1)
                head_trim = max(0, head_trim)
                new_timesteps = timesteps[head_trim:]
        else:
            N = total
            idx = torch.arange(N, device=timesteps.device, dtype=torch.float32)
            s = idx / (N - 1)
            alpha = float(self.strength * float(score))
            if self.mode == 'later_for_dense':
                gamma = max(0.2, 1.0 - 0.7 * alpha)
                s_warp = s.pow(gamma)
            else:
                gamma = max(0.2, 1.0 - 0.7 * alpha)
                s_warp = 1.0 - (1.0 - s).pow(gamma)
            t_asc = timesteps.flip(0)
            pos = s_warp * (N - 1)
            i0 = pos.floor().long()
            i1 = (i0 + 1).clamp(max=N - 1)
            w = (pos - i0.to(pos.dtype)).unsqueeze(-1)
            vals0 = t_asc.index_select(0, i0)
            vals1 = t_asc.index_select(0, i1)
            new_asc = (1.0 - w) * vals0.unsqueeze(-1) + w * vals1.unsqueeze(-1)
            new_asc = new_asc.squeeze(-1)
            new_desc = new_asc.flip(0)
            new_desc = new_desc.round().to(dtype=timesteps.dtype)
            new_desc = torch.minimum(new_desc, torch.cat([new_desc[:1], new_desc[:-1]], dim=0))
            new_desc[0] = timesteps[0]
            new_desc[-1] = timesteps[-1]
            self._base.timesteps = new_desc

    def _gaussian_kernel2d(self, kernel_size: int, sigma: float, device, dtype):
        coords = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1) / 2.0
        x = coords.view(1, -1)
        y = coords.view(-1, 1)
        kernel = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        return kernel

    def _build_roi_mask(self, positions, height: int, width: int, device, dtype):
        if positions is None:
            return None
        pos = positions[0] if isinstance(positions, list) else positions[0]
        if pos is None or pos.numel() == 0:
            return None
        if pos.ndim == 2 and pos.shape[1] >= 2:
            valid = (pos[:, 0] != -1)
            pts = pos[valid, :2]
        else:
            pts = pos
        if pts.numel() == 0:
            return None
        mask = torch.zeros((1, 1, height, width), device=device, dtype=dtype)
        xy = (pts[:, :2] / float(self.downsample_factor)).round().long()
        xy[:, 0] = xy[:, 0].clamp(0, width - 1)
        xy[:, 1] = xy[:, 1].clamp(0, height - 1)
        mask[0, 0, xy[:, 1], xy[:, 0]] = 1.0
        ksize = int(max(3, int(round(6 * self.roi_sigma)) | 1))
        kernel = self._gaussian_kernel2d(ksize, self.roi_sigma, device=device, dtype=dtype)
        kernel = kernel.view(1, 1, ksize, ksize)
        padding = ksize // 2
        mask = F.conv2d(mask, kernel, padding=padding)
        mmin = mask.amin(dim=(2, 3), keepdim=True)
        mmax = mask.amax(dim=(2, 3), keepdim=True)
        mask = (mask - mmin) / (mmax - mmin + 1e-6)
        return mask

    def step(self, model_output, timestep, sample, **kwargs):
        base_out = self._base.step(model_output, timestep, sample, **kwargs)
        if not self.roi_blend or self._positions is None:
            return base_out

        height, width = sample.shape[-2], sample.shape[-1]
        mask = self._build_roi_mask(self._positions, height, width, device=sample.device, dtype=sample.dtype)
        if mask is None or (mask <= 0).all():
            return base_out
        timesteps = self._base.timesteps
        tvec = timesteps.view(-1)
        diffs = (tvec.to(sample.device).to(torch.long) - timestep.to(sample.device).to(torch.long)).abs()
        idx = int(diffs.argmin().item())
        if self.mode == 'later_for_dense':
            j = min(idx + self.roi_delta_steps, tvec.shape[0] - 1)
        else:
            j = max(idx - self.roi_delta_steps, 0)
        alt_t = tvec[j]

        alt_out = self._base.step(model_output, alt_t, sample, **kwargs)
        if not hasattr(base_out, 'prev_sample') or not hasattr(alt_out, 'prev_sample'):
            return base_out
        prev = base_out.prev_sample
        prev_alt = alt_out.prev_sample
        w = mask.to(prev.dtype) * self.roi_strength
        blended = prev * (1.0 - w) + prev_alt * w
        try:
            base_out.prev_sample = blended
            return base_out
        except Exception:
            alt_out.prev_sample = blended
            return alt_out

    @torch.no_grad()
    def sample_train_timesteps(self, batch_size: int, positions, device: torch.device) -> torch.LongTensor:
        num_steps = int(self._base.config.num_train_timesteps)
        if num_steps <= 1:
            return torch.zeros(batch_size, device=device, dtype=torch.long)

        # we compute normalized density score per sample using count of valid positions
        counts = []
        if isinstance(positions, list):
            for pos in positions:
                if pos is None:
                    counts.append(0)
                    continue
                if pos.ndim == 2 and pos.shape[1] >= 2:
                    valid = (pos[:, 0] != -1)
                    counts.append(int(valid.sum().item()))
                else:
                    counts.append(int(pos.shape[0]))
        else:
            # Tensor (B, N, 2)
            for b in range(batch_size):
                pos = positions[b]
                if pos.ndim == 2 and pos.shape[1] >= 2:
                    valid = (pos[:, 0] != -1)
                    counts.append(int(valid.sum().item()))
                else:
                    counts.append(int(pos.shape[0]))

        counts_t = torch.tensor(counts, device=device, dtype=torch.float32)
        if counts_t.max() > 0:
            scores = counts_t / counts_t.max().clamp(min=1.0)
        else:
            scores = torch.zeros_like(counts_t)
        strength = self.strength
        if self.mode == 'later_for_dense':
            e = (1.0 - strength * scores).clamp(min=0.2)
        else:
            e = (1.0 + strength * scores).clamp(max=2.0)

        u = torch.rand(batch_size, device=device)
        u = torch.pow(u, e)
        timesteps = (u * num_steps).long().clamp(0, num_steps - 1)
        return timesteps


