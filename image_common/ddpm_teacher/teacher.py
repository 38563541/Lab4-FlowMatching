import torch
from .model import DiffusionModule
from .scheduler import DDPMScheduler

class DDPMTeacher:
    def __init__(self, ckpt_path, device):
        self.device = device
        self.model = self._load_model(ckpt_path)
        self.model.eval().to(device)

    def _load_model(self, ckpt_path):
        ddpm = DiffusionModule(None, None)
        ddpm.load(ckpt_path)
        
        T = ddpm.var_scheduler.num_train_timesteps
        ddpm.var_scheduler = DDPMScheduler(
            T,
            beta_1=1e-4,
            beta_T=0.02,
            mode='linear',
        ).to(self.device)
        
        return ddpm

    @property
    def image_resolution(self):
        return self.model.image_resolution

    @torch.no_grad()
    def sample(self, shape, class_label=None, guidance_scale=7.5, num_inference_timesteps=None, verbose=True):
        batch_size = shape[0]
        
        if self.model.network.use_cfg:
            assert class_label is not None, "Class label must be provided for CFG model."
            samples = self.model.sample(
                batch_size,
                class_label=class_label,
                guidance_scale=guidance_scale,
            )
        else:
            samples = self.model.sample(batch_size)
            
        return samples
