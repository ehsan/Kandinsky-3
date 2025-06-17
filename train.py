import os
from argparse import ArgumentParser

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

import pytorch_lightning as pl
from datasets import load_dataset
from omegaconf import OmegaConf

from kandinsky3.model.unet import UNet
from kandinsky3.movq import MoVQ
from kandinsky3.condition_encoders import T5TextConditionEncoder
from kandinsky3.condition_processors import T5TextConditionProcessor
from kandinsky3.model.diffusion import get_named_beta_schedule, BaseDiffusion


class TextImageDataset(torch.utils.data.Dataset):
    def __init__(self, name, split, image_field, text_field, image_size):
        self.ds = load_dataset(name, split=split)
        self.image_field = image_field
        self.text_field = text_field
        self.transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)
        ])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]
        image = sample[self.image_field]
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")
        image = self.transform(image)
        caption = sample[self.text_field]
        return {"image": image, "text": caption}


class KandinskyLightningModule(pl.LightningModule):
    def __init__(self, conf):
        super().__init__()
        self.save_hyperparameters(OmegaConf.to_container(conf, resolve=True))
        self.conf = conf

        self.unet = UNet(**conf.model.unet)
        self.movq = MoVQ(conf.model.movq.params)
        self.t5_processor = T5TextConditionProcessor(conf.data.tokens_length, conf.model.text_encoder.model_path)
        self.t5_encoder = T5TextConditionEncoder(
            conf.model.text_encoder.model_path,
            conf.model.unet.context_dim,
            low_cpu_mem_usage=conf.model.text_encoder.low_cpu_mem_usage,
            dtype=getattr(torch, conf.model.text_encoder.dtype)
        )

        betas = get_named_beta_schedule(**conf.model.diffusion.schedule_params)
        self.diffusion = BaseDiffusion(betas, **conf.model.diffusion.diffusion_params)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.conf.optimizer.lr, weight_decay=self.conf.optimizer.weight_decay)
        return opt

    def training_step(self, batch, batch_idx):
        images = batch["image"]
        texts = batch["text"]

        encoded = [self.t5_processor.encode(t)[0] for t in texts]
        input_ids = torch.stack([e['input_ids'] for e in encoded]).to(self.device)
        attention_mask = torch.stack([e['attention_mask'] for e in encoded]).to(self.device)
        condition_model_input = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
        with torch.no_grad():
            context, context_mask = self.t5_encoder(condition_model_input)
            latents = self.movq.encode(images)

        t = torch.randint(0, self.diffusion.num_timesteps, (images.size(0),), device=self.device)
        noise = torch.randn_like(latents)
        noisy_latents = self.diffusion.q_sample(latents, t, noise)
        pred_noise = self.unet(noisy_latents, t * self.diffusion.time_scale, context, context_mask.bool())

        loss = F.mse_loss(pred_noise, noise)
        self.log("loss", loss)
        return loss


def main():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    conf = OmegaConf.load(args.config)

    dm = TextImageDataset(
        conf.data.dataset_name,
        conf.data.split,
        conf.data.image_field,
        conf.data.text_field,
        conf.data.image_size,
    )
    dataloader = DataLoader(dm, batch_size=conf.data.batch_size, num_workers=conf.data.num_workers, shuffle=True, drop_last=True)

    model = KandinskyLightningModule(conf)
    trainer = pl.Trainer(
        max_epochs=conf.trainer.max_epochs,
        accelerator=conf.trainer.accelerator,
        precision=conf.trainer.precision,
    )
    trainer.fit(model, dataloader)


if __name__ == "__main__":
    main()
