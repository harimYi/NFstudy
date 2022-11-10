import torch
import torch.nn as nn
import torch.optim as optim
# PyTorch Lightning
import pytorch_lightning as pl
import numpy as np


class ImageFlow(pl.LightningModule):

    def __init__(self, flows, import_samples=8):
        """
        Inputs:
            flows: A list of flows (each a nn.Module) that should be applied on the images
            import_samples: Number of importance sampmles to use during testing
        """
        super(ImageFlow, self).__init__()
        self.flows = nn.ModuleList(flows)
        self.import_samples = import_samples
        # Create prior distribution for final latent space
        # loc means mean and scale is sqrt(variance)
        self.prior = torch.distributions.normal.Normal(loc=0.0, scale=1.0)

    def forward(self, imgs):
        # The forward function is only used for visualizing the graph
        return self._get_likelihood(imgs)

    def encoder(self, imgs):
        # Given a batch of images, return the latent representation z and
        # ldj (log determinant jacobian of the transformations
        z, ldj = imgs, torch.zeros(imgs.shape[0], device=self.device)
        for flow in self.flows:
            z, ldj = flow(z, ldj, reverse=False)
        return z, ldj

    def _get_likelihood(self, imgs, return_ll=False):
        """
        ll: log likelihood
        """
        # flows extract z representation from given imgs
        # and predict ldj (log determinant jacobian).
        z, ldj = self.encoder(imgs)
        log_pz = self.prior.log_prob(z).sum(dim=[1,2,3])
        # By using the results, log_px is calculated.
        # log_px = log_pz + log(|det(dz/dx)|) (ldj)
        log_px = log_pz + ldj
        # negative log likelihood (nll) should be minimized
        # since log_px is maximized if it is well predicted by flows
        nll = -log_px
        # bit per dimension
        # dimension includes width, height, and channels, which can be obtained imgs.shape[1:]
        bpd = nll * np.log2(np.exp(1)) / np.prod(imgs.shape[1:])
        return bpd.mean() if not return_ll else log_px

    @torch.no_grad()
    def sample(self, img_shape, z_init=None):
        """ Sample a batch of images from the flow """

        # Sample latent representation from prior
        if z_init is None:
            z = self.prior.sample(sample_shape=img_shape).to(self.device)
        else:
            z = z_init.to(self.device)

        # Transform z to x by inverting the flows
        ldj = torch.zeros(img_shape[0], device=self.device)
        for flow in reversed(self.flows):
            z, ldj = flow(z, ldj, reverse=True)
        return z

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.99)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        # Normalizing flows are trained by maximum likelihood => return bpd
        loss = self._get_likelihood(batch[0])
        self.log('train_bpd', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._get_likelihood(batch[0])
        self.log('val_bpd', loss)

    def test_step(self, batch, batch_idx):
        samples = []
        for _ in range(self.import_samples):
            # this obtains log_px from imgs.
            # That is, imgs --> pz --> log_px. Hence, log_px is the same as log_imgs.
            img_ll = self._get_likelihood(batch[0], return_ll=True)
            samples.append(img_ll)
        img_ll = torch.stack(samples, dim=-1)

        # img_ll has log_px
        # exp(img_ll) --> exp(log_px) --> px
        img_ll = torch.logsumexp(img_ll, dim=-1) - np.log(self.import_samples)

        # Calculate final bpd
        bpd = -img_ll * np.log2(np.exp(1)) / np.prod(batch[0].shape[1:])
        bpd = bpd.mean()

        self.log('test_bpd', bpd)




