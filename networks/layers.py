import torch
import torch.nn as nn


class CouplingLayer(nn.Module):

    def __init__(self, network, mask, c_in):
        """
        Coupling layer inside a normalizing flow.
        Input:
            network - A PyTorch nn.Module constituting the deep neural network for mu and sigma.
                      Output shape should be twice the channel size as the input.
            mask - Binary mask (0 or 1) where 0 denotes that the element should be transformed,
                   while 1 means the latent will be used as input to the NN.
                   CouplingLayer transforms half of z representation while the other half isn't changed.
            c_in - Number of input channels
        """
        super().__init__()
        self.network = network
        self.scaling_factor = nn.Parameter(torch.zeros(c_in))
        self.register_buffer('mask', mask)

    def forward(self, z, ldj, reverse=False, orig_img=None):
        """
        :param z: Latent input to the flow
        :param ldj: The current ldj of the previous flows.
        :param reverse: If True, we apply the inverse of the layer.
        :param orig_img (optional):
        :return:
        """
        z_in = z * self.mask  # z_{1:j}
        if orig_img is None:
            # Predict the bias and scale for transformation
            nn_out = self.network(z_in)
        else:
            nn_out = self.network(torch.cat([z_in, orig_img], dim=1))
        # Split nn_out to the scale and bias
        s, t = nn_out.chunk(2, dim=1)

        # Stabilize scaling output
        s_fac = self.scaling_factor.exp().view(1, -1, 1, 1)
        s = torch.tanh(s / s_fac) * s_fac

        # Mask outputs (only transform the second part)
        s = s * (1 - self.mask)  # scale for the second part, z_{j+1:d}
        t = t * (1 - self.mask)  # bias for the second part, z_{j+1:d}

        # Affine transformation
        if not reverse:
            z = (z + t) * torch.exp(s)
            ldj += s.sum(dim=[1,2,3])
        else:
            z = (z * torch.exp(-s)) - t
            ldj -= s.sum(dim=[1,2,3])
        return z, ldj



