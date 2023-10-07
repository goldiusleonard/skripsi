import math
import warnings

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from torchshard import register_parallel_dim, get_parallel_dim
from torchshard.nn.functional import parallel_cross_entropy
from torchshard.distributed import get_rank, scatter, copy, get_world_size


class ParallelMagFaceLoss(torch.nn.Module):
    """Based on codes by IrvinMeng, which uses torchshard for model parallelization.

    Simplifed version which acts as a criterion with parameters.
    """

    def __init__(
        self,
        num_features: int,
        classnum: int,
        u_margin: float,
        l_margin: float,
        u_a: float,
        l_a: float,
        lambda_g: float,
        scale: float = 64.0,
        easy_margin: bool = True,
    ):
        """Initialize ParallelMagFace

        Parameters
        ----------
        num_features : int
            Number of feature vectors outputted by backbone.
        classnum : int
            Number of identities/classes for target, must be divisible
            by the world size.
        u_margin : float
            Upper bound for margin.
        l_margin : float
            Lower bound for margin.
        u_a : float
            Upper bound for magnitude.
        l_a : float
            Lower bound for magnitude.
        lambda_g: float
            Factor used for G-loss.
        easy_margin: bool, optional
            Option to relax margin contraints, by default True
        scale : float, optional
            Hyperparameter s for logit scaling, by default 64.0
        """
        super().__init__()
        self.num_features = num_features
        self.classnum = classnum
        self.scale = scale
        self.u_margin = u_margin
        self.l_margin = l_margin
        self.u_a = u_a
        self.l_a = l_a
        self.lambda_g = lambda_g
        self.easy_margin = easy_margin

        self.weight = torch.Tensor(self.classnum, self.num_features).to(get_rank())

        self.reset_parameters()
        self.slice_params()

    def reset_parameters(self):
        """Reinitialize parameters."""
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def slice_params(self):
        """Slice params to prepare for Model Parallel"""
        self.weight = scatter(self.weight, dim=0)
        self.weight = Parameter(self.weight)
        register_parallel_dim(self.weight, -1)

    def _margin(self, x_norm: torch.Tensor) -> torch.Tensor:
        """Calculate adaptive margin based on magnitude.

        Parameters
        ----------
        x_norm : torch.Tensor
            Norm for each embeddings.

        Returns
        -------
        torch.Tensor
            Margin calculated from given norms.
        """
        margin = (self.u_margin - self.l_margin) / (self.u_a - self.l_a) * (
            x_norm - self.l_a
        ) + self.l_margin
        return margin

    def calc_loss_G(self, x_norm: torch.Tensor) -> torch.Tensor:
        """Calculate G loss as specified by MagFace.

        Parameters
        ----------
        x_norm : torch.Tensor
            Norms for all embeddings.

        Returns
        -------
        torch.Tensor
            G-Loss value.
        """
        g = 1 / (self.u_a ** 2) * x_norm + 1 / (x_norm)
        return torch.mean(g)

    def forward(
        self, x: torch.Tensor, x_norm: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Calculate logits with adaptive margin.

        Parameters
        ----------
        x : torch.Tensor
            Embeddings produced by backbone.
        x_norm : torch.Tensor
            Norms for each embeddings.
        labels : torch.Tensor
            Labels/targets for each embeddings.
            Type must be torch.LongTensor/torch.int64

        Returns
        -------
        torch.Tensor
            Total loss.
            It is obtained by adding softmax loss with G-loss weighted by lambda.
        """
        x = copy(x.float())

        ada_margin = self._margin(x_norm)
        cos_m = torch.cos(ada_margin)
        sin_m = torch.sin(ada_margin)

        # norm the weight
        weight_norm = F.normalize(self.weight, dim=-1)
        cos_theta = torch.mm(F.normalize(x), weight_norm.t())
        cos_theta = cos_theta.clamp(-1, 1)
        sin_theta = torch.sqrt(1.0 - torch.pow(cos_theta, 2))

        cos_theta_m = cos_theta * cos_m - sin_theta * sin_m

        if self.easy_margin:
            cos_theta_m = torch.where(
                cos_theta.float() > 0, cos_theta_m.float(), cos_theta.float()
            )
        else:
            mm = torch.sin(math.pi - ada_margin) * ada_margin
            threshold = torch.cos(math.pi - ada_margin)
            cos_theta_m = torch.where(
                cos_theta.float() > threshold,
                cos_theta_m.float(),
                cos_theta.float() - mm,
            )

        # multiply the scale in advance
        cos_theta_m = self.scale * cos_theta_m
        cos_theta = self.scale * cos_theta

        # set parallel attribute
        register_parallel_dim(cos_theta_m, -1)
        register_parallel_dim(cos_theta, -1)

        world_size = get_world_size()
        one_hot = torch.zeros(
            (cos_theta.size(0), world_size * cos_theta.size(1)), device=cos_theta.device
        )
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        loss_g = self.calc_loss_G(x_norm)

        one_hot = scatter(one_hot, dim=-1)
        output = one_hot * cos_theta_m + (1.0 - one_hot) * cos_theta

        # set parallel attribute
        parallel_dim = get_parallel_dim(cos_theta)
        register_parallel_dim(output, parallel_dim)

        loss = parallel_cross_entropy(output, labels)
        return loss + (self.lambda_g * loss_g)

    def extra_repr(self):
        return "inp_features={}, out_features={}, scale={}".format(
            self.num_features, self.classnum, self.scale
        )


class ParallelMagFaceLossAlt(torch.nn.Module):
    """Based on codes by IrvinMeng, which uses torchshard for model parallelization.

    This version simplifies the code and equation, closely resembling
    official implementation of ArcFace by InsightFace.

    Had issues with multi-gpu training.
    """

    def __init__(
        self,
        num_features: int,
        classnum: int,
        u_margin: float,
        l_margin: float,
        u_a: float,
        l_a: float,
        lambda_g: float,
        scale=64.0,
    ):
        """Initialize ParallelMagFace

        Parameters
        ----------
        num_features : int
            Number of feature vectors outputted by backbone.
        classnum : int
            Number of identities/classes for target, must be divisible
            by the world size.
        u_margin : float
            Upper bound for margin.
        l_margin : float
            Lower bound for margin.
        u_a : float
            Upper bound for magnitude.
        l_a : float
            Lower bound for magnitude.
        lambda_g: float
            Factor used for G-loss.
        scale : float, optional
            Hyperparameter s for logit scaling, by default 64.0
        """
        super().__init__()
        warnings.warn(
            "This implementation of ParallelMagFaceLoss still has issues on Multi-GPU training!"
        )

        self.num_features = num_features
        self.classnum = classnum
        self.scale = scale
        self.u_margin = u_margin
        self.l_margin = l_margin
        self.u_a = u_a
        self.l_a = l_a
        self.lambda_g = lambda_g

        self.weight = torch.Tensor(self.classnum, self.num_features).to(get_rank())

        self.reset_parameters()
        self.slice_params()

    def reset_parameters(self):
        """Reinitialize parameters."""
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def slice_params(self):
        """Slice params to prepare for Model Parallel"""
        self.weight = scatter(self.weight, dim=0)
        self.weight = Parameter(self.weight)
        register_parallel_dim(self.weight, -1)

    def _margin(self, x_norm) -> torch.Tensor:
        """Calculate adaptive margin based on magnitude.

        Parameters
        ----------
        x_norm : torch.Tensor
            Norm for each embeddings.

        Returns
        -------
        torch.Tensor
            Margin calculated from given norms.
        """
        margin = (self.u_margin - self.l_margin) / (self.u_a - self.l_a) * (
            x_norm - self.l_a
        ) + self.l_margin
        return margin

    def calc_loss_G(self, x_norm: torch.Tensor) -> torch.Tensor:
        """Calculate G loss as specified by MagFace.

        Parameters
        ----------
        x_norm : torch.Tensor
            Norms for all embeddings.

        Returns
        -------
        torch.Tensor
            G-Loss value.
        """
        g = 1 / (self.u_a ** 2) * x_norm + 1 / (x_norm)
        return torch.mean(g)

    def forward(
        self, x: torch.Tensor, x_norm: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Calculate logits with adaptive margin.

        Parameters
        ----------
        x : torch.Tensor
            Embeddings produced by backbone.
        x_norm : torch.Tensor
            Norms for each embeddings.
        labels : torch.Tensor
            Labels/targets for each embeddings.
            Type must be torch.LongTensor/torch.int64

        Returns
        -------
        torch.Tensor
            Total loss.
            It is obtained by adding softmax loss with G-loss weighted by lambda.
        """
        x = copy(x.float())

        margin = self._margin(x_norm)
        weight_norm = F.normalize(self.weight, dim=-1)
        cos_theta = F.linear(F.normalize(x), weight_norm.t())
        cos_theta.clamp_(-1, 1)

        index = torch.where(labels != -1)[0]
        m_hot = torch.zeros(
            index.size()[0], cos_theta.size()[1], device=cos_theta.device
        )
        m_hot.scatter_(1, labels[index, None], margin.view(-1, 1))
        cos_theta.acos_()
        cos_theta[index] += m_hot
        cos_theta.cos_().mul_(self.scale)

        register_parallel_dim(cos_theta, -1)

        loss = parallel_cross_entropy(cos_theta, labels)
        g_loss = self.calc_loss_G(x_norm)

        return loss + (self.lambda_g * g_loss)

    def extra_repr(self):
        return "inp_features={}, out_features={}, scale={}".format(
            self.num_features, self.classnum, self.scale
        )
