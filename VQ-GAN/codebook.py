import torch
import torch.nn as nn


class Codebook(nn.Module):
    def __init__(self, args):
        super(Codebook, self).__init__()
        self.num_codebook_vectors = args.num_codebook_vectors
        self.latent_dim = args.latent_dim
        self.beta = args.beta

        self.embedding = nn.Embedding(
            num_embeddings=self.num_codebook_vectors, embedding_dim=self.latent_dim
        )
        self.embedding.weight.data.uniform_(
            -1 / self.num_codebook_vectors, 1 / self.num_codebook_vectors
        )

    def forward(self, z_e):
        z_e = z_e.permute(
            0, 2, 3, 1
        ).contiguous()  # Move the channel dimension to the last and make contiguous
        z_flattened = z_e.view(
            -1, self.latent_dim
        )  # Flatten the tensor to shape (batch_size * height * width, latent_dim)
        d = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(z_flattened, self.embedding.weight.t())
        )  # Compute the distance between the latent vectors and the codebook vectors (z_e - e)^2 = z^2 + e^2 - 2ze -> d = z^2 + e^2 - 2ze
        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z_e.shape)

        loss = torch.mean((z_q.detach() - z_e) ** 2) + self.beta * torch.mean(
            (z_q - z_e.detach()) ** 2
        )

        z_q = z_e + (z_q - z_e).detach()
        z_q = z_q.permute(
            0, 3, 1, 2
        ).contiguous()  # Move the channel dimension back to the second dimension

        return z_q, min_encoding_indices, loss
