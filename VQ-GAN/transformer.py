import torch
import torch.nn as nn
import torch.nn.functional as F
from mingpt import GPT
from vqgan import VQGAN


class VQGANTransformer(nn.Module):
    def __init__(self, args):
        super(VQGANTransformer, self).__init__()

        self.sos_token = args.sos_token

        self.vqgan = self.load_vqgan(args)

        transformer_config = {
            "vocab_size": args.num_codebook_vectors,
            "block_size": 512,
            "n_layer": 24,
            "n_head": 16,
            "n_embd": 1024,
        }
        self.transformer = GPT(**transformer_config)
        self.pkeep = args.pkeep

    @torch.no_grad()
    def encode_to_z(self, x):
        quant_z, indices, _ = self.vqgan.encode(x)
        indices = indices.view(quant_z.shape[0], -1)
        return quant_z, indices

    @torch.no_grad()
    def z_to_image(self, indices, p1=16, p2=16):
        idx_to_vectors = self.vqgan.codebook.embedding(indices).reshape(
            indices.shape[0], p1, p2, 256
        )
        idx_to_vectors = idx_to_vectors.permute(0, 3, 1, 2)
        image = self.vqgan.decoder(idx_to_vectors)
        return image.cpu().detach().numpy()[0].transpose(1, 2, 0)

    @staticmethod
    def load_vqgan(args):
        model = VQGAN(args)
        model.load_checkpoint(args.checkpoint_path)
        model = model.eval()
        return model

    def forward(self, x):
        quant_z, indices = self.encode_to_z(
            x
        )  # encode the image to latent vectors and indices

        sos_tokens = (
            torch.ones(x.shape[0], 1) * self.sos_token
        )  # create a tensor of start of sequence tokens
        sos_tokens = sos_tokens.long().to("cuda")  # move the tensor to the GPU

        mask = torch.bernoulli(
            self.pkeep * torch.ones(indices.shape, device=indices.device)
        )  # create a mask to randomly replace some indices with random indices
        mask = mask.round().to(dtype=torch.int64)  # round the mask to 0 or 1
        random_indices = torch.randint_like(
            indices, self.transformer.config.vocab_size
        )  # create random indices to replace the original indices
        new_indices = (
            mask * indices + (1 - mask) * random_indices
        )  # replace the original indices with random indices based on the mask (0 will replace the index with a random index, 1 will keep the original index)

        new_indices = torch.cat(
            (sos_tokens, new_indices), dim=1
        )  # append the start of sequence token to the indices

        target = indices

        logits = self.transformer(
            new_indices[:, :-1]
        )  # new_indices[:, :-1] is from the first token to the second to last token because the last token is the target

        return logits, target

    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[:, [-1]]] = -float("Inf")
        return out

    @torch.no_grad()
    def sample(self, x, c, steps, temperature=1.0, top_k=100):
        """
        Sample from the model.

        Args:
            x (torch.Tensor): The input tensor.
            c (torch.Tensor): The context tensor.
            steps (int): The number of steps to sample.
            temperature (float, optional): The temperature. Defaults to 1.0.
            top_k (int, optional): The top k. Defaults to 100.

        Returns:
            torch.Tensor: The sampled tensor.
        """
        self.transformer.eval()
        x = torch.cat((c, x), dim=1)
        for _ in range(steps):
            # logits definition is unnormalized log probabilities of the next token. For example, logits = [0.1, 0.2, 0.3, 0.4] means that the probability of the next token being 0 is 0.1, the probability of the next token being 1 is 0.2, and so on. The logits are divided by the temperature to make the distribution more or less random.
            logits = (
                self.transformer(x)[:, -1, :] / temperature
            )  # only take the last token in the prediction sequence and divide by the temperature (higher temperature means more randomness)
            logits = self.top_k_logits(
                logits, top_k
            )  # top k sampling to avoid repetition
            probs = F.softmax(logits, dim=-1)  # convert logits to probabilities
            ix = torch.multinomial(probs, num_samples=1)  # sample from the distribution
            x = torch.cat((x, ix), dim=1)  # append the sampled token to the sequence

        x = x[:, c.shape[1] :]  # remove the context tokens
        self.transformer.train()
        return x

    @torch.no_grad()
    def log_images(self, x):
        """
        Log the images. This function does the following:
        1. Encode the image to latent vectors and indices.
        2. Sample the middle indices of the image.
        3. Decode the indices to an image.
        4. Sample the first indices of the image.
        5. Sample the rest of the indices.
        6. Decode the indices to an image.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            dict: The dictionary containing the input, reconstructed, half sampled, and full sampled images.
        """
        log = dict()

        quant_z, indices = self.encode_to_z(
            x
        )  # encode the image to latent vectors and indices
        sos_tokens = (
            torch.ones(x.shape[0], 1) * self.sos_token
        )  # create a tensor of start of sequence tokens
        sos_tokens = sos_tokens.long().to("cuda")  # move the tensor to the GPU

        start_indices = indices[
            :, indices.shape[1] // 2
        ]  # get the middle indices of the image
        start_indices = self.sample(
            start_indices, sos_tokens, steps=indices.shape[1] - start_indices.shape[1]
        )  # sample the rest of the indices
        half_sample = self.z_to_image(start_indices)  # decode the indices to an image

        start_indices = indices[:, :0]  # get the first indices of the image
        sample_indices = self.sample(
            start_indices, sos_tokens, steps=indices.shape[1]
        )  # sample the rest of the indices
        full_sample = self.z_to_image(sample_indices)

        x_rec = self.z_to_image(indices)  # decode the indices to an image

        log["input"] = x
        log["rec"] = x_rec
        log["half_sample"] = half_sample
        log["full_sample"] = full_sample

        return log
