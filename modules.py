from os import path
import json
from argparse import Namespace
from typing import Tuple, Union
import numpy as np
import torch
from torch import nn
import torch.functional as F


from esm import pretrained
from esm.inverse_folding.gvp_transformer import GVPTransformerModel
from esm.inverse_folding.gvp_transformer_encoder import GVPTransformerEncoder
from esm.model.esm2 import ESM2
from esm.data import Alphabet

SAVE_DIR = path.join(path.dirname(path.abspath(__file__)), "config")
DEFAULT_JOINT_EMBEDDING_DIM = 512
DEFAULT_POOL_LAST_N_LAYERS = 4
DEFAULT_TOKEN_POOL_STRATEGY = "bos"
DEFAULT_LAYER_POOL_STRAEGY = "weighted"


def generate_alphabet_args(model="esm2"):
    """
    Generate args dict for ESM2 Alphabet from Hub.

    Args:
        model (str, optional): Model name: esm2/esm_if. Defaults to "esm2".
        save (bool, optional): Save args to json file. Defaults to True.
    """
    assert model in ["esm2", "esm_if"], "Model Name must be esm2 or esm_if"
    if model == "esm2":
        _, alphabet = pretrained.esm2_t6_8M_UR50D()
    else:
        _, alphabet = pretrained.esm_if1_gvp4_t16_142M_UR50()
    args = {
        "standard_toks": alphabet.standard_toks,
        "prepend_toks": alphabet.prepend_toks,
        "append_toks": alphabet.append_toks,
        "prepend_bos": alphabet.prepend_bos,
        "append_eos": alphabet.append_eos,
        "use_msa": alphabet.use_msa,
    }

    return args


def generate_esm_if_args():
    model, _ = pretrained.esm_if1_gvp4_t16_142M_UR50()
    model_args = vars(model.args)
    return model_args


def load_esm2(model_config: dict) -> Tuple[ESM2, Alphabet]:
    """Load ESM2 model using saved args

    Args:
        model_config (dict): Model Config
        joint_embedding_dim (int): Joint Embedding Dimension

    Returns:
        Tuple[ESM2, Alphabet]: ESM2 model and Alphabet
    """
    # Load ESM-2 Base model args, alphabet args, and the given model config

    with open(path.join(SAVE_DIR, "default_alphabet_args.json"), "r") as f:
        alphabet_args = json.load(f)["esm2"]

    alphabet = Alphabet(**alphabet_args)
    model_config["alphabet"] = alphabet
    esm2 = _ESM2(args=Namespace(**model_config))

    return esm2, alphabet


def load_esm_if(model_config: dict) -> Tuple[GVPTransformerModel, Alphabet]:
    """Load ESM-IF model using saved args

    Args:
        model_config (dict): ESM-IF Args

    Returns:
        Tuple[GVPTransformerModel, Alphabet]: ESM-IF model and Alphabet
    """

    with open(path.join(SAVE_DIR, "default_alphabet_args.json"), "r") as f:
        alphabet_args = json.load(f)["esm_if"]

    with open(path.join(SAVE_DIR, "default_esm_if_args.json"), "r") as f:
        esm_if_args = json.load(f)

    # Update model args with the given model type args
    for arg, val in model_config.items():
        esm_if_args[arg] = val

    alphabet = Alphabet(**alphabet_args)
    esm_if_args["alphabet"] = alphabet

    esm_if = _ESM_IF(Namespace(**esm_if_args))

    return esm_if, alphabet


# TODO: revisit this.
def visualize_logits(self, layers: list) -> None:
    """Visualize Logits

    Args:
        layers (list): List of tuples of name and tensor containing logits
            eg: [("esm2_outs", esm2_outs), ("esm_if_outs", esm_if_outs)]
    """
    import matplotlib.pyplot as plt
    import torch

    # visualize histograms
    plt.figure(figsize=(20, 4))  # width and height of the plot
    legends = []
    for i, layer in enumerate(layers):  # note: exclude the output layer
        #   if isinstance(layer, Tanh):
        t = layer[1]
        print(
            "layer {%d (%10s)}: mean %+.2f, std %.2f"
            % (i, layer[0], t.mean(), t.std())
        )
        hy, hx = torch.histogram(t, density=True)
        plt.plot(hx[:-1].detach(), hy.detach())
        legends.append(f"layer {i} ({layer[0]}")
    plt.legend(legends)
    plt.title("activation distribution")


class _ESM2(ESM2):
    """Modified ESM2 Class with pooling and joint embedding projections"""

    def __init__(self, args: Namespace):
        """Initialize Modified ESM2 Model inherited from esm.model.esm2.ESM2

        Args:
            args (Namespace): Args containing model params / config
                - num_layers (int): Number of Layers
                - embed_dim (int): Embedding Dimension
                - attention_heads (int): Number of Attention Heads
                - token_dropout (bool): Whether to apply token dropout
                - layer_pool_strategy (str): Layer Pooling Strategy (weighted/attention)
                - pool_last_n_layers (int): Number of Hidden Layers to pool
                - token_pool_strategy (str): Token Pooling Strategy (mean/bos)
                - layer_pool_weights (torch.tensor): Layer weights for weighted layer pooling
                - joint_embedding_dim (int): Joint Embedding Dimension
                - alphabet (Union[Alphabet, str], optional): ESM2 Alphabet. Defaults to "ESM-1b".
        """

        super().__init__(
            args.num_layers,
            args.embed_dim,
            args.attention_heads,
            args.alphabet,
            args.token_dropout,
        )

        # Remove layers not used in loss calculation
        del self.lm_head
        del self.contact_head
        del self.emb_layer_norm_after

        # pooling
        assert (
            args.pool_last_n_layers <= args.num_layers
        ), "pool_last_n_layers must be less than or equal to than num_layers"

        self.pool_last_n_layers = args.pool_last_n_layers
        self.token_pool_strategy = args.token_pool_strategy
        self.layer_pool_strategy = args.layer_pool_strategy

        # Initialize hidden layer pooler
        if self.layer_pool_strategy is not None:
            self.layer_pooler = get_layer_pooler(
                self.layer_pool_strategy,
                self.pool_last_n_layers,
                args.layer_pool_weights,
            )

        self.after_pool_ln = nn.LayerNorm(args.embed_dim)

        # Project Sequence Embedding to Joint Embedding Space
        self.joint_embedding_projection = nn.Linear(
            args.embed_dim, args.joint_embedding_dim
        )
        self.norm_embedding = args.norm_embedding

    def forward(
        self,
        tokens: torch.tensor,
        need_head_weights=False,
    ):
        assert tokens.ndim == 2
        padding_mask = tokens.eq(self.padding_idx)

        x = self.embed_scale * self.embed_tokens(tokens)

        # Dropout
        if self.token_dropout:
            x.masked_fill_((tokens == self.mask_idx).unsqueeze(-1), 0.0)
            # x: B x T x C
            mask_ratio_train = 0.15 * 0.8
            src_lengths = (~padding_mask).sum(-1)
            mask_ratio_observed = (tokens == self.mask_idx).sum(-1).to(
                x.dtype
            ) / src_lengths
            x = (
                x
                * (1 - mask_ratio_train)
                / (1 - mask_ratio_observed)[:, None, None]
            )

        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

        # (B, T, E) => (T, B, E)
        x = x.transpose(0, 1)

        self_attn_padding_mask = (
            None if not padding_mask.any() else padding_mask
        )

        num_layers = len(self.layers)
        hidden_representations = []
        for layer_idx, layer in enumerate(self.layers):
            x, _ = layer(
                x,
                self_attn_padding_mask=self_attn_padding_mask,
                need_head_weights=need_head_weights,
            )
            if num_layers - layer_idx <= self.pool_last_n_layers:
                hidden_representations.append(x.transpose(0, 1))

        # Pool Tokens per layer. H * B * T * E -> H * B * E
        batch_padding_lens = (~padding_mask).sum(-1)
        pooled_representations = pool_tokens_per_layer(
            torch.stack(hidden_representations),
            batch_padding_lens,
            self.token_pool_strategy,
        )

        # Layer Pooling. H * B * E -> B * E
        if self.layer_pool_strategy is not None:
            pooled_representations = self.layer_pooler(pooled_representations)
        else:
            # Get the representations from the final layer
            pooled_representations = pooled_representations[-1]

        # Layer Norm after Pooling
        pooled_representations = self.after_pool_ln(pooled_representations)

        # Linear Projection to joint embedding dim & Another Layer Norm
        seq_projection = self.joint_embedding_projection(pooled_representations)
        # Shape: Batch size * Joint Embedding Dim
        if self.norm_embedding:
            seq_projection = normalize_embeddings(seq_projection)

        return seq_projection


class _ESM_IF(GVPTransformerEncoder):
    def __init__(self, args: Namespace):
        """Initialize ESM_IF Model inherited from GVPTransformerEncoder

        Args:
            args (argparse.Namespace): GVPTransformerEncoder Args. Other args:
                - layer_pool_strategy (str): Layer Pooling Strategy (weighted/attention)
                - pool_last_n_layers (int): Number of Hidden Layers to pool
                - token_pool_strategy (str): Token Pooling Strategy (mean/bos)
                - joint_embedding_dim (int): Joint Embedding Dimension
                - layer_pool_weights (torch.tensor): Layer weights for weighted layer pooling
            alphabet (Alphabet): ESM-IF Alphabet
        """
        encoder_embed_tokens = self.build_embedding(
            args.alphabet, args.encoder_embed_dim
        )
        super().__init__(args, args.alphabet, encoder_embed_tokens)

        # Not used in loss calculation (DDP error)
        del self.layer_norm

        self.num_layers = args.encoder_layers

        # pooling
        assert (
            args.pool_last_n_layers <= self.num_layers
        ), "pool_last_n_layers must be less than or equal to than num_layers"

        # Pool last n layers
        self.layer_pool_strategy = args.layer_pool_strategy
        self.pool_last_n_layers = args.pool_last_n_layers
        self.token_pool_strategy = args.token_pool_strategy

        # Initialize hidden layer pooler
        if self.layer_pool_strategy is not None:
            self.layer_pooler = get_layer_pooler(
                self.layer_pool_strategy,
                self.pool_last_n_layers,
                args.layer_pool_weights,
            )

        self.after_pool_ln = nn.LayerNorm(args.encoder_embed_dim)

        # Project Structural Embedding to Joint Embedding Space
        self.joint_embedding_projection = nn.Linear(
            args.encoder_embed_dim, args.joint_embedding_dim
        )
        self.norm_embedding = args.norm_embedding

    @classmethod
    def build_embedding(cls, dictionary, embed_dim):
        num_embeddings = len(dictionary)
        padding_idx = dictionary.padding_idx
        emb = nn.Embedding(num_embeddings, embed_dim, padding_idx)
        nn.init.normal_(emb.weight, mean=0, std=embed_dim**-0.5)
        nn.init.constant_(emb.weight[padding_idx], 0)
        return emb

    def forward(
        self,
        coords,
        encoder_padding_mask,
        confidence,
    ):
        x, _ = self.forward_embedding(coords, encoder_padding_mask, confidence)
        # account for padding while computing the representation
        x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x))

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # encoder layers
        hidden_representations = []
        for layer_idx, layer in enumerate(self.layers):
            x = layer(x, encoder_padding_mask=encoder_padding_mask)

            if self.num_layers - layer_idx <= self.pool_last_n_layers:
                hidden_representations.append(x.transpose(0, 1))

        # Pool Tokens per layer. H * B * T * E -> H * B * E
        batch_padding_lens = (~encoder_padding_mask).sum(-1)
        pooled_representations = pool_tokens_per_layer(
            torch.stack(hidden_representations),
            batch_padding_lens,
            self.token_pool_strategy,
        )

        # Layer Pooling. H * B * E -> B * E
        if self.layer_pool_strategy is not None:
            pooled_representations = self.layer_pooler(pooled_representations)
        else:
            # Get the representations from the final layer
            pooled_representations = pooled_representations[-1]

        # Layer Norm after Pooling
        pooled_representations = self.after_pool_ln(pooled_representations)

        # Linear Projection to joint embedding dim & Another Layer Norm
        structure_embedding = self.joint_embedding_projection(
            pooled_representations
        )
        # Shape: Batch size * Joint Embedding Dim
        if self.norm_embedding:
            structure_embedding = normalize_embeddings(structure_embedding)
        return structure_embedding


class WeightedLayerPooling(nn.Module):
    """Implementation of Weighted Layer Pooling Taken from
    https://www.kaggle.com/code/rhtsingh/utilizing-transformer-representations-efficiently

    """

    def __init__(
        self,
        pool_last_n_layers: int,
        layer_weights=None,
    ):
        """Init Params

        Args:
            pool_last_n_layers (int): Number of Hidden Layers in the Model
            layer_start (int): Take the last n layers for weight pooling
            layer_weights (_type_, torch.tensor): Layer weights. Defaults to None.
            pool_strategy (str): Pooling Strategy (mean/bos/eos).
                mean: take a mean of all token embeddings in a batch.
                bos: take the first token embedding in a batch.
                eos: take the last token embedding in a batch.
        """
        super().__init__()
        self.layer_weights = nn.Parameter(
            torch.tensor([1.0] * (pool_last_n_layers))
        )

    def forward(
        self,
        pooled_encoder_states: torch.tensor,
    ) -> torch.tensor:
        """Forward Function for Weighted Layer Pooling

        Args:
            pooled_encoder_states (torch.tensor): All hidden states pooled per layer of size
                Num Hidden Layers * Batch Size * Emb Size. Should be called after per_layer pooling

        Returns:
            torch.tensor: weighted average of all hidden states
        """
        weight_factor = (
            self.layer_weights.unsqueeze(-1)
            .unsqueeze(-1)
            .expand(pooled_encoder_states.size())
        )
        weighted_average = (weight_factor * pooled_encoder_states).sum(
            dim=0
        ) / self.layer_weights.sum()
        return weighted_average


class AttentionPooling(nn.Module):
    def __init__(self, num_layers: int, hidden_size: int, out_dim: int):
        super(AttentionPooling, self).__init__()
        self.pool_last_n_layers = num_layers
        self.hidden_size = hidden_size
        self.out_dim = out_dim
        self.dropout = nn.Dropout(0.1)

        q_t = np.random.normal(loc=0.0, scale=0.1, size=(1, self.hidden_size))
        self.q = nn.Parameter(torch.from_numpy(q_t)).float()
        w_ht = np.random.normal(
            loc=0.0, scale=0.1, size=(self.hidden_size, self.out_dim)
        )
        self.w_h = nn.Parameter(torch.from_numpy(w_ht)).float()

    def forward(self, all_hidden_states):
        hidden_states = torch.stack(
            [
                all_hidden_states[layer_i][:, 0].squeeze()
                for layer_i in range(0, self.pool_last_n_layers)
            ],
            dim=-1,
        )
        hidden_states = hidden_states.view(
            -1, self.pool_last_n_layers, self.hidden_size
        )
        out = self.attention(hidden_states)
        out = self.dropout(out)
        return out

    def attention(self, h):
        v = torch.matmul(self.q, h.transpose(-2, -1)).squeeze(1)
        v = F.softmax(v, -1)
        v_temp = torch.matmul(v.unsqueeze(1), h).transpose(-2, -1)
        v = torch.matmul(self.w_h.transpose(1, 0), v_temp).squeeze(2)
        return v


def get_layer_pooler(
    layer_pool_strategy: str,
    pool_last_n_layers: int,
    layer_weights: torch.tensor = None,
) -> nn.Module:
    """Get Pooler

    Args:
        layer_pool_strategy (str): Pooler Type (weighted/attention)
        num_hidden_layers (int): Number of Hidden Layers to pool
        pool_strategy (str): Pooling Strategy (mean/bos).
        layer_weights (None, torch.tensor): Layer weights. Defaults to None.

    Returns:
        nn.Module: Respective Pooler Module
    """
    assert layer_pool_strategy in [
        "weighted",
        "attention",
    ], "layer_pool_strategy must be weighted/attention"
    if layer_pool_strategy == "weighted":
        return WeightedLayerPooling(pool_last_n_layers, layer_weights)
    else:
        return AttentionPooling(pool_last_n_layers, 1024, 1024)


def return_mean_of_token_embeddings(
    encoder_states: torch.tensor, batch_padding_lens: torch.tensor
) -> torch.tensor:
    """
    Take mean of token embeddings from encoder states.
    Collapses the sequence dimension.


    Args:
        encoder_states (torch.tensor): Encoder States of shape Hidden Layers x Batch Size x Seq Len x Emb Size.
        batch_padding_lens (torch.tensor): Batch Padding lengths of length: Batch Size.

    Returns:
        torch.tensor: Output of shape (num_hidden_layers, batch_size, emb_size)
    """
    assert (
        encoder_states.ndim == 4
    ), "encoder_states must be of shape (num_hidden_layers, batch_size, seq_len, emb_size)"
    H, B, T, E = encoder_states.shape
    # assert (
    #     max(batch_padding_lens) == T
    # ), f"Max Padding Len: {batch_padding_lens} must be equal to T: {T}; H, B, T, E: {H, B, T, E}"
    pooled_encoder_states = torch.zeros(H, B, E, device=encoder_states.device)
    for i, tokens_len in enumerate(batch_padding_lens):
        if tokens_len <= 2:
            pooled_encoder_states[:, i] = pooled_encoder_states[
                :, i
            ] = encoder_states[:, i, 1]
        else:
            pooled_encoder_states[:, i] = encoder_states[
                :, i, 1 : tokens_len - 1
            ].mean(dim=1)
    return pooled_encoder_states


def return_bos_token_embedding(encoder_states: torch.tensor) -> torch.tensor:
    """Return bos token (beginning of sentence) token embedding

    Args:
        encoder_states (torch.tensor): Encoder States of shape Hidden Layers x Batch Size x Seq Len x Emb Size.

    Returns:
        torch.tensor: EOS tokens for all hidden states of shape Hidden Layers x Batch Size x Emb Size.
    """
    assert (
        len(encoder_states.shape) == 4
    ), "encoder_states must be of shape (num_hidden_layers, batch_size, seq_len, emb_size)"
    return encoder_states[:, :, 0, :]


def pool_tokens_per_layer(
    encoder_hidden_states: torch.tensor,
    batch_padding_lens: torch.tensor,
    pool_strategy: str,
) -> torch.tensor:
    """Pool Tokens per Layer from Encoder States

    Args:
        encoder_hidden_states (torch.tensor): Encoder States of Shape:
            Num Hidden Layers * Batch Size * Seq Len * Emb Size
        batch_padding_lens (torch.tensor | None): Padding Lens of shape Batch Size.
            Required for mean pooling
        pool_strategy (str): Pooling Strategy (mean/bos)

    Returns:
        torch.tensor: Pooled Tokens per Layer of shape Num Hidden Layers * Batch Size * Emb Size
    """
    if pool_strategy == "mean":
        assert (
            batch_padding_lens is not None
        ), "batch_padding_lens should not be None for mean pooling"
        pooled_hidden_states = return_mean_of_token_embeddings(
            encoder_hidden_states, batch_padding_lens
        )
    # bos
    elif pool_strategy == "bos":
        pooled_hidden_states = return_bos_token_embedding(encoder_hidden_states)
    else:
        raise ValueError(
            f"pool_strategy must be either mean/bos, not {pool_strategy}"
        )
    return pooled_hidden_states


def normalize_embeddings(embeddings: torch.tensor) -> torch.tensor:
    """Normalize Embeddings
        Taken from https://github.com/openai/CLIP/blob/a9b1bf5920416aaeaec965c25dd9e8f98c864f16/clip/model.py#L362

    Args:
        embeddings (torch.tensor): Embedding. (B, E)

    Returns:
        torch.tensor: Normalized Embeddings. (B, E)
    """
    return embeddings / embeddings.norm(dim=1, keepdim=True)


class VICReg(nn.Module):
    def __init__(self, args: dict):
        super().__init__()

        self.lamb = args["lambda"]
        self.mu = args["mu"]
        self.nu = args["nu"]
        num_layers = args["num_layers"]
        exp_dim = args["exp_dim"]
        self.joint_embedding_dim = args["joint_embedding_dim"]

        self.exp_net = [
            nn.Linear(self.joint_embedding_dim, exp_dim),
            nn.BatchNorm1d(exp_dim),
            nn.ReLU(),
        ]
        for _ in range(num_layers - 2):
            self.exp_net.extend(
                [
                    nn.Linear(exp_dim, exp_dim),
                    nn.BatchNorm1d(exp_dim),
                    nn.ReLU(),
                ]
            )
        self.exp_net.append(nn.Linear(exp_dim, exp_dim))
        self.exp_net = nn.Sequential(*self.exp_net)

    def forward(
        self, esm2_out: torch.Tensor, esm_if_out: torch.Tensor
    ) -> torch.Tensor:
        """Calculates the VICReg Loss between the ESM2 & ESM-IF Embeddings

        Args:
            esm2_out (torch.Tensor): ESM2 Embeddings
            esm_if_out (torch.Tensor): ESM-IF Embeddings

        Returns:
            torch.Tensor: VICReg Loss

        """

        esm2_out_exp = self.exp_net(esm2_out)
        esm_if_out_exp = self.exp_net(esm_if_out)

        B, exp_dim = esm2_out_exp.shape

        l2_loss_fn = nn.MSELoss()
        reg_loss = l2_loss_fn(esm2_out_exp, esm_if_out_exp) / B

        std_esm2 = torch.sqrt(torch.var(esm2_out_exp, dim=0) + 1e-4)
        std_esm_if = torch.sqrt(torch.var(esm_if_out_exp, dim=0) + 1e-4)
        relu = nn.ReLU()
        std_loss = torch.mean(relu(1 - std_esm2)) + torch.mean(
            relu(1 - std_esm_if)
        )  # 1 being the value of the hyper param gamma in the paper

        mean_esm2 = esm2_out_exp - torch.mean(esm2_out_exp, dim=0)
        mean_esm_if = esm_if_out_exp - torch.mean(esm_if_out_exp, dim=0)
        cov_esm2 = (mean_esm2.T @ mean_esm2) / (
            B - 1
        )  # cov is across the exp_dim --> exp_dim x exp_dim
        cov_esm_if = (mean_esm_if.T @ mean_esm_if) / (B - 1)
        cov_loss = (
            cov_esm2.pow(2).sum() - torch.trace(cov_esm2.pow(2))
        ) / exp_dim + (
            cov_esm_if.pow(2).sum() - torch.trace(cov_esm_if.pow(2))
        ) / exp_dim

        vicreg_loss = (
            self.lamb * reg_loss + self.mu * std_loss + self.nu * cov_loss
        )
        return vicreg_loss, {
            "expanded_esm2_out": esm2_out_exp,
            "expanded_esm_if_out": esm_if_out_exp,
        }
