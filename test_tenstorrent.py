import pytest, torch
from tenstorrent import (
    filter_dict,
    PairformerModule,
    DiffusionTransformerModule,
)
#from boltz.model.modules.trunk import PairformerModule as PairformerModuleTorch
#from boltz.model.modules.diffusion import (
#    DiffusionTransformer as DiffusionTransformerTorch,
#)

torch.set_grad_enabled(False)
torch.manual_seed(893)

state_dict = torch.load(
    "/home/yfan/.boltz/boltz1_conf.ckpt", map_location="cpu", mmap=True
)["state_dict"]


def median_relative_error(a, b):
    return ((a - b).abs() / b.abs()).median().item()


@pytest.mark.parametrize("seq_len", [100, 500, 1000])
def test_pairformer(seq_len):
    pairformer = PairformerModule(
        n_blocks=2,
        tri_att_head_dim=32,
        tri_att_n_heads=4,
        att_head_dim=24,
        att_n_heads=16,
    )
    #pairformer_torch = PairformerModuleTorch(
    #    token_s=384, token_z=128, num_blocks=2
    #).eval()
    pairformer_state_dict = filter_dict(state_dict, "pairformer_module")
    pairformer.load_state_dict(
        pairformer_state_dict,
        strict=False,
    )
    #pairformer_torch.load_state_dict(pairformer_state_dict, strict=False)
    s = 8 * torch.randn(1, seq_len, 384)
    z = 26 * torch.randn(1, seq_len, seq_len, 128)
    mask = torch.ones(1, seq_len)
    pair_mask = mask[:, :, None] * mask[:, None, :]
    s_tt, z_tt = pairformer(s, z, mask, pair_mask)
    #s_torch, z_torch = pairformer_torch(s, z, mask, pair_mask)
    #assert median_relative_error(s_tt, s_torch) < 1e-1, "s not accurate"
    #assert median_relative_error(z_tt, z_torch) < 1e-1, "z not accurate"


@pytest.mark.parametrize("seq_len", [100, 500, 1000])
def test_token_transformer(seq_len):
    token_transformer = DiffusionTransformerModule(
        n_layers=2,
        dim=768,
        n_heads=16,
    )
    #token_transformer_torch = DiffusionTransformerTorch(
    #    depth=2, heads=16, dim=768, dim_single_cond=768, dim_pairwise=128
    #).eval()
    token_transformer_state_dict = filter_dict(
        state_dict, "structure_module.score_model.token_transformer"
    )
    token_transformer.load_state_dict(
        token_transformer_state_dict,
        strict=False,
    )
    #token_transformer_torch.load_state_dict(token_transformer_state_dict, strict=False)
    a = 3 + 5 * torch.randn(1, seq_len, 768)
    s = -2 + 42 * torch.randn(1, seq_len, 768)
    z = 10 * torch.randn(1, seq_len, seq_len, 128)
    mask = torch.ones(1, seq_len)
    a_tt = token_transformer(
        a,
        s,
        z,
        mask,
    )
    #a_torch = token_transformer_torch(
    #    a,
    #    s,
    #    z,
    #    mask,
    #)
    #assert median_relative_error(a_tt, a_torch) < 1e-1, "a not accurate"
