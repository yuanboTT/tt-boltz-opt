import pytest, torch
import os

TT_BOLTZ_OPT = os.environ.get('TT_BOLTZ_OPT', '')

if TT_BOLTZ_OPT.lower() == 'use_bf16_in_l1':
    from tenstorrent_bf16_l1 import (
        filter_dict,
        PairformerModule,
        DiffusionTransformerModule,
    )
    print('Using BF16 in L1 ...')
elif TT_BOLTZ_OPT.lower() == 'use_bf16':
    from tenstorrent_bf16 import (
        filter_dict,
        PairformerModule,
        DiffusionTransformerModule,
    )
    print('Using BF16 ...')
else:
    from tenstorrent_fp32 import (
        filter_dict,
        PairformerModule,
        DiffusionTransformerModule,
    )
    print('Using FP32 ...')

import time
from tests.ttnn.utils_for_testing import assert_with_pcc

ENABLE_PCC = True

torch.set_grad_enabled(False)
torch.manual_seed(893)

state_dict = torch.load(
    "/home/yfan/.boltz/boltz1_conf.ckpt", map_location="cpu", mmap=True
)["state_dict"]


def median_relative_error(a, b):
    return ((a - b).abs() / b.abs()).median().item()

@pytest.mark.parametrize("seq_len", [100, 128, 256, 500, 1000])
def test_pairformer(seq_len):
    pairformer = PairformerModule(
        n_blocks=2,
        tri_att_head_dim=32,
        tri_att_n_heads=4,
        att_head_dim=24,
        att_n_heads=16,
    )
    pairformer_state_dict = filter_dict(state_dict, "pairformer_module")
    pairformer.load_state_dict(
        pairformer_state_dict,
        strict=False,
    )

    s = torch.load(f'test_data/pairformer_s_input{seq_len}.pt')
    z = torch.load(f'test_data/pairformer_z_input{seq_len}.pt')
    mask = torch.ones(1, seq_len)
    pair_mask = mask[:, :, None] * mask[:, None, :]

    if ENABLE_PCC:
        s_tt, z_tt = pairformer(s, z, mask, pair_mask)

    start = time.time()
    s_tt, z_tt = pairformer(s, z, mask, pair_mask)
    end = time.time()
    print(f'$$$YF: pairformer time: {end - start:.4f} seconds')

    if ENABLE_PCC:
        s_tt_correct = torch.load(f'test_data/pairformer_s_sl{seq_len}_tt.pt')
        z_tt_correct = torch.load(f'test_data/pairformer_z_sl{seq_len}_tt.pt')
        s_torch = torch.load(f'test_data/pairformer_s_sl{seq_len}_torch.pt')
        z_torch = torch.load(f'test_data/pairformer_z_sl{seq_len}_torch.pt')

        assert_with_pcc(s_tt_correct,   s_tt, pcc=0.9)
        assert_with_pcc(z_tt_correct,   z_tt, pcc=0.9)
        assert_with_pcc(s_torch,        s_tt, pcc=0.9)
        assert_with_pcc(z_torch,        z_tt, pcc=0.9)

@pytest.mark.parametrize("seq_len", [100, 128, 256, 500, 1000])
def test_token_transformer(seq_len):
    token_transformer = DiffusionTransformerModule(
        n_layers=2,
        dim=768,
        n_heads=16,
    )
    token_transformer_state_dict = filter_dict(
        state_dict, "structure_module.score_model.token_transformer"
    )
    token_transformer.load_state_dict(
        token_transformer_state_dict,
        strict=False,
    )
    a = torch.load(f'test_data/token_transformer_a_input{seq_len}.pt')
    s = torch.load(f'test_data/token_transformer_s_input{seq_len}.pt')
    z = torch.load(f'test_data/token_transformer_z_input{seq_len}.pt')
    mask = torch.ones(1, seq_len)

    if ENABLE_PCC:
        a_tt = token_transformer(
            a,
            s,
            z,
            mask,
        )

    start = time.time()
    a_tt = token_transformer(
        a,
        s,
        z,
        mask,
    )
    end = time.time()
    print(f'$$$YF: token_transformer time: {end - start:.4f} seconds')

    if ENABLE_PCC:
        a_tt_correct = torch.load(f'test_data/token_transformer_a_sl{seq_len}_tt.pt')
        a_torch = torch.load(f'test_data/token_transformer_a_sl{seq_len}_torch.pt')
        assert_with_pcc(a_tt_correct,   a_torch,    pcc=0.9)
        assert_with_pcc(a_torch,        a_tt,       pcc=0.9)
