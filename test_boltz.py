import torch
from boltz import (
    filter_dict,
    PairformerModule,
    DiffusionTransformerModule,
)
#from boltz.model.modules.trunk import PairformerModule as PairformerModuleTorch

#from boltz.model.modules.diffusion import (
#    DiffusionTransformer as DiffusionTransformerTorch,
#)

torch.set_grad_enabled(False)
torch.manual_seed(0)
#ttnn.enable_program_cache(device)

#state_dict = torch.load(
#    "/home/ttuser/mbahnas/boltz/boltz/boltz1_conf.ckpt", map_location="cpu", mmap=False
#)["state_dict"]
state_dict = torch.load("boltz1_conf_dict.pth")

def mean_absolute_error(a, b):
    return torch.mean(torch.abs(a - b))


def test_pairformer():
    pairformer = PairformerModule(
        n_blocks=1,
        tri_att_head_dim=32,
        tri_att_n_heads=4,
        att_head_dim=24,
        att_n_heads=16,
    )
    #pairformer_torch = PairformerModuleTorch(
    #    token_s=384, token_z=128, num_blocks=1
    #).eval()
    pairformer_state_dict = filter_dict(state_dict, "pairformer_module")
    pairformer.load_state_dict(
        pairformer_state_dict,
        strict=False,
    )
    #pairformer_torch.load_state_dict(pairformer_state_dict, strict=False)
    s = torch.randn(1, 128, 384)
    z = torch.randn(1, 128, 128, 128)
    mask = torch.ones(1, 128)
    pair_mask = mask[:, :, None] * mask[:, None, :]
    s_tt, z_tt = pairformer(s, z, mask, pair_mask)
    #s_torch, z_torch = pairformer_torch(s, z, mask, pair_mask)
    #print(mean_absolute_error(s_tt, s_torch), mean_absolute_error(z_tt, z_torch))
    
'''
def test_token_transformer():
    token_transformer = DiffusionTransformerModule(
        n_layers=24,
        dim=768,
        n_heads=16,
    )
    token_transformer_torch = DiffusionTransformerTorch(
        depth=24, heads=16, dim=768, dim_single_cond=768, dim_pairwise=128
    ).eval()
    token_transformer_state_dict = filter_dict(
        state_dict, "structure_module.score_model.token_transformer"
    )
    token_transformer.load_state_dict(
        token_transformer_state_dict,
        strict=False,
    )
    token_transformer_torch.load_state_dict(token_transformer_state_dict, strict=False)
    a = torch.randn(1, 128, 768)
    s = torch.randn(1, 128, 768)
    z = torch.randn(1, 128, 128, 128)
    mask = torch.ones(1, 128)
    a_tt = token_transformer(
        a,
        s,
        z,
        mask,
    )
    a_torch = token_transformer_torch(
        a,
        s,
        z,
        mask,
    )
    print(
        mean_absolute_error(a_tt, a_torch),
    )

if __name__ == '__main__':
    test_pairformer()
'''

