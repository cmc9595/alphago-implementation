"""Expand a checkpoint with identity residual blocks; never overwrite a model."""
import argparse
import hashlib
from pathlib import Path

import torch

from alphago.models import GoNet, NetConfig, expand_blocks


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--init', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--blocks', type=int, default=10)
    ap.add_argument('--global-pool', action='store_true', help='add zero-initialized global-context paths')
    ap.add_argument('--seed', type=int, default=20261117)
    args = ap.parse_args()
    source, target = Path(args.init), Path(args.out)
    if target.exists():
        raise FileExistsError(target)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    ck = torch.load(source, map_location='cpu', weights_only=False)
    old = GoNet(NetConfig(**ck['config'])).eval()
    old.load_state_dict(ck['model'], strict=True)
    net = expand_blocks(old, args.blocks, global_pool=True if args.global_pool else None)
    # Check all heads, not just value. This is initialization, not strength testing.
    x = torch.randint(0, 2, (4, old.cfg.in_planes, old.cfg.board_size, old.cfg.board_size)).float()
    with torch.no_grad():
        for before, after in zip(old.forward_heads(x), net.forward_heads(x)):
            if before is not None:
                torch.testing.assert_close(before, after, rtol=1e-6, atol=1e-7)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('xb') as stream:
        torch.save({'model': net.state_dict(), 'config': vars(net.cfg),
                    'expansion': {'source': str(source), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                                  'seed': args.seed, 'method': 'identity residual blocks; zero-initialized global-context projections'}}, stream)
    print(f'Saved {target}; {old.cfg.blocks} -> {net.cfg.blocks} blocks; global_pool={net.cfg.global_pool}; output equivalence verified')


if __name__ == '__main__':
    main()
