"""Seed blocks for every tars-ppo run (tars-test-rules sec. 4/5). Single source of truth,
imported by ppo.py, lifelong.py and stream.py. Dependency-free on purpose: no imports from
repo modules (ppo.py cannot import lifelong.py, which imports ppo.py).

Reserved stream blocks, each paired one-to-one with a fresh-agent block at +100:
  pilot / null_200     stream 500-509 -> fresh 600-609
  stage 1 (test)       stream 700-724 -> fresh 800-824   (Dan, 30 Oct 2026)
  stage 1 sweep/tuning stream 750-759 -> fresh 850-859   (decision, 30 Oct 2026; tuning only)
750-759 sits between 724 and 800 but overlaps no stream or fresh block (tested).
Dev stream seeds are 900-1899 with fresh seeds at +1000 (1900-2899); sizing/timing only.
"""

PILOT_STREAM_SEEDS = frozenset(range(500, 510))
STAGE1_STREAM_SEEDS = frozenset(range(700, 725))
SWEEP_STREAM_SEEDS = frozenset(range(750, 760))
FRESH_OFFSET = 100
SEED_BLOCKS = {  # name -> (stream seeds, fresh seeds)
    "pilot": (PILOT_STREAM_SEEDS, frozenset(s + FRESH_OFFSET for s in PILOT_STREAM_SEEDS)),
    "stage1": (STAGE1_STREAM_SEEDS, frozenset(s + FRESH_OFFSET for s in STAGE1_STREAM_SEEDS)),
    "sweep": (SWEEP_STREAM_SEEDS, frozenset(s + FRESH_OFFSET for s in SWEEP_STREAM_SEEDS)),
}
OFFICIAL_STREAM_SEEDS = frozenset().union(*(b[0] for b in SEED_BLOCKS.values()))
OFFICIAL_FRESH_SEEDS = frozenset().union(*(b[1] for b in SEED_BLOCKS.values()))
RESERVED_SEEDS = OFFICIAL_STREAM_SEEDS | OFFICIAL_FRESH_SEEDS
DEV_MIN_SEED = 900
DEV_MAX_SEED = 1900  # exclusive; dev fresh seeds are 1900-2899
DEV_FRESH_OFFSET = 1000
# Dev seeds registered in the demo-pool reference set (stream.py) at pilot length.
DEV_REFERENCE_SEEDS = tuple(range(900, 940))

# Shared layout (stage 1, decided 31 Oct 2026): the identity observation
# layout (8x8 grid, 3 channels -> 192 inputs) and the identity action permutation. With
# StreamConfig.shared_obs_layout / shared_action_perm on, every demo task and stream task 1
# use them; later obs / dynamics shifts permute away from them.
SHARED_OBS_LAYOUT = tuple(range(3 * 8 * 8))
SHARED_ACTION_PERM = (0, 1, 2, 3)
# sha256 of int32 bytes of SHARED_OBS_LAYOUT followed by SHARED_ACTION_PERM (tested)
SHARED_LAYOUT_HASH = "cacc9e17de197dfa294ad8dd267f2ff650804512c0975edcf7dde221a1ad737c"


def _span(block):
    return f"{min(block)}-{max(block)}"


# Derived from SEED_BLOCKS (tested): "stream 500-509/700-724/750-759, fresh 600-609/800-824/850-859"
BLOCKS_TEXT = ("stream " + "/".join(_span(b[0]) for b in SEED_BLOCKS.values())
               + ", fresh " + "/".join(_span(b[1]) for b in SEED_BLOCKS.values()))


def seed_block(stream_seed: int):
    """Name of the reserved stream block a seed belongs to, or None."""
    for name, (stream_seeds, _) in SEED_BLOCKS.items():
        if stream_seed in stream_seeds:
            return name
    return None


def fresh_seed(stream_seed: int) -> int:
    """Paired fresh-baseline seed: reserved stream blocks 500-509 / 700-724 / 750-759 -> +100
    (600-609 / 800-824 / 850-859), dev 9xx+ -> +1000."""
    if stream_seed in OFFICIAL_STREAM_SEEDS:
        return stream_seed + FRESH_OFFSET
    if stream_seed >= DEV_MIN_SEED:
        return stream_seed + DEV_FRESH_OFFSET
    raise ValueError(f"no fresh-seed mapping for stream seed {stream_seed}")
