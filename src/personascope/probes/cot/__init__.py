"""PMP probes — CoT (chain-of-thought) channel.

Status: **deferred from v1.** The probes here (`cot_content`,
`cot_faithfulness`) require a CoT-capture pipeline + alignment-sensitive
question batteries that aren't built. They remain in the repo as
scaffolding for v2 when:

  - the question battery for `cot_faithfulness` (MacDiarmid 3-pattern)
    is curated, and
  - the CoT capture step is added to the main runner so
    `cot_content` can post-hoc classify existing reasoning text.

Until then these probes are **orphan** — not in any tier, not wired
into `run_full_battery`. They can be invoked directly from caller code
for ad-hoc CoT analysis if you bring your own batteries.
"""
