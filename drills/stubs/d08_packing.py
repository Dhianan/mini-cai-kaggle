"""Drill 08 (45 min): sequence packing for training throughput.

Interview follow-ups: what is the cost of padding? Why do packed sequences need
position-id resets and a block-diagonal attention mask? What happens to a long
document at a row boundary?
"""

def pack_sequences(seqs: list[list[int]], seq_len: int, pad_id: int) -> tuple[list[list[int]], list[list[int]], list[list[int]]]:
    """Greedy packing in order. Sequences longer than seq_len are split into chunks.
Returns (rows, segment_ids, position_ids), each a list of rows of length seq_len.
  segment_ids: 1, 2, 3... per document within a row, 0 for padding
  position_ids: restart at 0 for each document segment, 0 for padding
The last row is padded with pad_id."""
    raise NotImplementedError()

def block_mask(segment_ids: list[int]) -> list[list[bool]]:
    """Causal, block-diagonal mask for one packed row: i may attend to j iff j <= i
and both tokens are in the same (non-padding) segment."""
    raise NotImplementedError()
