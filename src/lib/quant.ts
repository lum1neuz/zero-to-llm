// Quantization math shared by the Learn slider card and the content pipeline
// (the pipeline recomputes slider answers so content can't drift from the visual).

export const roundHalfEven = (x: number) => (Math.abs(x % 1) === 0.5 ? 2 * Math.round(x / 2) : Math.round(x));

export function quantStats(bits: number, weights: number[]) {
  const qmax = 2 ** (bits - 1) - 1;
  const amax = Math.max(...weights.map(Math.abs));
  const scale = amax / qmax;
  const q = weights.map((w) => Math.max(-qmax, Math.min(qmax, roundHalfEven(w / scale))) * scale);
  const err = weights.reduce((a, w, i) => a + Math.abs(w - q[i]), 0) / weights.length;
  return { qmax, scale, q, err };
}

export function fewestBitsUnder(limit: number, weights: number[], minBits: number, maxBits: number) {
  for (let b = minBits; b <= maxBits; b++) if (quantStats(b, weights).err < limit) return b;
  return null;
}
