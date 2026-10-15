import { quantStats } from "../../lib/quant.ts";

/** Original weights on top, where each lands on the int grid below. */
export function QuantGrid({ bits, weights }: { bits: number; weights: number[] }) {
  const { qmax, scale, q } = quantStats(bits, weights);
  const span = Math.max(...weights.map(Math.abs)) * 1.1;
  const x = (v: number) => 16 + ((v + span) / (2 * span)) * 288;
  const levels = [];
  for (let k = -qmax; k <= qmax; k++) levels.push(k * scale);
  return (
    <svg className="qviz" viewBox="0 0 320 112" role="img" aria-label={`int${bits} grid with ${2 * qmax + 1} levels`}>
      <text x="0" y="12">fp32</text>
      <text x="0" y="108">int{bits} grid ({2 * qmax + 1} levels)</text>
      <line className="ax" x1="16" x2="304" y1="28" y2="28" />
      <line className="ax" x1="16" x2="304" y1="80" y2="80" />
      {levels.map((v, k) => <line key={k} className="lv" x1={x(v)} x2={x(v)} y1="72" y2="88" />)}
      {weights.map((w, i) => <line key={`c${i}`} className="cn" x1={x(w)} y1="28" x2={x(q[i])} y2="80" />)}
      {weights.map((w, i) => <circle key={`o${i}`} className="o" cx={x(w)} cy="28" r="3.6" />)}
      {q.map((v, i) => <rect key={`q${i}`} className="q" x={x(v) - 3.5} y="76.5" width="7" height="7" transform={`rotate(45 ${x(v)} 80)`} />)}
    </svg>
  );
}
