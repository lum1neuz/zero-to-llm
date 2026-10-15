// Generates the PWA icons (PNG + SVG) without image dependencies: a sky-blue tile,
// a gold sun and a pale rolling steppe horizon. Run once; outputs are committed.
import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { deflateSync } from "node:zlib";

const out = fileURLToPath(new URL("../public/icons/", import.meta.url));
mkdirSync(out, { recursive: true });

const SKY = [0x1c, 0x5a, 0xb6], SUN = [0xf0, 0xc3, 0x6b], LAND = [0xd8, 0xe4, 0xf7], LAND2 = [0x9c, 0xbc, 0xea];
const hill = (x: number) => 0.68 + 0.05 * Math.sin(x * 5.2 + 0.6);
const hill2 = (x: number) => 0.78 + 0.035 * Math.sin(x * 7.5 + 2.1);

function color(u: number, v: number, maskable: boolean): number[] | null {
  const r = maskable ? 0 : 0.2; // corner radius as a fraction
  const inset = 0;
  const cx = Math.min(Math.max(u, r + inset), 1 - r - inset), cy = Math.min(Math.max(v, r + inset), 1 - r - inset);
  if (!maskable && Math.hypot(u - cx, v - cy) > r) return null;
  if (v > hill2(u)) return LAND2;
  if (v > hill(u)) return LAND;
  if (Math.hypot(u - 0.66, v - 0.4) < 0.15) return SUN;
  return SKY;
}

function png(size: number, maskable: boolean) {
  const raw = Buffer.alloc((size * 4 + 1) * size);
  const S = 4; // supersampling
  for (let y = 0; y < size; y++) {
    raw[y * (size * 4 + 1)] = 0;
    for (let x = 0; x < size; x++) {
      let r = 0, g = 0, b = 0, a = 0;
      for (let sy = 0; sy < S; sy++) for (let sx = 0; sx < S; sx++) {
        const c = color((x + (sx + 0.5) / S) / size, (y + (sy + 0.5) / S) / size, maskable);
        if (c) { r += c[0]; g += c[1]; b += c[2]; a++; }
      }
      const o = y * (size * 4 + 1) + 1 + x * 4;
      raw[o] = a ? r / a : 0; raw[o + 1] = a ? g / a : 0; raw[o + 2] = a ? b / a : 0; raw[o + 3] = (a / (S * S)) * 255;
    }
  }
  const chunk = (type: string, data: Buffer) => {
    const len = Buffer.alloc(4); len.writeUInt32BE(data.length);
    const td = Buffer.concat([Buffer.from(type), data]);
    const crc = Buffer.alloc(4); crc.writeUInt32BE(crc32(td) >>> 0);
    return Buffer.concat([len, td, crc]);
  };
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(size, 0); ihdr.writeUInt32BE(size, 4); ihdr[8] = 8; ihdr[9] = 6;
  return Buffer.concat([Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]), chunk("IHDR", ihdr), chunk("IDAT", deflateSync(raw)), chunk("IEND", Buffer.alloc(0))]);
}

function crc32(buf: Buffer) {
  let c = ~0;
  for (const byte of buf) { c ^= byte; for (let k = 0; k < 8; k++) c = c & 1 ? (c >>> 1) ^ 0xedb88320 : c >>> 1; }
  return ~c;
}

writeFileSync(join(out, "icon-192.png"), png(192, false));
writeFileSync(join(out, "icon-512.png"), png(512, false));
writeFileSync(join(out, "maskable-512.png"), png(512, true));
const path = (f: (x: number) => number) => {
  let d = "M0 100";
  for (let i = 0; i <= 50; i++) d += ` L${i * 2} ${(f(i / 50) * 100).toFixed(2)}`;
  return d + " L100 100 Z";
};
writeFileSync(join(out, "icon.svg"), `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><clipPath id="r"><rect width="100" height="100" rx="20"/></clipPath><g clip-path="url(#r)"><rect width="100" height="100" fill="#1C5AB6"/><circle cx="66" cy="40" r="15" fill="#F0C36B"/><path d="${path(hill)}" fill="#D8E4F7"/><path d="${path(hill2)}" fill="#9CBCEA"/></g></svg>\n`);
console.log("✓ icons in public/icons/");
