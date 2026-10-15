import type { ReactNode } from "react";

const svg = (body: ReactNode, size = 16, view = "0 0 16 16") => (
  <svg viewBox={view} width={size} height={size} aria-hidden="true">{body}</svg>
);
export const I = {
  flame: svg(<path d="M8 1c.6 2.6 4 4.3 4 8.2A4 4 0 0 1 8 15a4 4 0 0 1-4-4c0-1.7.8-2.9 1.8-3.6.1 1.4.9 2.3 1.7 2.3C7 7 7.1 3.8 8 1z" fill="currentColor" />, 14),
  shield: svg(<path d="M8 1.5 13 3.5v4c0 3.2-2.2 5.6-5 7-2.8-1.4-5-3.8-5-7v-4z" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinejoin="round" />, 13),
  check: svg(<path d="M3 8.5l3 3 7-7" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" />, 18),
  cross: svg(<path d="M4 4l8 8M12 4l-8 8" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" />, 18),
  bulb: svg(<path d="M8 1.5a4.5 4.5 0 0 0-2.6 8.2c.4.3.6.7.6 1.1V12h4v-1.2c0-.4.2-.8.6-1.1A4.5 4.5 0 0 0 8 1.5zM6 13.6h4M6.8 15h2.4" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />),
  lock: svg(<><rect x="3" y="7" width="10" height="7" rx="1.5" fill="currentColor" /><path d="M5 7V5a3 3 0 0 1 6 0v2" fill="none" stroke="currentColor" strokeWidth="1.6" /></>, 18),
  play: svg(<path d="M4.5 2.8v10.4l8.6-5.2z" fill="currentColor" />),
  pause: svg(<><rect x="3.5" y="2.5" width="3" height="11" rx="1" fill="currentColor" /><rect x="9.5" y="2.5" width="3" height="11" rx="1" fill="currentColor" /></>),
  path: svg(<><circle cx="7" cy="5" r="2.5" fill="none" stroke="currentColor" strokeWidth="1.8" /><circle cx="17" cy="12" r="2.5" fill="none" stroke="currentColor" strokeWidth="1.8" /><circle cx="7" cy="19" r="2.5" fill="none" stroke="currentColor" strokeWidth="1.8" /><path d="M9 6.5l6 4M15 13.5l-6 4" stroke="currentColor" strokeWidth="1.8" /></>, 22, "0 0 24 24"),
  plan: svg(<><rect x="3.5" y="5" width="17" height="15" rx="2" fill="none" stroke="currentColor" strokeWidth="1.8" /><path d="M3.5 10h17M8 3v4M16 3v4" stroke="currentColor" strokeWidth="1.8" /></>, 22, "0 0 24 24"),
  review: svg(<path d="M4 12a8 8 0 1 0 2.4-5.7M4 4v4h4" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />, 22, "0 0 24 24"),
  me: svg(<><circle cx="12" cy="8.5" r="3.5" fill="none" stroke="currentColor" strokeWidth="1.8" /><path d="M5 20c1-3.5 3.8-5 7-5s6 1.5 7 5" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" /></>, 22, "0 0 24 24"),
};

let toastTimer: ReturnType<typeof setTimeout> | undefined;
export function toast(msg: string) {
  document.querySelectorAll(".toast").forEach((t) => t.remove());
  const t = document.createElement("div");
  t.className = "toast";
  t.textContent = msg;
  t.setAttribute("role", "status");
  document.body.appendChild(t);
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.remove(), 2800);
}

export const Html = ({ html, as: Tag = "span", className }: { html: string; as?: "span" | "div" | "p"; className?: string }) => (
  <Tag className={className} dangerouslySetInnerHTML={{ __html: html }} />
);
