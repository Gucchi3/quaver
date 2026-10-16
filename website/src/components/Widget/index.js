// An interactive figure from website/static/widgets/, shown in a sandboxed iframe.
// The figure gets the colors of the site as CSS variables, its params as window.widgetParams, and tells this page its height.

import {useEffect, useId, useRef, useState} from 'react';

import useBaseUrl from '@docusaurus/useBaseUrl';
import {useColorMode} from '@docusaurus/theme-common';

import styles from './styles.module.css';

// The colors of the site, as the variables that the figures use
const THEMES = {
  light: {text: '#2f2219', muted: '#6b5a4c', faint: '#a08868', bg: '#fbfaf5', bg2: '#f5efe4', border: '#dccbad', borderStrong: '#c4ad8a', accent: '#b5442a', onAccent: '#ffffff'},
  dark: {text: '#ecdfcb', muted: '#c7b29b', faint: '#87705a', bg: '#1e1915', bg2: '#262019', border: '#4a3c2e', borderStrong: '#66533f', accent: '#dd7350', onAccent: '#1e1915'},
};

// The color ramps of the figures: [50, 100, 200, 600, 800]
const RAMPS = {
  purple: ['#EEEDFE', '#CECBF6', '#AFA9EC', '#534AB7', '#3C3489'],
  teal: ['#E1F5EE', '#9FE1CB', '#5DCAA5', '#0F6E56', '#085041'],
  coral: ['#FAECE7', '#F5C4B3', '#F0997B', '#993C1D', '#712B13'],
  pink: ['#FBEAF0', '#F4C0D1', '#ED93B1', '#993556', '#72243E'],
  gray: ['#F1EFE8', '#D3D1C7', '#B4B2A9', '#5F5E5A', '#444441'],
  blue: ['#E6F1FB', '#B5D4F4', '#85B7EB', '#185FA5', '#0C447C'],
  green: ['#EAF3DE', '#C0DD97', '#97C459', '#3B6D11', '#27500A'],
  amber: ['#FAEEDA', '#FAC775', '#EF9F27', '#854F0B', '#633806'],
  red: ['#FCEBEB', '#F7C1C1', '#F09595', '#A32D2D', '#791F1F'],
};

const BASE_CSS = `
*,*::before,*::after{box-sizing:border-box}
html,body{margin:0;padding:0;background:transparent}
body{display:flow-root;font-family:var(--font-sans);font-size:16px;line-height:1.7;color:var(--text-primary);-webkit-font-smoothing:antialiased;overflow-wrap:break-word}
h1,h2,h3{font-weight:500;line-height:1.3;margin:0 0 .5em}
p{margin:0 0 1em}
a{color:var(--text-accent)}
code,pre,kbd{font-family:var(--font-mono);font-size:.9em}
button{font:inherit;font-size:14px;line-height:1.4;color:var(--text-primary);background:transparent;border:1px solid var(--border-strong);border-radius:var(--radius);padding:6px 14px;cursor:pointer}
button:hover{background:var(--surface-1)}
button:disabled{opacity:.45;cursor:default}
button:focus-visible,input:focus-visible,select:focus-visible{outline:2px solid var(--fill-accent);outline-offset:1px}
select,input[type=number],input[type=text]{font:inherit;font-size:14px;color:var(--text-primary);background:var(--surface-2);border:1px solid var(--border-strong);border-radius:var(--radius);padding:6px 10px;min-height:36px}
input[type=range],input[type=checkbox],input[type=radio]{accent-color:var(--fill-accent)}
table{border-collapse:collapse}
th,td{padding:4px 8px;border-bottom:1px solid var(--border)}
svg text{font-family:var(--font-sans)}
.t{font-size:14px;fill:var(--text-primary)}
.ts{font-size:12px;fill:var(--text-secondary)}
.th{font-size:14px;font-weight:500;fill:var(--text-primary)}
.box{fill:var(--surface-1);stroke:var(--border-strong)}
.arr{stroke:var(--text-secondary);stroke-width:1.5;fill:none}
.leader{stroke:var(--text-muted);stroke-width:.5;stroke-dasharray:3 3;fill:none}
.sr-only{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap;border:0}
`;

function rampCss() {
  let css = '';
  for (const [name, [c50, c100, c200, c600, c800]] of Object.entries(RAMPS)) {
    const shapes = `.c-${name}>rect,.c-${name}>circle,.c-${name}>ellipse,rect.c-${name},circle.c-${name},ellipse.c-${name}`;
    const dark = shapes.split(',').map((shape) => `[data-mode=dark] ${shape}`).join(',');
    css += `${shapes}{fill:${c50};stroke:${c600}}.c-${name}>.t,.c-${name}>.th{fill:${c800}}.c-${name}>.ts{fill:${c600}}`;
    css += `${dark}{fill:${c800};stroke:${c200}}[data-mode=dark] .c-${name}>.t,[data-mode=dark] .c-${name}>.th{fill:${c100}}[data-mode=dark] .c-${name}>.ts{fill:${c200}}`;
  }
  return css;
}

function themeCss(theme) {
  const mix = (color, percent) => `color-mix(in srgb, ${color} ${percent}%, transparent)`;
  const vars = {
    '--surface-0': theme.bg, '--surface-1': theme.bg2, '--surface-2': theme.bg, '--surface-3': theme.bg,
    '--text-primary': theme.text, '--text-secondary': theme.muted, '--text-muted': theme.faint,
    '--border': theme.border, '--border-strong': theme.borderStrong, '--border-stronger': theme.borderStrong,
    '--fill-primary': theme.text, '--on-primary': theme.bg,
    '--text-accent': theme.accent, '--fill-accent': theme.accent, '--on-accent': theme.onAccent,
    '--bg-accent': mix(theme.accent, 15), '--border-accent': theme.accent,
    '--font-sans': 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif', '--font-mono': 'ui-monospace, SFMono-Regular, Menlo, monospace',
    '--radius': '3px',
    '--p': theme.text, '--s': theme.muted, '--t': theme.faint, '--bg2': theme.bg2, '--b': theme.border,
    '--color-text-primary': theme.text, '--color-text-secondary': theme.muted, '--color-text-tertiary': theme.faint, '--color-text-info': theme.accent,
    '--color-background-primary': theme.bg, '--color-background-secondary': theme.bg2, '--color-background-tertiary': theme.bg2,
    '--color-border-primary': theme.borderStrong, '--color-border-secondary': theme.borderStrong, '--color-border-tertiary': theme.border,
    '--border-radius-md': '3px', '--border-radius-lg': '6px',
  };
  return `:root{${Object.entries(vars).map(([key, value]) => `${key}:${value};`).join('')}}`;
}

function buildDoc(html, id, params, mode) {
  const resize = `<script>(function(){function send(){parent.postMessage({__widget:${JSON.stringify(id)},h:document.documentElement.scrollHeight},'*');}new ResizeObserver(send).observe(document.documentElement);window.addEventListener('load',send);send();})();</script>`;
  return `<!doctype html><html data-mode="${mode}"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><style>${themeCss(THEMES[mode])}${BASE_CSS}${rampCss()}</style><script>window.widgetParams=${JSON.stringify(params || {})};</script></head><body>${html}${resize}</body></html>`;
}

export default function Widget({src, params, title}) {
  const {colorMode} = useColorMode();
  const url         = useBaseUrl(`/widgets/${src}.html`);
  const id          = useId();
  const frame       = useRef(null);
  const [html, setHtml]     = useState(null);
  const [height, setHeight] = useState(360);

  // Load the figure once
  useEffect(() => {
    let alive = true;
    fetch(url).then((response) => response.text()).then((text) => alive && setHtml(text));
    return () => { alive = false; };
  }, [url]);

  // Take the height that the figure sends
  useEffect(() => {
    const onMessage = (event) => {
      if (frame.current && event.source === frame.current.contentWindow && event.data && event.data.__widget === id) {
        setHeight(Math.min(Math.max(Number(event.data.h) || 0, 40), 4000));
      }
    };
    window.addEventListener('message', onMessage);
    return () => window.removeEventListener('message', onMessage);
  }, [id]);

  return (
    <figure className={styles.widget}>
      {html === null ? (
        <div className={styles.loading} style={{height}}>Loading the figure…</div>
      ) : (
        <iframe ref={frame} className={styles.frame} title={title || src} sandbox="allow-scripts" srcDoc={buildDoc(html, id, params, colorMode)} style={{height}} />
      )}
    </figure>
  );
}
