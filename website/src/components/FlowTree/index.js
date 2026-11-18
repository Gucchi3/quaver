// モデルが Quaver を呼ぶ順を、開いたり閉じたりできる木で見せる。モードごとにタブを 1 つ持つ。
// データは website/scripts/flow_trace.py から来る。これはガイドのモデルを profiler の下で動かす。

import {useMemo, useState} from 'react';

import Link from '@docusaurus/Link';

import flow from '@site/src/data/flow.json';
import styles from './styles.module.css';

// 呼び出しの API のページとその見出し: 関数は名前で、メソッドはメソッドの名前で、内部用の関数はそのクラスで探す
function apiLink(node) {
  if (node.file === 'guide') {
    return '/docs/guides/custom-layer';
  }
  const match = node.file.match(/^quaver\/(\w+)\/(\w+)\.py$/);
  if (!match) {
    return null;
  }
  const parts  = node.name.split('.');
  const last   = parts[parts.length - 1];
  const anchor = (last.startsWith('_') ? parts[0] : last).toLowerCase();
  return anchor.startsWith('_') ? `/docs/api/${match[1]}/${match[2]}` : `/docs/api/${match[1]}/${match[2]}#${anchor}`;
}

function Call({node, depth, open, toggle, path}) {
  const isOpen   = open.has(path);
  const hasKids  = node.children.length > 0;
  const internal = node.name.split('.').some((part) => part.startsWith('_') && !part.startsWith('__'));
  const link     = apiLink(node);
  const name     = <code className={styles.name}>{node.name}</code>;
  return (
    <li className={styles.item}>
      <div className={`${styles.row} ${internal ? styles.internal : ''}`}>
        {hasKids ? (
          <button type="button" className={styles.toggle} aria-expanded={isOpen} aria-label={`${isOpen ? 'Close' : 'Open'} ${node.name}`} onClick={() => toggle(path)}>
            <span className={isOpen ? styles.caretOpen : styles.caret} aria-hidden="true" />
          </button>
        ) : (
          <span className={styles.leaf} aria-hidden="true" />
        )}
        <span className={styles.label}>
          {node.module && <span className={styles.module}>{node.module}</span>}
          {link ? <Link to={link}>{name}</Link> : name}
          {node.count > 1 && <span className={styles.count}>×{node.count}</span>}
          {node.note && <span className={styles.note}>{node.note}</span>}
        </span>
      </div>
      {hasKids && isOpen && (
        <ul className={styles.list}>
          {node.children.map((child, index) => <Call key={index} node={child} depth={depth + 1} open={open} toggle={toggle} path={`${path}.${index}`} />)}
        </ul>
      )}
    </li>
  );
}

// 木のすべてのパスを、ある深さまで集める
function paths(nodes, prefix, depth) {
  return nodes.flatMap((node, index) => {
    const path = `${prefix}${index}`;
    return depth <= 0 || node.children.length === 0 ? [] : [path, ...paths(node.children, `${path}.`, depth - 1)];
  });
}

export default function FlowTree() {
  const [modeIndex, setModeIndex] = useState(2);
  const mode = flow.modes[modeIndex];

  // 初めは、モデルとその層を開いておく
  const start = useMemo(() => new Set(paths(mode.calls, '', 2)), [mode]);
  const [open, setOpen] = useState(start);
  const toggle = (path) => setOpen((now) => {
    const next = new Set(now);
    next.has(path) ? next.delete(path) : next.add(path);
    return next;
  });
  const choose = (index) => {
    setModeIndex(index);
    setOpen(new Set(paths(flow.modes[index].calls, '', 2)));
  };

  return (
    <div className={styles.flow}>
      <div className={styles.tabs} role="tablist" aria-label="Mode">
        {flow.modes.map((item, index) => (
          <button key={item.key} type="button" role="tab" aria-selected={index === modeIndex} className={index === modeIndex ? styles.tabActive : styles.tab} onClick={() => choose(index)}>
            {item.title}
          </button>
        ))}
      </div>
      <p className={styles.about}>{mode.about}</p>
      <div className={styles.actions}>
        <button type="button" className={styles.action} onClick={() => setOpen(new Set(paths(mode.calls, '', 99)))}>Open all</button>
        <button type="button" className={styles.action} onClick={() => setOpen(new Set(paths(mode.calls, '', 2)))}>Layers only</button>
      </div>
      <ul className={styles.tree}>
        {mode.calls.map((node, index) => <Call key={`${mode.key}-${index}`} node={node} depth={0} open={open} toggle={toggle} path={`${index}`} />)}
      </ul>
    </div>
  );
}
