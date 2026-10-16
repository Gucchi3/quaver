// The calls that a model makes into Quaver, in order, as an open-and-close tree, with one tab for each mode.
// The data comes from website/scripts/flow_trace.py, which runs the model of the guides under a profiler.

import {useMemo, useState} from 'react';

import Link from '@docusaurus/Link';

import flow from '@site/src/data/flow.json';
import styles from './styles.module.css';

// The API page of a call, and the heading on it: functions by their name, methods by their method name, private helpers by their class
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

// Every path in a tree, down to a depth
function paths(nodes, prefix, depth) {
  return nodes.flatMap((node, index) => {
    const path = `${prefix}${index}`;
    return depth <= 0 || node.children.length === 0 ? [] : [path, ...paths(node.children, `${path}.`, depth - 1)];
  });
}

export default function FlowTree() {
  const [modeIndex, setModeIndex] = useState(2);
  const mode = flow.modes[modeIndex];

  // Open the model and its layers at the start
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
