// The sheet of what Quaver supports, on the "What is Quaver?" page.
// Each item links to its Learn page, with the quiet dotted line of all Learn links.
// Each number format shows its bits as a row of small squares (small square cells): sign, exponent, mantissa, or integer bits.

import Link from '@docusaurus/Link';

import styles from './styles.module.css';

// Bits of each format, from the top bit: s = sign, e = exponent, m = mantissa, i = integer bit
const VALUE_FORMATS = [
  {name: 'int8', bits: 'siiiiiii'},
  {name: 'int4', bits: 'siii'},
  {name: 'int2', bits: 'si'},
  {name: 'uint8', bits: 'iiiiiiii'},
  {name: 'uint4', bits: 'iiii'},
  {name: 'uint2', bits: 'ii'},
  {name: 'fp4', note: 'e2m1', bits: 'seem'},
  {name: 'ufp4', note: 'e2m2', bits: 'eemm'},
];

const SCALE_FORMATS = [
  {name: 'ue4m3', note: 'NVFP4', bits: 'eeeemmm'},
  {name: 'uint6', bits: 'iiiiii'},
  {name: 'uint10', bits: 'iiiiiiiiii'},
];

const BIT_NAMES = {s: 'sign', e: 'exponent', m: 'mantissa', i: 'integer'};

// The Learn page of each item
const LEARN = {
  'Formats'                              : 'int-fp',
  'Scales'                               : 'scale-zero-point',
  'Layers'                               : 'integer-inference',
  'int8'                                 : 'int-fp',
  'int4'                                 : 'int-fp',
  'int2'                                 : 'int-fp',
  'uint8'                                : 'int-fp',
  'uint4'                                : 'int-fp',
  'uint2'                                : 'int-fp',
  'fp4'                                  : 'int-fp',
  'ufp4'                                 : 'int-fp',
  'ue4m3'                                : 'block-scales',
  'uint6'                                : 'block-scales',
  'uint10'                               : 'block-scales',
  'per tensor'                           : 'granularity',
  'per channel'                          : 'granularity',
  'per block'                            : 'granularity',
  'per token'                            : 'granularity',
  'measured each time'                   : 'clipping-range',
  'moving average of the range'          : 'clipping-range',
  'learned with LSQ or LSQ+'             : 'qat',
  'shared with other tensors'            : 'shared-scales',
  'symmetric'                            : 'symmetric',
  'asymmetric (INT formats)'             : 'asymmetric',
  'Conv2d'                               : 'integer-inference',
  'Linear'                               : 'integer-inference',
  'residual add'                         : 'residual-add',
  'average pooling'                      : 'average-pooling',
  'SimA attention'                       : 'sima-attention',
  'block-scaled Conv2d and Linear'       : 'block-scales',
  'ReLU'                                 : 'activations',
  'ReLU6'                                : 'activations',
  '16-bit multiplier and shift (default)': 'requantization',
  'shift only'                           : 'requantization',
};

function Item({name, children}) {
  const page = LEARN[name];
  return <Link to={`/docs/learn/${page}`}>{children}</Link>;
}

// The other rows: a group name, then rows of a label and its items
const GROUPS = [
  {
    name: 'Scales',
    rows: [
      {label: 'How many', items: ['per tensor', 'per channel', 'per block', 'per token']},
      {label: 'How they are set', items: ['measured each time', 'moving average of the range', 'learned with LSQ or LSQ+', 'shared with other tensors']},
      {label: 'Zero point', items: ['symmetric', 'asymmetric (INT formats)']},
    ],
  },
  {
    name: 'Layers',
    rows: [
      {label: 'Layers', items: ['Conv2d', 'Linear', 'residual add', 'average pooling', 'SimA attention', 'block-scaled Conv2d and Linear']},
      {label: 'Activations', items: ['ReLU', 'ReLU6']},
      {label: 'Requantization', items: ['16-bit multiplier and shift (default)', 'shift only']},
    ],
  },
];

function Format({name, note, bits}) {
  const counts = [...bits].reduce((all, bit) => ({...all, [bit]: (all[bit] ?? 0) + 1}), {});
  const label  = `${name}: ${bits.length} bits (` + Object.entries(counts).map(([bit, count]) => `${count} ${BIT_NAMES[bit]}`).join(', ') + ')';
  return (
    <li className={styles.format} aria-label={label}>
      <span className={styles.formatName}>
        <Item name={name}><code>{name}</code></Item>
        {note && <span className={styles.note}>{note}</span>}
      </span>
      <span className={styles.bits} aria-hidden="true">
        {[...bits].map((bit, index) => <span key={index} className={`${styles.bit} ${styles[bit]}`} />)}
      </span>
    </li>
  );
}

function Row({label, children}) {
  return (
    <div className={styles.row}>
      <dt className={styles.label}>{label}</dt>
      <dd className={styles.items}>{children}</dd>
    </div>
  );
}

export default function SupportSheet() {
  return (
    <div className={styles.sheet}>
      <section className={styles.group}>
        <h4 className={styles.groupName}><Item name="Formats">Formats</Item></h4>
        <dl className={styles.rows}>
          <Row label="Values">
            <ul className={styles.formats}>{VALUE_FORMATS.map((format) => <Format key={format.name} {...format} />)}</ul>
          </Row>
          <Row label="Block scales">
            <ul className={styles.formats}>{SCALE_FORMATS.map((format) => <Format key={format.name} {...format} />)}</ul>
          </Row>
        </dl>
        <p className={styles.legend} aria-hidden="true">
          {Object.entries(BIT_NAMES).map(([bit, name]) => (
            <span key={bit}><span className={`${styles.bit} ${styles[bit]}`} />{name}</span>
          ))}
        </p>
      </section>
      {GROUPS.map((group) => (
        <section key={group.name} className={styles.group}>
          <h4 className={styles.groupName}><Item name={group.name}>{group.name}</Item></h4>
          <dl className={styles.rows}>
            {group.rows.map((row) => (
              <Row key={row.label} label={row.label}>
                <ul className={styles.tags}>{row.items.map((item) => <li key={item}><Item name={item}>{item}</Item></li>)}</ul>
              </Row>
            ))}
          </dl>
        </section>
      ))}
    </div>
  );
}
