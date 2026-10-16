// The path from Quaver to the device: Quaver trains and checks the model, model_export() writes model.json and model.bin, and the device reads model.bin from its memory.

import styles from './styles.module.css';

const STEPS = [
  {name: 'Quaver', role: 'Train the model with QAT, and check it with the integer mode.'},
  {name: 'model.json + model.bin', role: 'What to compute, and every number packed for the memory of the device.'},
  {name: 'Your device', role: 'Reads model.bin from its memory (for example PSRAM), and runs each layer with integer math.'},
];

// What happens between two steps
const HANDOFFS = [
  ['model_export()'],
  ['write model.bin', 'into memory'],
];

export default function DeviceFlow() {
  return (
    <figure className={styles.flow}>
      <ol className={styles.steps}>
        {STEPS.map((step, index) => (
          <li key={step.name} className={styles.stepItem}>
            <div className={styles.step}>
              <span className={styles.name}>{step.name}</span>
              <span className={styles.role}>{step.role}</span>
            </div>
            {index < HANDOFFS.length && (
              <div className={styles.handoff}>
                <span className={styles.things}>{HANDOFFS[index].map((thing) => <span key={thing}>{thing}</span>)}</span>
                <span className={styles.arrow} aria-hidden="true" />
              </div>
            )}
          </li>
        ))}
      </ol>
    </figure>
  );
}
