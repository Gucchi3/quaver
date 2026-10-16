import Link from '@docusaurus/Link';
import useBaseUrl from '@docusaurus/useBaseUrl';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Heading from '@theme/Heading';
import Layout from '@theme/Layout';
import ThemedImage from '@theme/ThemedImage';

import styles from './index.module.css';

// The three points of Quaver under the logo
const FEATURES = [
  {
    title: 'Same integer math as the device',
    text: 'In the integer mode, each layer computes with integer math only, like the device. The training mode uses the same functions after the multiply-accumulate step, so both modes give the same output, bit for bit.',
  },
  {
    title: 'INT, FP4, and NVFP4',
    text: 'Use INT8, INT4, INT2, or FP4 (e2m1), with one scale per tensor, per channel, or per block. Layers with NVFP4 block scales and SimA attention also run with integer math only.',
  },
  {
    title: 'Fast QAT',
    text: 'A training step never waits for the GPU to send values to the CPU. So you can capture the whole step as a CUDA graph and run it fast.',
  },
];


export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  const logo         = {light: useBaseUrl('/img/quaver-logo-light.svg'), dark: useBaseUrl('/img/quaver-logo-dark.svg')};
  return (
    <Layout description={siteConfig.tagline}>
      <header className={styles.hero}>
        <div className="container">
          <Heading as="h1" className={styles.title}>
            <ThemedImage className={styles.logo} alt={siteConfig.title} sources={logo} />
          </Heading>
          <p className={styles.tagline}>{siteConfig.tagline}</p>
          <div className={styles.buttons}>
            <Link className="button button--primary button--lg" to="/docs/quickstart">Try it in 5 minutes</Link>
            <Link className="button button--secondary button--outline button--lg" to="/docs/intro">Read the docs</Link>
          </div>
        </div>
      </header>
      <main className="container">
        <div className={`row ${styles.features}`}>
          {FEATURES.map(({title, text}) => (
            <section key={title} className="col col--4">
              <Heading as="h2" className={styles.featureTitle}>{title}</Heading>
              <p>{text}</p>
            </section>
          ))}
        </div>
      </main>
    </Layout>
  );
}
