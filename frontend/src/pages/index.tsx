import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';

import styles from './index.module.css';

/**
 * T009: Hero Section with Gradient Text
 */
function HomepageHeader() {
  return (
    <header className={styles.heroBanner}>
      <div className="container">
        <Heading as="h1" className={styles.heroTitle}>
          From Digital Brains to{' '}
          <span className={styles.gradientText}>Embodied Bodies</span>
        </Heading>
        <p className={styles.heroSubtitle}>
          Master the complete Physical AI stack: ROS 2, NVIDIA Isaac Sim, and VLA architectures.
          Build intelligent robots that perceive, reason, and act in the real world.
        </p>
        <div className={styles.buttons}>
          <Link
            className={clsx('button button--primary button--lg', styles.primaryButton)}
            to="/docs/intro">
            Get Started
          </Link>
          <Link
            className={clsx('button button--outline button--lg', styles.secondaryButton)}
            to="/docs/intro">
            Chat with Book
          </Link>
        </div>
      </div>
    </header>
  );
}

/**
 * T011: Feature Grid Data
 */
const features = [
  {
    title: 'ROS 2',
    icon: '🤖',
    description: 'Build robust robot applications with the Robot Operating System. Learn nodes, topics, services, and actions for real-time control.',
  },
  {
    title: 'Isaac Sim',
    icon: '🎮',
    description: 'Train and test robots in photorealistic simulation. NVIDIA Omniverse-powered physics and sensor simulation at scale.',
  },
  {
    title: 'RealSense',
    icon: '👁️',
    description: 'Integrate depth cameras and 3D perception. Visual SLAM, object detection, and spatial understanding for autonomy.',
  },
];

/**
 * T011: Feature Card Component
 */
function FeatureCard({ title, icon, description }: { title: string; icon: string; description: string }) {
  return (
    <div className={styles.featureCard}>
      <div className={styles.featureIcon}>{icon}</div>
      <h3 className={styles.featureTitle}>{title}</h3>
      <p className={styles.featureDescription}>{description}</p>
    </div>
  );
}

/**
 * T011: Feature Grid Section
 */
function HomepageFeatures() {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className={styles.featureGrid}>
          {features.map((feature, idx) => (
            <FeatureCard key={idx} {...feature} />
          ))}
        </div>
      </div>
    </section>
  );
}

/**
 * Main Homepage Component
 */
export default function Home(): JSX.Element {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title="Physical AI & Humanoid Robotics"
      description="Physical AI & Humanoid Robotics Textbook - From digital AI to embodied intelligence. Learn ROS 2, NVIDIA Isaac Sim, and VLA architectures.">
      <HomepageHeader />
      <main>
        <HomepageFeatures />
      </main>
    </Layout>
  );
}
