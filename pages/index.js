import Head from "next/head";
import styles from "../styles/Home.module.css";

export default function Home() {
  return (
    <div className={styles.container}>
      <Head>
        <title>Music Generation Using Deep Learning</title>
      </Head>

      <main className={styles.main}>
        <p>This page lets you try music generated with deep learning.</p>

        <p>
          <button onClick={() => alert("WIP.")}>Generate Now</button>
        </p>
      </main>
    </div>
  );
}
