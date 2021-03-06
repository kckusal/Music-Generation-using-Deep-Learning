import Head from "next/head";
import { useEffect, useState, useCallback } from "react";
import styles from "../styles/Home.module.css";

const GENERATE_API_URL =
  "https://music-generation-using-deep-learning.vercel.app/generate";

export default function Home() {
  const [isGenerating, setIsGenerating] = useState(false);
  const [result, setResult] = useState();
  const [error, setError] = useState();

  const generate = useCallback(() => {
    setIsGenerating(true);
    setError(undefined);

    fetch(GENERATE_API_URL)
      .then((res) => res.text)
      .then((result) => setResult(result))
      .catch((e) => {
        setError(e);
        console.error({ e });
      })
      .finally(() => setIsGenerating(false));
  }, []);

  return (
    <div className={styles.container}>
      <Head>
        <title>Music Generation Using Deep Learning</title>
      </Head>

      <main className={styles.main}>
        <h2>About:</h2>
        <p>This page lets you try music generated with Deep Learning.</p>

        <p>
          <button onClick={generate}>Generate New</button>
        </p>

        <br />
        <h2>Result:</h2>
        <p>
          {isGenerating
            ? "Generating..."
            : result
            ? ""
            : "You have NOT generated any music yet."}
        </p>
        <p>
          &nbsp;
          {!isGenerating &&
            (error ? <em>Something went wrong! Try again.</em> : result)}
        </p>
      </main>
    </div>
  );
}
