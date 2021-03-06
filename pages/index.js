import Head from "next/head";
import { useEffect, useState, useCallback, useRef } from "react";
import styles from "../styles/Home.module.css";

const GENERATE_API_URL =
  "https://music-generation-using-deep-learning.vercel.app/generate";

export default function Home() {
  const instrumentRef = useRef();

  const [isGenerating, setIsGenerating] = useState(false);
  const [result, setResult] = useState();
  const [error, setError] = useState();

  const [isPlaying, setIsPlaying] = useState(false);

  useEffect(() => {
    instrumentRef.current = Instrument;
  }, []);

  const generate = useCallback(() => {
    setIsGenerating(true);
    setError(undefined);

    fetch(GENERATE_API_URL)
      .then((res) => res.text())
      .then((result) => setResult(result))
      .catch((e) => {
        setError(e);
        console.error({ e });
      })
      .finally(() => setIsGenerating(false));
  }, []);

  const play = useCallback(() => {
    if (!instrumentRef.current) {
      alert("Cannot find Instrument reference from musical.js library.");
      return;
    }

    if (!result) {
      alert("No ABC text found to play!");
      return;
    }

    const inst = new instrumentRef.current();
    setIsPlaying(true);
    inst.play(result, () => {
      setIsPlaying(false);
    });
  }, [result, instrumentRef.current]);

  return (
    <div className={styles.container}>
      <Head>
        <title>Music Generation Using Deep Learning</title>

        <script type="text/javascript" src={"scripts/musical.min.js"}></script>
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
            ? "The following ABC text was generated:"
            : "You have NOT generated any music yet."}
        </p>
        <pre>
          {!isGenerating &&
            (error ? (
              <em>Something went wrong! Try again.</em>
            ) : (
              <code>{result}</code>
            ))}
        </pre>

        <p>
          &nbsp;
          {result && (
            <button disabled={isPlaying} onClick={play}>
              Play
            </button>
          )}
        </p>
      </main>
    </div>
  );
}
