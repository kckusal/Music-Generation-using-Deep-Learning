import Head from "next/head";
import { useEffect, useState, useCallback, useRef } from "react";
import { useRouter } from "next/router";

import styles from "../styles/Home.module.css";

const GENERATE_API_URL = "https://7a6b081d2d2f.ngrok.io/generate";
const SAMPLE_GENERATE_API_URL =
  "https://music-generation-using-deep-learning.vercel.app/generate/sample";

export default function Home() {
  const router = useRouter();
  const instrumentRef = useRef();

  const [generateAPI, setGenerateAPI] = useState(GENERATE_API_URL);
  const [isGenerating, setIsGenerating] = useState(false);
  const [result, setResult] = useState("");
  const [error, setError] = useState();

  const [isPlaying, setIsPlaying] = useState(false);

  useEffect(() => {
    if (!["/", "/sample"].includes(router.pathname)) {
      router.push("/");
      return;
    }

    window.setGenerateAPI = setGenerateAPI;
  }, []);

  useEffect(() => {
    instrumentRef.current = new Instrument();
    window.instrument = instrumentRef.current;
  }, []);

  const generate = useCallback(() => {
    setIsGenerating(true);
    setError(undefined);

    fetch(generateAPI)
      .then((res) => res.text())
      .then((result) => setResult(result))
      .catch((e) => {
        setError(e);
        console.error({ e });
        alert("Some error occurred during generation.");
      })
      .finally(() => setIsGenerating(false));
  }, [router.pathname, generateAPI]);

  const play = useCallback(() => {
    if (!instrumentRef.current) {
      alert("Cannot find Instrument reference from musical.js library.");
      return;
    }

    if (!result) {
      alert("No ABC text found to play!");
      return;
    }

    setIsPlaying(true);
    instrumentRef.current.play(result, () => {
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
        <h2>Music Generation with Deep Learning</h2>
    
        <h2>About:</h2>
        <p>This page lets you try music generated with Deep Learning.</p>

        <p>
          <label>Generate API URL: </label>
          <input
            type="text"
            value={generateAPI}
            onChange={(e) => setGenerateAPI(e.target.value)}
            style={{ width: "100%" }}
          />
        </p>

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
        <div>
          {!isGenerating &&
            (error ? (
              <em>Something went wrong! Try again.</em>
            ) : (
              <code>{result}</code>
            ))}
        </div>

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
