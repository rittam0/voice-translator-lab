import { useEffect, useRef, useState } from "react";
import {
  ApiError,
  fetchAudioUrl,
  SourceLanguage,
  TargetLanguage,
  translate,
  TranslationResult,
} from "./api";
import Waveform from "./Waveform";

type State =
  | "idle"
  | "permission"
  | "recording"
  | "stopping"
  | "processing"
  | "completed"
  | "error";

const MAX_RECORDING_MS = 15_000;
const MIN_RECORDING_MS = 2_700;

function chooseMimeType(): string {
  const choices = ["audio/webm;codecs=opus", "audio/ogg;codecs=opus", "audio/mp4"];
  return choices.find((type) => MediaRecorder.isTypeSupported(type)) ?? "";
}

export default function App() {
  const [source, setSource] = useState<SourceLanguage>("en");
  const [target, setTarget] = useState<TargetLanguage>("ja");
  const [state, setState] = useState<State>("idle");
  const [status, setStatus] = useState("Tap to begin");
  const [analyser, setAnalyser] = useState<AnalyserNode | null>(null);
  const [result, setResult] = useState<TranslationResult | null>(null);
  const [playbackBlocked, setPlaybackBlocked] = useState(false);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const startedAtRef = useRef(0);
  const stopTimerRef = useRef<number | null>(null);
  const audioRef = useRef<HTMLAudioElement>(null);
  const blobUrlRef = useRef<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const releaseBlobUrl = () => {
    if (blobUrlRef.current) {
      URL.revokeObjectURL(blobUrlRef.current);
      blobUrlRef.current = null;
    }
  };

  const releaseCapture = () => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    void audioContextRef.current?.close();
    audioContextRef.current = null;
    setAnalyser(null);
    if (stopTimerRef.current !== null) window.clearTimeout(stopTimerRef.current);
    stopTimerRef.current = null;
  };

  useEffect(
    () => () => {
      releaseCapture();
      releaseBlobUrl();
      abortRef.current?.abort();
    },
    [],
  );

  const startRecording = async () => {
    if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
      setState("error");
      setStatus("This browser does not support microphone recording.");
      return;
    }
    releaseBlobUrl();
    setResult(null);
    setPlaybackBlocked(false);
    setState("permission");
    setStatus("Waiting for microphone permission…");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
        },
      });
      streamRef.current = stream;
      const context = new AudioContext();
      audioContextRef.current = context;
      const sourceNode = context.createMediaStreamSource(stream);
      const analyserNode = context.createAnalyser();
      analyserNode.fftSize = 256;
      analyserNode.smoothingTimeConstant = 0.72;
      sourceNode.connect(analyserNode);
      setAnalyser(analyserNode);

      const mimeType = chooseMimeType();
      const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
      recorderRef.current = recorder;
      chunksRef.current = [];
      recorder.ondataavailable = (event) => {
        if (event.data.size) chunksRef.current.push(event.data);
      };
      recorder.onerror = () => {
        releaseCapture();
        setState("error");
        setStatus("Recording failed. Check your microphone and try again.");
      };
      recorder.onstop = () => void finishRecording(recorder.mimeType);
      startedAtRef.current = performance.now();
      recorder.start(250);
      setState("recording");
      setStatus("Listening… tap again to translate");
      stopTimerRef.current = window.setTimeout(() => stopRecording(), MAX_RECORDING_MS);
    } catch (error) {
      releaseCapture();
      setState("error");
      const denied =
        error instanceof DOMException &&
        (error.name === "NotAllowedError" || error.name === "SecurityError");
      setStatus(
        denied
          ? "Microphone access was denied. Allow it in browser settings and retry."
          : "The microphone could not be opened.",
      );
    }
  };

  const stopRecording = () => {
    if (recorderRef.current?.state !== "recording") return;
    setState("stopping");
    setStatus("Finishing your recording…");
    recorderRef.current.stop();
  };

  const finishRecording = async (mimeType: string) => {
    const duration = performance.now() - startedAtRef.current;
    releaseCapture();
    const blob = new Blob(chunksRef.current, {
      type: mimeType || "audio/webm",
    });
    if (!blob.size) {
      setState("error");
      setStatus("No audio was recorded. Please try again.");
      return;
    }
    if (duration < MIN_RECORDING_MS) {
      setState("error");
      setStatus("That was too short. Record at least 3 seconds.");
      return;
    }
    setState("processing");
    setStatus(`Transcribing and creating ${target === "ja" ? "Japanese" : "French"} speech… model loading can take time.`);
    const controller = new AbortController();
    abortRef.current = controller;
    const timeout = window.setTimeout(() => controller.abort(), 10 * 60 * 1000);
    try {
      const translation = await translate(blob, source, target, controller.signal);
      const url = await fetchAudioUrl(translation.audio_url, controller.signal);
      blobUrlRef.current = url;
      setResult(translation);
      setState("completed");
      setStatus(`${target === "ja" ? "Japanese" : "French"} translation ready`);
      setPlaybackBlocked(false);
      queueMicrotask(async () => {
        if (!audioRef.current) return;
        audioRef.current.src = url;
        try {
          await audioRef.current.play();
        } catch {
          setPlaybackBlocked(true);
        }
      });
    } catch (error) {
      setState("error");
      const apiError = error instanceof ApiError ? error : null;
      setStatus(
        apiError
          ? `${apiError.stage ? `${apiError.stage}: ` : ""}${apiError.message}`
          : "Translation failed. Please try again.",
      );
    } finally {
      window.clearTimeout(timeout);
      abortRef.current = null;
    }
  };

  const handleMicrophone = () => {
    if (state === "recording") stopRecording();
    else if (!["permission", "stopping", "processing"].includes(state))
      void startRecording();
  };

  const retryPlayback = async () => {
    try {
      await audioRef.current?.play();
      setPlaybackBlocked(false);
    } catch {
      setStatus("Playback is still blocked. Use the audio controls below.");
    }
  };

  return (
    <main className="app">
      <header>
        <p className="kicker">VOICE TRANSLATOR · PROTOTYPE</p>
        <h1>
          Speak naturally.
          <span> Hear your translation.</span>
        </h1>
        <p className="intro">
          A 3–15 second clip is translated locally, then converted toward the
          character of your voice. Clone only a consenting speaker.
        </p>
      </header>

      <section className="translator" aria-live="polite">
        <div className="language-row">
          <label htmlFor="source-language">Speaking in</label>
          <select
            id="source-language"
            value={source}
            onChange={(event) => setSource(event.target.value as SourceLanguage)}
            disabled={["permission", "recording", "stopping", "processing"].includes(
              state,
            )}
          >
            <option value="en">English</option>
            <option value="hi">Hindi</option>
            <option value="ory">Odia · Experimental</option>
          </select>
          <label className="target" htmlFor="target-language">
            <span>Translate to</span>
            <select
              id="target-language"
              value={target}
              onChange={(event) => setTarget(event.target.value as TargetLanguage)}
              disabled={["permission", "recording", "stopping", "processing"].includes(state)}
            >
              <option value="ja">Japanese</option>
              <option value="fr">French</option>
            </select>
          </label>
        </div>

        <Waveform
          analyser={analyser}
          active={state === "recording"}
          frozen={state === "stopping" || state === "processing"}
        />

        <button
          className={`microphone microphone--${state}`}
          onClick={handleMicrophone}
          disabled={["permission", "stopping", "processing"].includes(state)}
          aria-label={state === "recording" ? "Stop recording" : "Start recording"}
        >
          <span className="mic-icon" aria-hidden="true" />
        </button>
        <p className={`status status--${state}`}>{status}</p>

        {result && (
          <section className="result" aria-label="Translation result">
            <p className="japanese" lang={result.target_language}>
              {result.translated_text}
            </p>
            <p className="reference">{result.english_reference}</p>
            <p className="latency">
              {(result.timings.inference_ms / 1000).toFixed(1)}s warm inference
              {result.cold_start ? " · cold start" : ""}
            </p>
            <audio
              ref={audioRef}
              controls
              onEnded={releaseBlobUrl}
              aria-label="Translated audio"
            />
            {playbackBlocked && (
              <button className="play-fallback" onClick={() => void retryPlayback()}>
                Play translated audio
              </button>
            )}
          </section>
        )}
      </section>
    </main>
  );
}
