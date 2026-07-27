const recordButton = document.querySelector("#record");
const statusText = document.querySelector("#status");
const result = document.querySelector("#result");
const source = document.querySelector("#source");
const target = document.querySelector("#target");
let recorder;
let chunks = [];

document.querySelector("#swap").addEventListener("click", () => {
  [source.value, target.value] = [target.value, source.value];
});
source.addEventListener("change", () => { target.value = source.value === "en" ? "hi" : "en"; });
target.addEventListener("change", () => { source.value = target.value === "en" ? "hi" : "en"; });

recordButton.addEventListener("click", async () => {
  if (recorder?.state === "recording") {
    recorder.stop();
    return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    recorder = new MediaRecorder(stream);
    chunks = [];
    recorder.ondataavailable = event => chunks.push(event.data);
    recorder.onstop = () => submit(new Blob(chunks, { type: recorder.mimeType }), stream);
    recorder.start();
    recordButton.classList.add("active");
    recordButton.querySelector("span").textContent = "Stop & translate";
    statusText.textContent = "Recording… speak naturally";
    result.hidden = true;
  } catch (error) {
    statusText.textContent = `Microphone unavailable: ${error.message}`;
  }
});

async function submit(blob, stream) {
  stream.getTracks().forEach(track => track.stop());
  recordButton.disabled = true;
  recordButton.classList.remove("active");
  recordButton.querySelector("span").textContent = "Start recording";
  statusText.textContent = "Running ASR → translation → voice cloning…";
  const form = new FormData();
  form.append("audio", blob, "recording.webm");
  form.append("source_language", source.value);
  form.append("target_language", target.value);
  try {
    const response = await fetch("/api/translate", { method: "POST", body: form });
    const body = await response.json();
    if (!response.ok) {
      const detail = body.detail;
      throw new Error(typeof detail === "object" ? `${detail.stage}: ${detail.message}` : detail);
    }
    document.querySelector("#transcript").textContent = body.transcript;
    document.querySelector("#translation").textContent = body.translation;
    document.querySelector("#playback").src = body.audio_url;
    for (const [id, key] of [["asr", "asr_ms"], ["nmt", "translation_ms"], ["tts", "tts_ms"], ["total", "total_ms"]]) {
      document.querySelector(`#${id}`).textContent = `${Math.round(body.timings[key])} ms`;
    }
    result.hidden = false;
    statusText.textContent = "Translation complete";
  } catch (error) {
    statusText.textContent = `Could not translate — ${error.message}`;
  } finally {
    recordButton.disabled = false;
  }
}
