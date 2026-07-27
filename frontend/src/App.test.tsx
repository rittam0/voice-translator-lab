import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import App from "./App";

class FakeAnalyser {
  fftSize = 0;
  smoothingTimeConstant = 0;
  frequencyBinCount = 64;
  getByteTimeDomainData(values: Uint8Array) {
    values.fill(128);
  }
}

class FakeAudioContext {
  createMediaStreamSource() {
    return { connect: vi.fn() };
  }
  createAnalyser() {
    return new FakeAnalyser();
  }
  close = vi.fn().mockResolvedValue(undefined);
}

class FakeMediaRecorder {
  static isTypeSupported = vi.fn(() => true);
  state: RecordingState = "inactive";
  mimeType = "audio/webm;codecs=opus";
  ondataavailable: ((event: BlobEvent) => void) | null = null;
  onstop: (() => void) | null = null;
  onerror: (() => void) | null = null;
  constructor(
    public stream: MediaStream,
    public options?: MediaRecorderOptions,
  ) {}
  start() {
    this.state = "recording";
  }
  stop() {
    this.ondataavailable?.({ data: new Blob(["voice"], { type: this.mimeType }) } as BlobEvent);
    this.state = "inactive";
    this.onstop?.();
  }
}

const track = { stop: vi.fn() };
const stream = { getTracks: () => [track] } as unknown as MediaStream;
const successfulResponse = {
  request_id: "request-1",
  source_language: "en",
  target_language: "ja",
  source_transcript: "Hello",
  translated_text: "こんにちは",
  english_reference: "Hello",
  audio_mime_type: "audio/wav",
  audio_url: "/api/audio/request-1.wav",
  models: {},
  timings: { inference_ms: 3210 },
  cold_start: false,
};
let fakeClock = 0;

beforeEach(() => {
  vi.clearAllMocks();
  Object.defineProperty(window, "MediaRecorder", {
    configurable: true,
    value: FakeMediaRecorder,
  });
  Object.defineProperty(window, "AudioContext", {
    configurable: true,
    value: FakeAudioContext,
  });
  Object.defineProperty(navigator, "mediaDevices", {
    configurable: true,
    value: { getUserMedia: vi.fn().mockResolvedValue(stream) },
  });
  fakeClock = 0;
  vi.spyOn(performance, "now").mockImplementation(() => fakeClock);
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve(successfulResponse),
      blob: () => Promise.resolve(new Blob(["RIFF"], { type: "audio/wav" })),
    }),
  );
});

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

async function recordAndStop() {
  const user = userEvent.setup();
  await user.click(screen.getByRole("button", { name: "Start recording" }));
  await screen.findByRole("button", { name: "Stop recording" });
  fakeClock = 4000;
  await user.click(screen.getByRole("button", { name: "Stop recording" }));
}

test("offers English/Hindi/Odia and Japanese/French", async () => {
  render(<App />);
  const selector = screen.getByLabelText("Speaking in");
  expect(selector).toHaveValue("en");
  await userEvent.selectOptions(selector, "hi");
  expect(selector).toHaveValue("hi");
  await userEvent.selectOptions(selector, "ory");
  expect(selector).toHaveValue("ory");
  expect(screen.getByRole("option", { name: "Odia · Experimental" })).toBeVisible();
  const target = screen.getByLabelText("Translate to");
  await userEvent.selectOptions(target, "fr");
  expect(target).toHaveValue("fr");
});

test("moves through recording and renders a successful Japanese result", async () => {
  render(<App />);
  await recordAndStop();
  expect(await screen.findByText("こんにちは")).toBeVisible();
  expect(screen.getByText("Hello")).toBeVisible();
  expect(screen.getByText("3.2s warm inference")).toBeVisible();
  await waitFor(() => expect(HTMLMediaElement.prototype.play).toHaveBeenCalled());
  expect(fetch).toHaveBeenCalledTimes(2);
});

test("shows one-tap fallback when automatic playback is blocked", async () => {
  vi.mocked(HTMLMediaElement.prototype.play)
    .mockRejectedValueOnce(new DOMException("blocked"))
    .mockResolvedValueOnce(undefined);
  render(<App />);
  await recordAndStop();
  const fallback = await screen.findByRole("button", {
    name: "Play translated audio",
  });
  await userEvent.click(fallback);
  expect(HTMLMediaElement.prototype.play).toHaveBeenCalledTimes(2);
});

test("handles microphone denial", async () => {
  vi.mocked(navigator.mediaDevices.getUserMedia).mockRejectedValue(
    new DOMException("denied", "NotAllowedError"),
  );
  render(<App />);
  await userEvent.click(screen.getByRole("button", { name: "Start recording" }));
  expect(await screen.findByText(/Microphone access was denied/)).toBeVisible();
});

test("renders structured API failure", async () => {
  vi.mocked(fetch).mockResolvedValue({
    ok: false,
    json: () =>
      Promise.resolve({
        detail: {
          stage: "voice_conversion",
          code: "conversion_failed",
          message: "Best-effort voice conversion failed.",
        },
      }),
  } as Response);
  render(<App />);
  await recordAndStop();
  expect(
    await screen.findByText(/voice_conversion: Best-effort voice conversion failed/),
  ).toBeVisible();
});

test("keeps a frozen waveform and loading state while backend is pending", async () => {
  vi.mocked(fetch).mockReturnValue(new Promise(() => {}));
  render(<App />);
  await recordAndStop();
  expect(await screen.findByText(/Transcribing and creating Japanese speech/)).toBeVisible();
  expect(screen.getByLabelText("Live microphone waveform")).toHaveClass(
    "waveform--frozen",
  );
  expect(screen.getByRole("button", { name: "Start recording" })).toBeDisabled();
});

test("shows backend unavailable instead of a result", async () => {
  vi.mocked(fetch).mockRejectedValue(new TypeError("network down"));
  render(<App />);
  await recordAndStop();
  expect(await screen.findByText("The translator backend is unavailable.")).toBeVisible();
  expect(screen.queryByLabelText("Translation result")).not.toBeInTheDocument();
});

test("shows backend timeout without a fake success", async () => {
  vi.mocked(fetch).mockRejectedValue(new DOMException("aborted", "AbortError"));
  render(<App />);
  await recordAndStop();
  expect(
    await screen.findByText("Translation timed out. Please try a shorter recording."),
  ).toBeVisible();
  expect(screen.queryByLabelText("Translation result")).not.toBeInTheDocument();
});

test("rejects an empty recording before calling the backend", async () => {
  class EmptyMediaRecorder extends FakeMediaRecorder {
    stop() {
      this.state = "inactive";
      this.onstop?.();
    }
  }
  Object.defineProperty(window, "MediaRecorder", {
    configurable: true,
    value: EmptyMediaRecorder,
  });
  render(<App />);
  await recordAndStop();
  expect(await screen.findByText("No audio was recorded. Please try again.")).toBeVisible();
  expect(fetch).not.toHaveBeenCalled();
});

test("submits the selected target language", async () => {
  render(<App />);
  await userEvent.selectOptions(screen.getByLabelText("Translate to"), "fr");
  await recordAndStop();
  await screen.findByText("こんにちは");
  const request = vi.mocked(fetch).mock.calls[0][1];
  expect((request?.body as FormData).get("target_language")).toBe("fr");
});

test("revokes generated Blob URL after playback and on replacement", async () => {
  render(<App />);
  await recordAndStop();
  const player = await screen.findByLabelText("Translated audio");
  fireEvent.ended(player);
  expect(URL.revokeObjectURL).toHaveBeenCalledWith("blob:test-audio");
});
