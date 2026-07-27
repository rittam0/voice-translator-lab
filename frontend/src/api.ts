export type SourceLanguage = "en" | "hi" | "ory";
export type TargetLanguage = "ja" | "fr";

export interface TranslationResult {
  request_id: string;
  source_language: SourceLanguage;
  target_language: TargetLanguage;
  source_transcript: string | null;
  translated_text: string;
  english_reference: string;
  audio_mime_type: "audio/wav";
  audio_url: string;
  models: Record<string, string>;
  timings: {
    inference_ms: number;
    [key: string]: number;
  };
  cold_start: boolean;
}

export class ApiError extends Error {
  constructor(
    message: string,
    readonly stage?: string,
    readonly code?: string,
  ) {
    super(message);
  }
}

const API_URL = (import.meta.env.VITE_API_URL || window.location.origin).replace(
  /\/$/,
  "",
);
const API_TOKEN = import.meta.env.VITE_API_TOKEN ?? "";
const authHeaders: Record<string, string> = API_TOKEN
  ? { Authorization: `Bearer ${API_TOKEN}` }
  : {};

export async function translate(
  audio: Blob,
  source: SourceLanguage,
  target: TargetLanguage,
  signal: AbortSignal,
): Promise<TranslationResult> {
  const form = new FormData();
  const extension = audio.type.includes("ogg")
    ? "ogg"
    : audio.type.includes("mp4")
      ? "m4a"
      : audio.type.includes("wav")
        ? "wav"
        : "webm";
  form.append("audio", audio, `recording.${extension}`);
  form.append("source_language", source);
  form.append("target_language", target);
  let response: Response;
  try {
    response = await fetch(`${API_URL}/api/translate`, {
      method: "POST",
      body: form,
      headers: authHeaders,
      signal,
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      throw new ApiError("Translation timed out. Please try a shorter recording.");
    }
    throw new ApiError("The translator backend is unavailable.");
  }
  let body: unknown;
  try {
    body = await response.json();
  } catch {
    throw new ApiError("The backend returned a malformed response.");
  }
  if (!response.ok) {
    const detail = (body as { detail?: unknown }).detail;
    if (detail && typeof detail === "object") {
      const error = detail as Record<string, unknown>;
      throw new ApiError(
        String(error.message ?? "Translation failed."),
        String(error.stage ?? ""),
        String(error.code ?? ""),
      );
    }
    throw new ApiError("Translation failed.");
  }
  const result = body as Partial<TranslationResult>;
  if (
    !["ja", "fr"].includes(String(result.target_language)) ||
    typeof result.translated_text !== "string" ||
    typeof result.english_reference !== "string" ||
    typeof result.audio_url !== "string" ||
    typeof result.timings?.inference_ms !== "number"
  ) {
    throw new ApiError("The backend returned an incomplete response.");
  }
  return result as TranslationResult;
}

export async function fetchAudioUrl(path: string, signal: AbortSignal): Promise<string> {
  const url = path.startsWith("http") ? path : `${API_URL}${path}`;
  const response = await fetch(url, { signal, headers: authHeaders });
  if (!response.ok) throw new ApiError("Translated audio could not be downloaded.");
  return URL.createObjectURL(await response.blob());
}
