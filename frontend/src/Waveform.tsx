import { useEffect, useRef } from "react";

interface WaveformProps {
  analyser: AnalyserNode | null;
  active: boolean;
  frozen: boolean;
}

export default function Waveform({ analyser, active, frozen }: WaveformProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const context = canvas.getContext("2d");
    if (!context) return;
    let animation = 0;
    const values = new Uint8Array(analyser?.frequencyBinCount ?? 64);

    const draw = () => {
      const ratio = window.devicePixelRatio || 1;
      const width = canvas.clientWidth;
      const height = canvas.clientHeight;
      if (canvas.width !== width * ratio || canvas.height !== height * ratio) {
        canvas.width = width * ratio;
        canvas.height = height * ratio;
      }
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
      context.clearRect(0, 0, width, height);
      if (analyser && active) analyser.getByteTimeDomainData(values);
      context.beginPath();
      context.strokeStyle = "#06B6D4";
      context.lineWidth = 2;
      context.shadowColor = "#06B6D4";
      context.shadowBlur = active ? 10 : 0;
      for (let index = 0; index < values.length; index += 1) {
        const x = (index / (values.length - 1)) * width;
        const amplitude = active ? (values[index] - 128) / 128 : 0;
        const y = height / 2 + amplitude * height * 0.42;
        if (index === 0) context.moveTo(x, y);
        else context.lineTo(x, y);
      }
      context.stroke();
      if (active) animation = requestAnimationFrame(draw);
    };
    draw();
    return () => cancelAnimationFrame(animation);
  }, [analyser, active]);

  return (
    <canvas
      ref={canvasRef}
      className={`waveform ${frozen ? "waveform--frozen" : ""}`}
      aria-label="Live microphone waveform"
    />
  );
}
