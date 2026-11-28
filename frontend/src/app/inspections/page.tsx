
"use client";

import { useState } from "react";
import {
  AlertCircle,
  Camera,
  ImagePlus,
  Loader2,
  Upload,
  X,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

type Detection = {
  class_id: number;
  class_name: string;
  confidence: number;
  bbox: {
    x1: number;
    y1: number;
    x2: number;
    y2: number;
  };
};

type PredictionResponse = {
  filename: string;
  image_width: number;
  image_height: number;
  detection_count: number;
  detections: Detection[];
};

export default function InspectionsPage() {
  const [image, setImage] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function handleImageChange(
    event: React.ChangeEvent<HTMLInputElement>
  ) {
    const file = event.target.files?.[0];

    if (!file) return;

    if (!file.type.startsWith("image/")) {
      setError("Please select a valid image file.");
      return;
    }

    // Create a local browser preview of the selected image.
    const imageUrl = URL.createObjectURL(file);

    if (preview) {
      URL.revokeObjectURL(preview);
    }

    setImage(file);
    setPreview(imageUrl);
    setResult(null);
    setError(null);
  }

  function clearImage() {
    if (preview) {
      URL.revokeObjectURL(preview);
    }

    setImage(null);
    setPreview(null);
    setResult(null);
    setError(null);
  }

  async function runDetection() {
    if (!image) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      // FormData sends the image file as multipart form data.
      const formData = new FormData();
      formData.append("file", image);

      const response = await fetch(
        "http://localhost:8000/predict",
        {
          method: "POST",
          body: formData,
        }
      );

      if (!response.ok) {
        let message = "Prediction request failed.";

        try {
          const errorData = await response.json();
          message = errorData.detail || message;
        } catch {
          // Keep the default error message if response isn't JSON.
        }

        throw new Error(message);
      }

      const prediction: PredictionResponse =
        await response.json();

      setResult(prediction);
    } catch (err) {
      if (err instanceof Error) {
        setError(
          err.message.includes("Failed to fetch")
            ? "Cannot connect to the API. Make sure FastAPI is running at localhost:8000."
            : err.message
        );
      } else {
        setError("An unexpected error occurred.");
      }
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="min-h-screen bg-[#f6f7f9] p-6 md:p-10">
      <div className="mx-auto max-w-5xl space-y-8">
        <div>
          <p className="text-sm font-medium text-slate-500">
            RoadSight AI / Inspections
          </p>
          <h1 className="mt-2 text-3xl font-semibold tracking-tight">
            Road damage inspection
          </h1>
          <p className="mt-2 text-slate-500">
            Upload a road image and analyze it using your trained YOLO26s model.
          </p>
        </div>

        <Card className="rounded-2xl border-slate-200 shadow-sm">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Camera className="size-5" />
              Inspect a road image
            </CardTitle>
            <CardDescription>
              Choose a JPG, PNG, or other supported image.
            </CardDescription>
          </CardHeader>

          <CardContent className="space-y-5">
            {!preview ? (
              <label className="flex min-h-72 cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed border-slate-300 bg-slate-50 p-8 text-center transition hover:border-slate-500">
                <div className="mb-4 flex size-14 items-center justify-center rounded-2xl bg-white shadow-sm">
                  <ImagePlus className="size-6 text-slate-600" />
                </div>
                <p className="font-semibold">
                  Choose an image to inspect
                </p>
                <p className="mt-2 text-sm text-slate-500">
                  Select a road image from your computer
                </p>
                <input
                  type="file"
                  accept="image/*"
                  className="hidden"
                  onChange={handleImageChange}
                />
              </label>
            ) : (
              <>
                <div className="flex items-center justify-between gap-3">
                  <p className="min-w-0 truncate text-sm text-slate-600">
                    {image?.name}
                  </p>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={clearImage}
                    disabled={loading}
                    className="shrink-0 gap-2"
                  >
                    <X className="size-4" />
                    Remove
                  </Button>
                </div>

                {/* Image and SVG share the same coordinate system.
                    This keeps the boxes aligned with the original image. */}
                <div className="relative w-full overflow-hidden rounded-xl border bg-white">
                  <img
                    src={preview}
                    alt="Road image being inspected"
                    className="block h-auto w-full"
                  />

                  {result && (
                    <svg
                      className="absolute inset-0 h-full w-full"
                      viewBox={`0 0 ${result.image_width} ${result.image_height}`}
                      preserveAspectRatio="none"
                    >
                      {result.detections.map((detection, index) => {
                        const { x1, y1, x2, y2 } = detection.bbox;
                        const boxWidth = x2 - x1;
                        const boxHeight = y2 - y1;

                        return (
                          <g key={`${detection.class_id}-${index}`}>
                            <rect
                              x={x1}
                              y={y1}
                              width={boxWidth}
                              height={boxHeight}
                              fill="rgba(16, 185, 129, 0.08)"
                              stroke="#10b981"
                              strokeWidth="3"
                              vectorEffect="non-scaling-stroke"
                            />
                            <text
                              x={Math.max(4, x1)}
                              y={Math.max(18, y1 - 6)}
                              fill="#ffffff"
                              stroke="#047857"
                              strokeWidth="0.5"
                              fontSize="16"
                              fontWeight="600"
                            >
                              {detection.class_name}{" "}
                              {(detection.confidence * 100).toFixed(1)}%
                            </text>
                          </g>
                        );
                      })}
                    </svg>
                  )}
                </div>

                <Button
                  onClick={runDetection}
                  disabled={!image || loading}
                  className="w-full gap-2 rounded-xl"
                >
                  {loading ? (
                    <>
                      <Loader2 className="size-4 animate-spin" />
                      Analyzing image...
                    </>
                  ) : (
                    <>
                      <Upload className="size-4" />
                      Run damage detection
                    </>
                  )}
                </Button>
              </>
            )}

            {error && (
              <div className="flex items-start gap-3 rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-700">
                <AlertCircle className="mt-0.5 size-4 shrink-0" />
                <p>{error}</p>
              </div>
            )}

            {result && (
              <div className="space-y-4">
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div>
                    <h2 className="font-semibold">Detection results</h2>
                    <p className="text-sm text-slate-500">
                      {result.detection_count} damage object(s) detected
                    </p>
                  </div>

                  <Badge variant="secondary">
                    YOLO26s
                  </Badge>
                </div>

                {result.detection_count === 0 ? (
                  <div className="rounded-xl border border-slate-200 bg-slate-50 p-5 text-sm text-slate-600">
                    No damage objects were detected at the current confidence threshold.
                    This does not guarantee that the road is damage-free.
                  </div>
                ) : (
                  <div className="space-y-3">
                    {result.detections.map((detection, index) => (
                      <div
                        key={`${detection.class_id}-${index}`}
                        className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-slate-200 bg-white p-4"
                      >
                        <div>
                          <p className="font-medium capitalize">
                            {detection.class_name}
                          </p>
                          <p className="mt-1 text-xs text-slate-500">
                            Detection {index + 1}
                          </p>
                        </div>

                        <Badge variant="outline">
                          {(detection.confidence * 100).toFixed(1)}% confidence
                        </Badge>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </CardContent>
        </Card>

        <p className="text-center text-xs text-slate-400">
          RoadSight AI · Predictions are model outputs and require human review.
        </p>
      </div>
    </main>
  );
}
