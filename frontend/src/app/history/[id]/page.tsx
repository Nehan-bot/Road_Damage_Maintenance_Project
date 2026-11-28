"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import {
  ArrowLeft,
  CalendarClock,
  ImageIcon,
  Loader2,
  MapPin,
  ScanSearch,
} from "lucide-react";

import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

type BoundingBox = {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
};

type Detection = {
  class_id: number;
  class_name: string;
  confidence: number;
  bbox: BoundingBox;
};

type Inspection = {
  id: number;
  filename: string;
  created_at: string;
  image_width: number;
  image_height: number;
  detection_count: number;
  detections: Detection[];
  image_path?: string | null;
  priority: string | null;
  priority_confidence: number | null;
  priority_probabilities: Record<string, number> | null;
};

type InspectionDetailPageProps = {
  params: Promise<{ id: string }>;
};

function priorityVariant(priority: string | null) {
  if (priority === "High") return "destructive" as const;
  if (priority === "Medium") return "secondary" as const;
  return "outline" as const;
}

export default function InspectionDetailPage({
  params,
}: InspectionDetailPageProps) {
  const [id, setId] = useState("");
  const [inspection, setInspection] = useState<Inspection | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function resolveParams() {
      const resolved = await params;
      if (!cancelled) {
        setId(resolved.id);
      }
    }

    resolveParams();

    return () => {
      cancelled = true;
    };
  }, [params]);

  useEffect(() => {
    if (!id) return;

    let cancelled = false;

    async function loadInspection() {
      try {
        setLoading(true);
        setError(null);

        const response = await fetch(
          `http://localhost:8000/inspections/${id}`
        );

        if (!response.ok) {
          if (response.status === 404) {
            throw new Error("Inspection not found.");
          }
          throw new Error("Could not load inspection details.");
        }

        const data: Inspection = await response.json();

        if (!cancelled) {
          setInspection(data);
        }
      } catch (err) {
        if (!cancelled) {
          setError(
            err instanceof Error
              ? err.message
              : "Unable to load inspection details."
          );
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    loadInspection();

    return () => {
      cancelled = true;
    };
  }, [id]);

  const imageUrl = id
    ? `http://localhost:8000/inspections/${id}/image`
    : "";

  return (
    <main className="min-h-screen bg-[#f6f7f9] p-6 md:p-10">
      <div className="mx-auto max-w-6xl space-y-8">
        <div className="flex items-center gap-4">
          <Link
            href="/history"
            aria-label="Back to inspection history"
            className="flex size-10 items-center justify-center rounded-lg border border-slate-200 bg-white transition hover:bg-slate-50"
          >
            <ArrowLeft className="size-4" />
          </Link>

          <div>
            <p className="text-sm text-slate-500">
              RoadSight AI / History / Inspection #{id || "..."}
            </p>
            <h1 className="mt-1 text-3xl font-semibold tracking-tight">
              Inspection details
            </h1>
          </div>
        </div>

        {loading && (
          <Card className="rounded-2xl border-slate-200 shadow-sm">
            <CardContent className="flex items-center justify-center gap-3 py-16 text-slate-500">
              <Loader2 className="size-5 animate-spin" />
              Loading inspection details...
            </CardContent>
          </Card>
        )}

        {!loading && error && (
          <Card className="rounded-2xl border-slate-200 shadow-sm">
            <CardContent className="py-12">
              <div className="rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-700">
                {error}
              </div>
              <Link
                href="/history"
                className="mt-4 inline-flex items-center rounded-lg bg-slate-900 px-4 py-2 text-sm font-medium text-white transition hover:bg-slate-700"
              >
                Back to history
              </Link>
            </CardContent>
          </Card>
        )}

        {!loading && !error && inspection && (
          <>
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-5">
              <Card className="rounded-2xl border-slate-200 shadow-sm">
                <CardContent className="p-5">
                  <p className="text-sm text-slate-500">
                    Detections
                  </p>
                  <p className="mt-2 text-3xl font-semibold">
                    {inspection.detection_count}
                  </p>
                </CardContent>
              </Card>

              <Card className="rounded-2xl border-slate-200 shadow-sm">
                <CardContent className="p-5">
                  <p className="text-sm text-slate-500">
                    Image dimensions
                  </p>
                  <p className="mt-2 text-2xl font-semibold">
                    {inspection.image_width} × {inspection.image_height}
                  </p>
                </CardContent>
              </Card>

              <Card className="rounded-2xl border-slate-200 shadow-sm">
                <CardContent className="p-5">
                  <p className="text-sm text-slate-500">
                    Inspection ID
                  </p>
                  <p className="mt-2 text-3xl font-semibold">
                    #{inspection.id}
                  </p>
                </CardContent>
              </Card>

              <Card className="rounded-2xl border-slate-200 shadow-sm">
                <CardContent className="p-5">
                  <p className="text-sm text-slate-500">
                    Estimated priority
                  </p>
                  <div className="mt-2">
                    <Badge variant={priorityVariant(inspection.priority)}>
                      {inspection.priority ?? "Unavailable"}
                    </Badge>
                  </div>
                </CardContent>
              </Card>

              <Card className="rounded-2xl border-slate-200 shadow-sm">
                <CardContent className="p-5">
                  <p className="text-sm text-slate-500">
                    Priority confidence
                  </p>
                  <p className="mt-2 text-3xl font-semibold">
                    {inspection.priority_confidence !== null
                      ? `${(
                          inspection.priority_confidence * 100
                        ).toFixed(0)}%`
                      : "—"}
                  </p>
                </CardContent>
              </Card>
            </div>

            <div className="grid gap-6 lg:grid-cols-[1.5fr_1fr]">
              <Card className="rounded-2xl border-slate-200 shadow-sm">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <ImageIcon className="size-5" />
                    Inspection image
                  </CardTitle>
                  <CardDescription>
                    Road image with YOLO damage detections.
                  </CardDescription>
                </CardHeader>

                <CardContent>
                  <div
                    className="relative w-full overflow-hidden rounded-xl border border-slate-200 bg-slate-100"
                    style={{
                      aspectRatio: `${inspection.image_width} / ${inspection.image_height}`,
                    }}
                  >
                    <img
                      src={imageUrl}
                      alt={`Road inspection image: ${inspection.filename}`}
                      className="absolute inset-0 h-full w-full object-contain"
                    />

                    <svg
                      viewBox={`0 0 ${inspection.image_width} ${inspection.image_height}`}
                      preserveAspectRatio="none"
                      className="absolute inset-0 h-full w-full"
                    >
                      {inspection.detections.map((detection, index) => {
                        const { x1, y1, x2, y2 } = detection.bbox;
                        const boxWidth = x2 - x1;
                        const boxHeight = y2 - y1;

                        const label = `${detection.class_name} ${(
                          detection.confidence * 100
                        ).toFixed(0)}%`;

                        return (
                          <g key={`${inspection.id}-box-${index}`}>
                            <rect
                              x={x1}
                              y={y1}
                              width={boxWidth}
                              height={boxHeight}
                              fill="none"
                              stroke="#22c55e"
                              strokeWidth={4}
                            />
                            <text
                              x={x1}
                              y={Math.max(y1 - 8, 18)}
                              fill="#ffffff"
                              fontSize={16}
                              fontWeight="bold"
                              paintOrder="stroke"
                              stroke="#166534"
                              strokeWidth={4}
                              strokeLinejoin="round"
                            >
                              {label}
                            </text>
                          </g>
                        );
                      })}
                    </svg>
                  </div>

                  <p className="mt-3 break-all text-sm text-slate-500">
                    {inspection.filename}
                  </p>
                </CardContent>
              </Card>

              <Card className="rounded-2xl border-slate-200 shadow-sm">
                <CardHeader>
                  <CardTitle>Inspection information</CardTitle>
                  <CardDescription>
                    Details recorded for this prediction.
                  </CardDescription>
                </CardHeader>

                <CardContent className="space-y-5">
                  <div>
                    <p className="text-sm text-slate-500">
                      Inspection date
                    </p>
                    <p className="mt-1 flex items-center gap-2 font-medium">
                      <CalendarClock className="size-4 text-slate-500" />
                      {new Date(
                        inspection.created_at
                      ).toLocaleString()}
                    </p>
                  </div>

                  <div>
                    <p className="text-sm text-slate-500">
                      Detected damage
                    </p>
                    <p className="mt-1 font-medium">
                      {inspection.detection_count} object(s)
                    </p>
                  </div>

                  <div className="rounded-xl bg-slate-50 p-4">
                    <div className="flex items-center gap-2 text-sm font-medium">
                      <MapPin className="size-4 text-slate-500" />
                      Location
                    </div>
                    <p className="mt-2 text-sm text-slate-500">
                      No GPS location is currently recorded for this image.
                    </p>
                  </div>

                  <div className="rounded-xl border border-amber-200 bg-amber-50 p-4">
                    <p className="text-sm font-medium text-amber-900">
                      Priority estimate limitation
                    </p>
                    <p className="mt-2 text-sm text-amber-800">
                      This is an exploratory machine-learning estimate based
                      on detected damage counts and confidence values. It
                      is not a confirmed engineering severity assessment.
                    </p>
                  </div>
                </CardContent>
              </Card>
            </div>

            <Card className="rounded-2xl border-slate-200 shadow-sm">
              <CardHeader>
                <CardTitle>Priority probability breakdown</CardTitle>
                <CardDescription>
                  Random Forest class probabilities for this inspection.
                </CardDescription>
              </CardHeader>

              <CardContent>
                {inspection.priority_probabilities ? (
                  <div className="space-y-4">
                    {["High", "Medium", "Low"].map((priority) => {
                      const probability =
                        inspection.priority_probabilities?.[priority] ?? 0;

                      return (
                        <div key={priority}>
                          <div className="mb-1 flex justify-between text-sm">
                            <span className="font-medium">
                              {priority}
                            </span>
                            <span className="text-slate-500">
                              {(probability * 100).toFixed(1)}%
                            </span>
                          </div>

                          <div className="h-3 overflow-hidden rounded-full bg-slate-100">
                            <div
                              className="h-full rounded-full bg-slate-800"
                              style={{
                                width: `${probability * 100}%`,
                              }}
                            />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                ) : (
                  <p className="text-sm text-slate-500">
                    No priority probability data is saved for this record.
                    This may be an inspection created before priority
                    integration.
                  </p>
                )}
              </CardContent>
            </Card>

            <Card className="rounded-2xl border-slate-200 shadow-sm">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <ScanSearch className="size-5" />
                  Detection results
                </CardTitle>
                <CardDescription>
                  Damage categories and YOLO confidence for each detected object.
                </CardDescription>
              </CardHeader>

              <CardContent>
                {inspection.detections.length === 0 ? (
                  <div className="py-10 text-center">
                    <p className="font-medium">
                      No road damage detected
                    </p>
                    <p className="mt-2 text-sm text-slate-500">
                      The model did not find damage above its confidence threshold.
                    </p>
                  </div>
                ) : (
                  <div className="space-y-4">
                    {inspection.detections.map((detection, index) => (
                      <div
                        key={`${inspection.id}-${index}`}
                        className="rounded-xl border border-slate-200 bg-white p-4"
                      >
                        <div className="flex flex-col justify-between gap-3 sm:flex-row sm:items-center">
                          <div>
                            <p className="font-medium capitalize">
                              {detection.class_name}
                            </p>
                            <p className="mt-1 text-xs text-slate-500">
                              Detection #{index + 1}
                            </p>
                          </div>

                          <Badge variant="secondary">
                            {(detection.confidence * 100).toFixed(1)}%
                            confidence
                          </Badge>
                        </div>

                        <div className="mt-4 grid grid-cols-2 gap-3 text-sm sm:grid-cols-4">
                          {(
                            [
                              ["X1", detection.bbox.x1],
                              ["Y1", detection.bbox.y1],
                              ["X2", detection.bbox.x2],
                              ["Y2", detection.bbox.y2],
                            ] as const
                          ).map(([label, value]) => (
                            <div
                              key={label}
                              className="rounded-lg bg-slate-50 p-3"
                            >
                              <p className="text-xs text-slate-500">
                                {label}
                              </p>
                              <p className="mt-1 font-medium">
                                {value.toFixed(1)}
                              </p>
                            </div>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>

            <div className="flex flex-wrap gap-3">
              <Link
                href="/history"
                className="inline-flex items-center rounded-lg border border-slate-200 bg-white px-4 py-2 text-sm font-medium transition hover:bg-slate-50"
              >
                <ArrowLeft className="mr-2 size-4" />
                Back to history
              </Link>

              <Link
                href="/inspections"
                className="inline-flex items-center rounded-lg bg-slate-900 px-4 py-2 text-sm font-medium text-white transition hover:bg-slate-700"
              >
                New inspection
              </Link>
            </div>
          </>
        )}
      </div>
    </main>
  );
}