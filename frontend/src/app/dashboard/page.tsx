"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import {
  Activity,
  AlertTriangle,
  Camera,
  CheckCircle2,
  ChevronRight,
  Clock,
  LayoutDashboard,
  MapPin,
  Settings,
  ShieldCheck,
  Upload,
} from "lucide-react";

import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";

const API_URL = "http://localhost:8000";

type Detection = {
  class_id?: number;
  class_name: string;
  confidence: number;
  bbox?: {
    x1: number;
    y1: number;
    x2: number;
    y2: number;
  };
};

type InspectionSummary = {
  id: number;
  filename: string;
  created_at: string;
  image_width?: number;
  image_height?: number;
  detection_count?: number;
  priority?: string | null;
  priority_confidence?: number | null;
};

type InspectionDetail = InspectionSummary & {
  detections?: Detection[];
};

type Inspection = InspectionSummary & {
  detections: Detection[];
};

const DAMAGE_CLASSES = [
  "longitudinal crack",
  "transverse crack",
  "alligator crack",
  "pothole",
];

const navigation = [
  { label: "Overview", icon: LayoutDashboard, href: "/dashboard" },
  { label: "Road inspections", icon: Camera, href: "/inspections" },
  { label: "Inspection history", icon: Clock, href: "/history" },
  { label: "Priority map", icon: MapPin, href: "/history" },
  { label: "Model monitoring", icon: Activity, href: "/dashboard" },
  { label: "Settings", icon: Settings, href: "/dashboard" },
];

function formatDate(dateString: string) {
  const date = new Date(dateString);

  if (Number.isNaN(date.getTime())) {
    return dateString;
  }

  return date.toLocaleString();
}

function priorityClass(priority?: string | null) {
  switch (priority?.toLowerCase()) {
    case "high":
      return "bg-red-100 text-red-700";
    case "medium":
      return "bg-amber-100 text-amber-700";
    case "low":
      return "bg-emerald-100 text-emerald-700";
    default:
      return "bg-slate-100 text-slate-600";
  }
}

const linkButtonClass =
  "inline-flex items-center justify-center gap-2 rounded-xl bg-slate-900 px-4 py-2.5 text-sm font-medium text-white transition hover:bg-slate-700";

const outlineLinkClass =
  "inline-flex items-center justify-center rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm font-medium text-slate-700 transition hover:bg-slate-50";

function DistributionBar({
  label,
  count,
  maxCount,
  colorClass,
}: {
  label: string;
  count: number;
  maxCount: number;
  colorClass: string;
}) {
  const width = maxCount > 0 ? (count / maxCount) * 100 : 0;

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between gap-3 text-sm">
        <span className="capitalize text-slate-600">{label}</span>
        <span className="font-semibold tabular-nums text-slate-900">
          {count}
        </span>
      </div>

      <div className="h-2.5 overflow-hidden rounded-full bg-slate-100">
        <div
          className={`h-full rounded-full transition-all duration-500 ${colorClass}`}
          style={{ width: `${width}%` }}
        />
      </div>
    </div>
  );
}

export default function DashboardPage() {
  const [inspections, setInspections] = useState<Inspection[]>([]);
  const [loading, setLoading] = useState(true);
  const [apiError, setApiError] = useState("");

  useEffect(() => {
    let cancelled = false;

    async function loadInspections() {
      try {
        setLoading(true);
        setApiError("");

        // First get the inspection summaries.
        const response = await fetch(`${API_URL}/inspections`);

        if (!response.ok) {
          throw new Error("Could not load inspection history.");
        }

        const summaries: InspectionSummary[] = await response.json();

        if (!Array.isArray(summaries)) {
          throw new Error("The API returned an unexpected response.");
        }

        // Then get full detections for each inspection.
        // The list endpoint only contains detection_count,
        // while the detail endpoint contains the detections array.
        const detailedInspections = await Promise.all(
          summaries.map(async (summary): Promise<Inspection> => {
            try {
              const detailResponse = await fetch(
                `${API_URL}/inspections/${summary.id}`
              );

              if (!detailResponse.ok) {
                return {
                  ...summary,
                  detections: [],
                };
              }

              const detail: InspectionDetail =
                await detailResponse.json();

              return {
                ...summary,
                ...detail,
                detections: Array.isArray(detail.detections)
                  ? detail.detections
                  : [],
              };
            } catch {
              // Keep the summary record even if its detail request fails.
              return {
                ...summary,
                detections: [],
              };
            }
          })
        );

        if (!cancelled) {
          setInspections(detailedInspections);
        }
      } catch (error) {
        if (!cancelled) {
          setApiError(
            error instanceof Error
              ? error.message
              : "Could not connect to the backend API."
          );
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    loadInspections();

    return () => {
      cancelled = true;
    };
  }, []);

  const analytics = useMemo(() => {
    const classCounts: Record<string, number> = {};

    DAMAGE_CLASSES.forEach((className) => {
      classCounts[className] = 0;
    });

    let totalDetections = 0;

    inspections.forEach((inspection) => {
      // Use detection_count from the summary response when available.
      // This is the reliable total from the list endpoint.
      totalDetections +=
        typeof inspection.detection_count === "number"
          ? inspection.detection_count
          : inspection.detections.length;

      // Use the full detections array to calculate class distribution.
      inspection.detections.forEach((detection) => {
        const className = detection.class_name?.toLowerCase();

        if (className && className in classCounts) {
          classCounts[className] += 1;
        }
      });
    });

    const highPriority = inspections.filter(
      (inspection) => inspection.priority?.toLowerCase() === "high"
    ).length;

    const mediumPriority = inspections.filter(
      (inspection) => inspection.priority?.toLowerCase() === "medium"
    ).length;

    const lowPriority = inspections.filter(
      (inspection) => inspection.priority?.toLowerCase() === "low"
    ).length;

    const mostCommonClass = Object.entries(classCounts).sort(
      (a, b) => b[1] - a[1]
    )[0];

    return {
      totalInspections: inspections.length,
      totalDetections,
      highPriority,
      mediumPriority,
      lowPriority,
      classCounts,
      mostCommonClass:
        mostCommonClass && mostCommonClass[1] > 0
          ? mostCommonClass[0]
          : "No detections yet",
      mostCommonCount:
        mostCommonClass && mostCommonClass[1] > 0
          ? mostCommonClass[1]
          : 0,
    };
  }, [inspections]);

  const recentInspections = inspections.slice(0, 5);

  const maxDamageCount = Math.max(
    0,
    ...Object.values(analytics.classCounts)
  );

  const maxPriorityCount = Math.max(
    analytics.highPriority,
    analytics.mediumPriority,
    analytics.lowPriority,
    0
  );

  return (
    <main className="min-h-screen bg-[#f6f7f9] text-slate-900">
      <div className="flex min-h-screen">
        <aside className="hidden w-64 shrink-0 border-r border-slate-200 bg-white p-5 md:flex md:flex-col">
          <div className="flex items-center gap-3 px-2 py-3">
            <div className="flex size-10 items-center justify-center rounded-xl bg-slate-900 text-white">
              <ShieldCheck className="size-5" />
            </div>

            <div>
              <p className="text-lg font-bold tracking-tight">
                RoadSight AI
              </p>
              <p className="text-xs text-slate-500">
                Road intelligence
              </p>
            </div>
          </div>

          <p className="mb-3 mt-10 px-3 text-xs font-semibold uppercase tracking-wider text-slate-400">
            Workspace
          </p>

          <nav className="space-y-1">
            {navigation.map((item) => {
              const Icon = item.icon;

              return (
                <Link
                  key={item.label}
                  href={item.href}
                  className={`flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-sm font-medium transition ${
                    item.label === "Overview"
                      ? "bg-slate-900 text-white"
                      : "text-slate-600 hover:bg-slate-100"
                  }`}
                >
                  <Icon className="size-4" />
                  {item.label}
                </Link>
              );
            })}
          </nav>

          <div className="mt-auto rounded-2xl bg-slate-50 p-4">
            <div className="mb-2 flex items-center gap-2 text-sm font-semibold">
              {apiError ? (
                <AlertTriangle className="size-4 text-amber-600" />
              ) : (
                <CheckCircle2 className="size-4 text-emerald-600" />
              )}
              System status
            </div>

            <p className="text-xs leading-5 text-slate-500">
              {apiError
                ? "Backend API is not reachable. Start FastAPI to load dashboard data."
                : "Dashboard connected to the inspection history API."}
            </p>
          </div>
        </aside>

        <section className="min-w-0 flex-1">
          <header className="flex h-20 items-center justify-between border-b border-slate-200 bg-white px-6 md:px-10">
            <div>
              <p className="text-sm text-slate-500">
                Workspace / Overview
              </p>
              <h1 className="text-xl font-semibold tracking-tight">
                Road condition dashboard
              </h1>
            </div>

            <Badge
              variant="secondary"
              className="gap-2 rounded-full px-3 py-1.5"
            >
              <span
                className={`size-2 rounded-full ${
                  apiError ? "bg-amber-500" : "bg-emerald-500"
                }`}
              />
              {apiError ? "API disconnected" : "API connected"}
            </Badge>
          </header>

          <div className="mx-auto max-w-7xl space-y-8 p-6 md:p-10">
            <div className="flex flex-col justify-between gap-4 sm:flex-row sm:items-end">
              <div>
                <p className="text-sm font-medium text-slate-500">
                  Road Operations
                </p>

                <h2 className="mt-2 text-3xl font-semibold tracking-tight">
                  Dashboard overview
                </h2>

                <p className="mt-2 text-slate-500">
                  Monitor detected road damage and review maintenance
                  priority estimates.
                </p>
              </div>

              <Link href="/inspections" className={linkButtonClass}>
                <Upload className="size-4" />
                New inspection
              </Link>
            </div>

            {apiError && (
              <div className="rounded-xl border border-amber-200 bg-amber-50 p-4 text-sm text-amber-800">
                {apiError}
                <p className="mt-1 text-xs">
                  Make sure the backend is running at {API_URL}.
                </p>
              </div>
            )}

            {/* Summary cards */}
            <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
              <Card className="rounded-2xl border-slate-200 shadow-sm">
                <CardHeader className="flex flex-row items-start justify-between space-y-0 pb-3">
                  <div>
                    <CardDescription>Road inspections</CardDescription>
                    <CardTitle className="mt-2 text-3xl font-semibold">
                      {loading ? "—" : analytics.totalInspections}
                    </CardTitle>
                  </div>

                  <div className="rounded-xl bg-slate-100 p-2.5">
                    <Camera className="size-5 text-slate-600" />
                  </div>
                </CardHeader>

                <CardContent>
                  <p className="text-sm font-medium text-slate-600">
                    Saved in database
                  </p>
                  <p className="mt-1 text-xs text-slate-500">
                    Total image inspections recorded
                  </p>
                </CardContent>
              </Card>

              <Card className="rounded-2xl border-slate-200 shadow-sm">
                <CardHeader className="flex flex-row items-start justify-between space-y-0 pb-3">
                  <div>
                    <CardDescription>Damage detected</CardDescription>
                    <CardTitle className="mt-2 text-3xl font-semibold">
                      {loading ? "—" : analytics.totalDetections}
                    </CardTitle>
                  </div>

                  <div className="rounded-xl bg-slate-100 p-2.5">
                    <AlertTriangle className="size-5 text-slate-600" />
                  </div>
                </CardHeader>

                <CardContent>
                  <p className="text-sm font-medium text-slate-600">
                    YOLO detections
                  </p>
                  <p className="mt-1 text-xs text-slate-500">
                    Total detected damage instances
                  </p>
                </CardContent>
              </Card>

              <Card className="rounded-2xl border-slate-200 shadow-sm">
                <CardHeader className="flex flex-row items-start justify-between space-y-0 pb-3">
                  <div>
                    <CardDescription>High priority</CardDescription>
                    <CardTitle className="mt-2 text-3xl font-semibold">
                      {loading ? "—" : analytics.highPriority}
                    </CardTitle>
                  </div>

                  <div className="rounded-xl bg-slate-100 p-2.5">
                    <ShieldCheck className="size-5 text-slate-600" />
                  </div>
                </CardHeader>

                <CardContent>
                  <p className="text-sm font-medium text-red-600">
                    Requires review
                  </p>
                  <p className="mt-1 text-xs text-slate-500">
                    Exploratory priority estimates
                  </p>
                </CardContent>
              </Card>
            </div>

            {/* Analytics charts */}
            <div className="grid gap-6 xl:grid-cols-2">
              <Card className="rounded-2xl border-slate-200 shadow-sm">
                <CardHeader>
                  <CardTitle>Damage-class distribution</CardTitle>
                  <CardDescription>
                    Total detected instances by damage category
                  </CardDescription>
                </CardHeader>

                <CardContent className="space-y-5">
                  {DAMAGE_CLASSES.map((className) => (
                    <DistributionBar
                      key={className}
                      label={className}
                      count={
                        loading
                          ? 0
                          : analytics.classCounts[className]
                      }
                      maxCount={maxDamageCount}
                      colorClass="bg-blue-600"
                    />
                  ))}

                  {!loading && analytics.totalDetections === 0 && (
                    <p className="pt-2 text-xs text-slate-500">
                      No damage detections have been recorded yet.
                    </p>
                  )}
                </CardContent>
              </Card>

              <Card className="rounded-2xl border-slate-200 shadow-sm">
                <CardHeader>
                  <CardTitle>Priority distribution</CardTitle>
                  <CardDescription>
                    Inspections grouped by model priority estimate
                  </CardDescription>
                </CardHeader>

                <CardContent className="space-y-5">
                  <DistributionBar
                    label="High"
                    count={loading ? 0 : analytics.highPriority}
                    maxCount={maxPriorityCount}
                    colorClass="bg-red-500"
                  />

                  <DistributionBar
                    label="Medium"
                    count={loading ? 0 : analytics.mediumPriority}
                    maxCount={maxPriorityCount}
                    colorClass="bg-amber-500"
                  />

                  <DistributionBar
                    label="Low"
                    count={loading ? 0 : analytics.lowPriority}
                    maxCount={maxPriorityCount}
                    colorClass="bg-emerald-500"
                  />

                  <Separator />

                  <p className="text-xs leading-5 text-slate-500">
                    Priority values are exploratory model estimates and
                    should be reviewed by maintenance personnel.
                  </p>
                </CardContent>
              </Card>
            </div>

            {/* Insights and recent records */}
            <div className="grid gap-6 xl:grid-cols-[1fr_1.5fr]">
              <Card className="rounded-2xl border-slate-200 shadow-sm">
                <CardHeader>
                  <CardTitle>Inspection insights</CardTitle>
                  <CardDescription>
                    Summary of saved inspection results
                  </CardDescription>
                </CardHeader>

                <CardContent className="space-y-4">
                  <div className="rounded-xl bg-blue-50 p-4">
                    <p className="text-xs font-medium text-blue-700">
                      Most detected damage
                    </p>

                    <p className="mt-2 text-lg font-semibold capitalize text-slate-900">
                      {loading
                        ? "Loading..."
                        : analytics.mostCommonClass}
                    </p>

                    <p className="mt-1 text-sm text-slate-600">
                      {loading ? "—" : analytics.mostCommonCount} detections
                    </p>
                  </div>

                  <div className="flex items-center justify-between rounded-xl bg-slate-50 p-4">
                    <div>
                      <p className="text-sm font-medium">
                        Average detections per inspection
                      </p>
                      <p className="mt-1 text-xs text-slate-500">
                        Across saved records
                      </p>
                    </div>

                    <p className="text-xl font-semibold tabular-nums">
                      {loading || analytics.totalInspections === 0
                        ? "0.0"
                        : (
                            analytics.totalDetections /
                            analytics.totalInspections
                          ).toFixed(1)}
                    </p>
                  </div>

                  <div className="flex items-center justify-between rounded-xl bg-slate-50 p-4">
                    <div>
                      <p className="text-sm font-medium">
                        Inspections with detections
                      </p>
                      <p className="mt-1 text-xs text-slate-500">
                        At least one detected defect
                      </p>
                    </div>

                    <p className="text-xl font-semibold tabular-nums">
                      {loading
                        ? "—"
                        : inspections.filter(
                            (inspection) =>
                              (inspection.detection_count ?? 0) > 0
                          ).length}
                    </p>
                  </div>
                </CardContent>
              </Card>

              <Card className="rounded-2xl border-slate-200 shadow-sm">
                <CardHeader>
                  <div className="flex items-center justify-between gap-3">
                    <div>
                      <CardTitle>Recent inspections</CardTitle>
                      <CardDescription>
                        Latest saved road assessment activity
                      </CardDescription>
                    </div>

                    <Link href="/history" className={outlineLinkClass}>
                      View history
                    </Link>
                  </div>
                </CardHeader>

                <CardContent>
                  {loading ? (
                    <p className="py-8 text-center text-sm text-slate-500">
                      Loading inspection records...
                    </p>
                  ) : recentInspections.length === 0 ? (
                    <div className="rounded-xl border border-dashed border-slate-300 p-8 text-center">
                      <div className="mx-auto mb-3 flex size-12 items-center justify-center rounded-xl bg-slate-100">
                        <Camera className="size-5 text-slate-500" />
                      </div>

                      <p className="font-medium">
                        No inspections recorded yet
                      </p>

                      <p className="mt-1 text-sm text-slate-500">
                        Upload a road image to create your first inspection.
                      </p>

                      <Link
                        href="/inspections"
                        className={`${linkButtonClass} mt-4`}
                      >
                        Start inspection
                      </Link>
                    </div>
                  ) : (
                    <div className="space-y-3">
                      {recentInspections.map((inspection) => (
                        <Link
                          key={inspection.id}
                          href={`/history/${inspection.id}`}
                          className="flex items-center gap-3 rounded-xl border border-slate-200 p-3 transition hover:bg-slate-50"
                        >
                          <div className="flex size-10 shrink-0 items-center justify-center rounded-lg bg-slate-100">
                            <Camera className="size-4 text-slate-600" />
                          </div>

                          <div className="min-w-0 flex-1">
                            <p className="truncate text-sm font-medium">
                              {inspection.filename}
                            </p>

                            <p className="mt-1 text-xs text-slate-500">
                              {formatDate(inspection.created_at)}
                            </p>

                            <p className="mt-1 flex items-center gap-1 text-xs text-slate-500">
                              <Clock className="size-3" />
                              {inspection.detection_count ?? 0} detections
                            </p>
                          </div>

                          <Badge className={priorityClass(inspection.priority)}>
                            {inspection.priority ?? "Unknown"}
                          </Badge>

                          <ChevronRight className="size-4 shrink-0 text-slate-400" />
                        </Link>
                      ))}
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>

            <p className="text-center text-xs text-slate-400">
              RoadSight AI · Computer vision for smarter road maintenance
            </p>
          </div>
        </section>
      </div>
    </main>
  );
}