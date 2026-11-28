
"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { ArrowLeft, Camera, Clock, Loader2 } from "lucide-react";

type Inspection = {
  id: number;
  filename: string;
  created_at: string;
  detections: {
    class_name: string;
    confidence: number;
  }[];
  priority?: string | null;
};

export default function HistoryPage() {
  const [inspections, setInspections] = useState<Inspection[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    async function loadHistory() {
      try {
        const response = await fetch("http://localhost:8000/inspections");

        if (!response.ok) {
          throw new Error("Unable to load inspection history.");
        }

        const data = await response.json();
        setInspections(data);
      } catch (err) {
        setError(
          err instanceof Error
            ? err.message
            : "Could not connect to the backend."
        );
      } finally {
        setLoading(false);
      }
    }

    loadHistory();
  }, []);

  return (
    <main className="min-h-screen bg-slate-50 p-6 text-slate-900 md:p-10">
      <div className="mx-auto max-w-5xl">
        <Link
          href="/dashboard"
          className="mb-6 inline-flex items-center gap-2 text-sm text-slate-600 hover:text-slate-900"
        >
          <ArrowLeft className="size-4" />
          Back to dashboard
        </Link>

        <div className="mb-8">
          <h1 className="text-3xl font-bold tracking-tight">
            Inspection history
          </h1>
          <p className="mt-2 text-slate-500">
            View your saved road image inspections and analysis results.
          </p>
        </div>

        {loading && (
          <div className="flex items-center gap-3 rounded-xl border bg-white p-6 text-slate-600">
            <Loader2 className="size-5 animate-spin" />
            Loading inspection records...
          </div>
        )}

        {error && (
          <div className="rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-700">
            {error}
          </div>
        )}

        {!loading && !error && inspections.length === 0 && (
          <div className="rounded-2xl border border-dashed border-slate-300 bg-white p-10 text-center">
            <Camera className="mx-auto mb-4 size-10 text-slate-400" />
            <h2 className="font-semibold">No inspections yet</h2>
            <p className="mt-2 text-sm text-slate-500">
              Analyse a road image to create your first inspection record.
            </p>
            <Link
              href="/inspections"
              className="mt-5 inline-flex rounded-xl bg-slate-900 px-5 py-3 text-sm font-semibold text-white hover:bg-slate-700"
            >
              Analyse an image
            </Link>
          </div>
        )}

        {!loading && !error && inspections.length > 0 && (
          <div className="space-y-3">
            {inspections.map((inspection) => (
              <Link
                key={inspection.id}
                href={`/history/${inspection.id}`}
                className="flex flex-col gap-3 rounded-2xl border border-slate-200 bg-white p-5 transition hover:border-slate-300 hover:shadow-sm sm:flex-row sm:items-center"
              >
                <div className="flex size-11 shrink-0 items-center justify-center rounded-xl bg-slate-100">
                  <Camera className="size-5 text-slate-600" />
                </div>

                <div className="min-w-0 flex-1">
                  <h2 className="truncate font-semibold">
                    {inspection.filename}
                  </h2>
                  <p className="mt-1 text-sm text-slate-500">
                    {inspection.detections?.length ?? 0} damage detections
                  </p>
                  <p className="mt-1 flex items-center gap-1 text-xs text-slate-400">
                    <Clock className="size-3" />
                    {new Date(inspection.created_at).toLocaleString()}
                  </p>
                </div>

                <span className="w-fit rounded-full bg-slate-100 px-3 py-1.5 text-xs font-medium text-slate-700">
                  {inspection.priority ?? "Priority not available"}
                </span>
              </Link>
            ))}
          </div>
        )}
      </div>
    </main>
  );
}
