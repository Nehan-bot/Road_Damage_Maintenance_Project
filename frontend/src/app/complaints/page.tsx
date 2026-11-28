
"use client";

import { useEffect, useState, type FormEvent } from "react";
import Link from "next/link";
import {
  AlertTriangle,
  ArrowLeft,
  ClipboardList,
  Loader2,
  Send,
} from "lucide-react";

const API_URL = "http://localhost:8000";

type ComplaintResult = {
  id: number;
  complaint_text: string;
  damage_types?: string[];
  damage_type?: string;
  urgency: string;
  location_text: string | null;
  summary: string;
  damage_confidence?: number | null;
  urgency_confidence?: number | null;
  location_is_verified?: boolean;
  urgency_is_estimated?: boolean;
  nlp_model?: string | null;
  created_at: string;
};

function urgencyColor(urgency: string) {
  switch (urgency.toLowerCase()) {
    case "high":
      return "bg-red-100 text-red-700 border-red-200";
    case "medium":
      return "bg-amber-100 text-amber-700 border-amber-200";
    default:
      return "bg-green-100 text-green-700 border-green-200";
  }
}

function confidencePercent(value?: number | null) {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return "Not recorded";
  }

  return `${(value * 100).toFixed(1)}%`;
}

function getDamageTypes(complaint: ComplaintResult) {
  if (Array.isArray(complaint.damage_types)) {
    return complaint.damage_types;
  }

  if (complaint.damage_type) {
    return [complaint.damage_type];
  }

  return ["Not recorded"];
}

export default function ComplaintsPage() {
  const [text, setText] = useState("");
  const [result, setResult] = useState<ComplaintResult | null>(null);
  const [history, setHistory] = useState<ComplaintResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function loadHistory() {
    try {
      const response = await fetch(`${API_URL}/api/complaints`);

      if (!response.ok) {
        throw new Error("Could not load complaint history.");
      }

      const data = await response.json();
      setHistory(Array.isArray(data) ? data : []);
    } catch {
      setError(
        "Could not load complaint history. Check that the backend is running."
      );
    }
  }

  useEffect(() => {
    loadHistory();
  }, []);

  async function analyzeComplaint(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    if (text.trim().length < 10) {
      setError("Please enter a complaint with at least 10 characters.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch(`${API_URL}/api/complaints/analyze`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          text: text.trim(),
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Complaint analysis failed.");
      }

      setResult(data);
      setText("");
      await loadHistory();
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Could not analyze the complaint."
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="min-h-screen bg-slate-50 p-4 md:p-8">
      <div className="mx-auto max-w-6xl space-y-6">
        <header>
          <Link
            href="/dashboard"
            className="mb-3 inline-flex items-center gap-2 text-sm text-slate-500 hover:text-slate-900"
          >
            <ArrowLeft className="size-4" />
            Back to dashboard
          </Link>

          <p className="text-sm font-medium text-slate-500">
            RoadSight AI / NLP
          </p>

          <h1 className="mt-1 text-3xl font-semibold tracking-tight text-slate-900">
            Road complaint analysis
          </h1>

          <p className="mt-2 text-sm text-slate-500">
            Analyze road damage, estimated urgency and mentioned location from
            complaint text using trained NLP models.
          </p>
        </header>

        <div className="flex items-start gap-3 rounded-2xl border border-amber-200 bg-amber-50 p-4 text-amber-900">
          <AlertTriangle className="mt-0.5 size-5 shrink-0" />
          <div>
            <p className="font-semibold">NLP results are estimates</p>
            <p className="mt-1 text-sm leading-6">
              This prototype uses fine-tuned DistilBERT models to estimate
              damage type and urgency. Confidence scores represent the models'
              predicted probabilities and are not guarantees of correctness.
              Locations are extracted from complaint text and are not
              GPS-verified. Urgency is not an engineering safety assessment.
            </p>
          </div>
        </div>

        <div className="grid gap-6 lg:grid-cols-[1.2fr_0.8fr]">
          <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
            <h2 className="text-lg font-semibold text-slate-900">
              Submit a road complaint
            </h2>

            <p className="mt-1 text-sm text-slate-500">
              Describe the damage and where it was noticed.
            </p>

            <form onSubmit={analyzeComplaint} className="mt-5 space-y-4">
              <textarea
                value={text}
                onChange={(event) => setText(event.target.value)}
                rows={7}
                maxLength={3000}
                placeholder="Example: There is a large pothole near the bus stop. Vehicles are swerving and the road is dangerous."
                className="w-full resize-y rounded-xl border border-slate-200 p-4 text-sm leading-6 text-slate-800 outline-none focus:border-slate-400"
              />

              <div className="flex items-center justify-between text-xs text-slate-400">
                <span>Minimum 10 characters</span>
                <span>{text.length}/3000</span>
              </div>

              {error && (
                <div className="rounded-xl border border-red-200 bg-red-50 p-3 text-sm text-red-700">
                  {error}
                </div>
              )}

              <button
                type="submit"
                disabled={loading || text.trim().length < 10}
                className="inline-flex w-full items-center justify-center gap-2 rounded-xl bg-slate-900 px-4 py-3 font-medium text-white hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {loading ? (
                  <>
                    <Loader2 className="size-4 animate-spin" />
                    Analyzing complaint...
                  </>
                ) : (
                  <>
                    <Send className="size-4" />
                    Analyze complaint
                  </>
                )}
              </button>
            </form>
          </section>

          <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
            <h2 className="text-lg font-semibold text-slate-900">
              Analysis result
            </h2>

            {!result ? (
              <div className="mt-8 flex flex-col items-center justify-center text-center">
                <ClipboardList className="size-10 text-slate-300" />
                <p className="mt-3 text-sm text-slate-500">
                  Submit a complaint to see the analysis.
                </p>
              </div>
            ) : (
              <div className="mt-5 space-y-5">
                <div>
                  <p className="text-xs font-medium uppercase tracking-wide text-slate-400">
                    Complaint ID
                  </p>
                  <p className="mt-1 text-2xl font-semibold text-slate-900">
                    #{result.id}
                  </p>
                </div>

                <div>
                  <p className="text-xs font-medium uppercase tracking-wide text-slate-400">
                    Estimated urgency
                  </p>
                  <span
                    className={`mt-2 inline-flex rounded-full border px-3 py-1.5 text-sm font-semibold ${urgencyColor(result.urgency)}`}
                  >
                    {result.urgency}
                  </span>

                  <p className="mt-2 text-sm text-slate-600">
                    Model confidence:{" "}
                    <span className="font-semibold text-slate-900">
                      {confidencePercent(result.urgency_confidence)}
                    </span>
                  </p>
                </div>

                <div>
                  <p className="text-xs font-medium uppercase tracking-wide text-slate-400">
                    Damage type
                  </p>
                  <div className="mt-2 flex flex-wrap gap-2">
                    {getDamageTypes(result).map((damage) => (
                      <span
                        key={damage}
                        className="rounded-lg bg-slate-100 px-3 py-1.5 text-sm font-medium text-slate-700"
                      >
                        {damage}
                      </span>
                    ))}
                  </div>

                  <p className="mt-2 text-sm text-slate-600">
                    Model confidence:{" "}
                    <span className="font-semibold text-slate-900">
                      {confidencePercent(result.damage_confidence)}
                    </span>
                  </p>
                </div>

                <div>
                  <p className="text-xs font-medium uppercase tracking-wide text-slate-400">
                    Mentioned location
                  </p>
                  <p className="mt-1 font-medium text-slate-800">
                    {result.location_text || "Not identified"}
                  </p>
                  <p className="mt-1 text-xs text-amber-700">
                    Not GPS-verified
                  </p>
                </div>

                <div>
                  <p className="text-xs font-medium uppercase tracking-wide text-slate-400">
                    NLP summary
                  </p>
                  <p className="mt-1 text-sm leading-6 text-slate-700">
                    {result.summary}
                  </p>
                </div>

                <div className="rounded-lg bg-slate-50 p-3">
                  <p className="text-xs text-slate-500">
                    Model: {result.nlp_model || "Trained NLP model"}
                  </p>
                </div>
              </div>
            )}
          </section>
        </div>

        <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
          <div className="flex items-center justify-between gap-3">
            <h2 className="text-lg font-semibold text-slate-900">
              Complaint history
            </h2>
            <span className="text-sm text-slate-500">
              {history.length} records
            </span>
          </div>

          {history.length === 0 ? (
            <p className="mt-5 text-sm text-slate-500">
              No complaints have been analyzed yet.
            </p>
          ) : (
            <div className="mt-4 space-y-3">
              {history.map((complaint) => (
                <div
                  key={complaint.id}
                  className="rounded-xl border border-slate-100 p-4"
                >
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <p className="font-medium text-slate-800">
                      Complaint #{complaint.id}
                    </p>
                    <span
                      className={`rounded-full border px-2.5 py-1 text-xs font-semibold ${urgencyColor(complaint.urgency)}`}
                    >
                      {complaint.urgency}
                    </span>
                  </div>

                  <p className="mt-2 text-sm text-slate-600">
                    {complaint.summary}
                  </p>

                  <div className="mt-3 grid gap-2 text-xs text-slate-500 sm:grid-cols-2">
                    <p>
                      Damage confidence:{" "}
                      <span className="font-medium text-slate-700">
                        {confidencePercent(complaint.damage_confidence)}
                      </span>
                    </p>
                    <p>
                      Urgency confidence:{" "}
                      <span className="font-medium text-slate-700">
                        {confidencePercent(complaint.urgency_confidence)}
                      </span>
                    </p>
                  </div>

                  <p className="mt-2 line-clamp-2 text-xs text-slate-400">
                    {complaint.complaint_text}
                  </p>
                </div>
              ))}
            </div>
          )}
        </section>

        <p className="text-center text-xs text-slate-400">
          RoadSight AI · NLP complaint analysis prototype
        </p>
      </div>
    </main>
  );
}
