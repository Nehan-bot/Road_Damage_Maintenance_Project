
import Link from "next/link";
import {
  ArrowRight,
  Activity,
  AlertTriangle,
  Camera,
  CheckCircle2,
  ChevronRight,
  MapPin,
  ShieldCheck,
  Upload,
} from "lucide-react";

const features = [
  {
    icon: Camera,
    title: "AI-powered damage detection",
    description:
      "Analyse road images and identify visible cracks, potholes, and other road surface damage.",
  },
  {
    icon: Activity,
    title: "Road condition insights",
    description:
      "Turn image detections into useful inspection records and damage summaries.",
  },
  {
    icon: ShieldCheck,
    title: "Maintenance prioritization",
    description:
      "Use detected damage information to generate exploratory maintenance priority estimates for review.",
  },
];

const damageTypes = [
  "Longitudinal cracks",
  "Transverse cracks",
  "Alligator cracks",
  "Potholes",
];

export default function LandingPage() {
  return (
    <main className="min-h-screen overflow-hidden bg-white text-slate-900">
      {/* Navigation */}
      <header className="relative z-10 border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-5 lg:px-10">
          <Link href="/" className="flex items-center gap-3">
            <div className="flex size-11 items-center justify-center rounded-xl bg-slate-900 text-white">
              <ShieldCheck className="size-6" />
            </div>

            <div>
              <p className="text-lg font-bold tracking-tight">
                RoadSight AI
              </p>
              <p className="text-xs text-slate-500">
                Intelligent road maintenance
              </p>
            </div>
          </Link>

          <nav className="hidden items-center gap-8 text-sm font-medium text-slate-600 md:flex">
            <a href="#why-road-safety" className="transition hover:text-slate-900">
              Why road safety
            </a>
            <a href="#how-it-works" className="transition hover:text-slate-900">
              How it works
            </a>
            <a href="#features" className="transition hover:text-slate-900">
              Features
            </a>
          </nav>

          <Link
            href="/dashboard"
            className="inline-flex items-center gap-2 rounded-xl border border-slate-200 px-4 py-2.5 text-sm font-semibold transition hover:bg-slate-50"
          >
            Dashboard
            <ArrowRight className="size-4" />
          </Link>
        </div>
      </header>

      {/* Hero */}
      <section className="relative isolate">
        <div className="absolute inset-0 -z-10 bg-gradient-to-br from-slate-50 via-white to-blue-50" />
        <div className="absolute right-0 top-0 -z-10 h-[500px] w-[500px] rounded-full bg-blue-100/40 blur-3xl" />

        <div className="mx-auto grid max-w-7xl items-center gap-14 px-6 py-20 lg:grid-cols-2 lg:px-10 lg:py-28">
          <div>
            <div className="mb-6 inline-flex items-center gap-2 rounded-full border border-blue-100 bg-white px-4 py-2 text-sm font-medium text-slate-700 shadow-sm">
              <span className="size-2 rounded-full bg-emerald-500" />
              Computer vision for safer roads
            </div>

            <h1 className="max-w-2xl text-4xl font-bold leading-tight tracking-tight sm:text-5xl lg:text-6xl">
              Smarter roads.
              <span className="block text-blue-700">
                Safer journeys.
              </span>
            </h1>

            <p className="mt-6 max-w-xl text-lg leading-8 text-slate-600">
              RoadSight AI helps identify road surface damage from images
              and organise inspection results into useful road maintenance
              insights.
            </p>

            <div className="mt-9 flex flex-col gap-3 sm:flex-row">
              <Link
                href="/inspections"
                className="inline-flex items-center justify-center gap-2 rounded-xl bg-slate-900 px-6 py-3.5 text-sm font-semibold text-white shadow-lg shadow-slate-900/10 transition hover:bg-slate-700"
              >
                <Upload className="size-4" />
                Get Started – Analyse an Image
                <ArrowRight className="size-4" />
              </Link>

              <Link
                href="/dashboard"
                className="inline-flex items-center justify-center gap-2 rounded-xl border border-slate-200 bg-white px-6 py-3.5 text-sm font-semibold text-slate-700 transition hover:bg-slate-50"
              >
                View Dashboard
                <ChevronRight className="size-4" />
              </Link>
            </div>

            <div className="mt-8 flex flex-wrap items-center gap-x-6 gap-y-3 text-sm text-slate-500">
              <span className="flex items-center gap-2">
                <CheckCircle2 className="size-4 text-emerald-600" />
                Image-based inspection
              </span>
              <span className="flex items-center gap-2">
                <CheckCircle2 className="size-4 text-emerald-600" />
                Damage classification
              </span>
            </div>
          </div>

          {/* Hero visual */}
          <div className="relative">
            <div className="absolute -inset-5 rounded-[2rem] bg-blue-100/50 blur-2xl" />

            <div className="relative overflow-hidden rounded-3xl border border-slate-200 bg-slate-950 p-5 shadow-2xl shadow-slate-900/15 sm:p-7">
              <div className="mb-5 flex items-center justify-between">
                <div>
                  <p className="text-sm font-semibold text-white">
                    Road image analysis
                  </p>
                  <p className="mt-1 text-xs text-slate-400">
                    Example detection interface
                  </p>
                </div>

                <span className="inline-flex items-center gap-2 rounded-full bg-emerald-400/10 px-3 py-1.5 text-xs font-medium text-emerald-300">
                  <span className="size-2 rounded-full bg-emerald-400" />
                  AI analysis
                </span>
              </div>

              <div className="relative flex aspect-[4/3] items-center justify-center overflow-hidden rounded-2xl bg-gradient-to-br from-slate-700 via-slate-800 to-slate-950">
                <div className="absolute inset-0 opacity-20">
                  <div className="absolute left-1/2 top-0 h-full w-1 -translate-x-1/2 border-l-4 border-dashed border-white" />
                  <div className="absolute left-[20%] top-0 h-full w-px bg-white/30" />
                  <div className="absolute right-[20%] top-0 h-full w-px bg-white/30" />
                </div>

                <div className="absolute left-[16%] top-[24%] h-[24%] w-[30%] rounded-lg border-2 border-amber-400">
                  <span className="absolute -top-6 left-0 rounded bg-amber-400 px-2 py-1 text-[10px] font-bold text-slate-950">
                    Crack detected
                  </span>
                </div>

                <div className="absolute bottom-[18%] right-[13%] h-[25%] w-[34%] rounded-lg border-2 border-red-400">
                  <span className="absolute -top-6 left-0 rounded bg-red-400 px-2 py-1 text-[10px] font-bold text-slate-950">
                    Pothole detected
                  </span>
                </div>

                <div className="relative z-10 flex size-16 items-center justify-center rounded-2xl border border-white/20 bg-white/10 backdrop-blur">
                  <Camera className="size-8 text-white" />
                </div>

                <div className="absolute bottom-4 left-4 right-4 flex items-center justify-between rounded-xl border border-white/10 bg-slate-950/70 px-4 py-3 backdrop-blur">
                  <div>
                    <p className="text-xs text-slate-400">
                      Detected damage
                    </p>
                    <p className="mt-1 text-sm font-semibold text-white">
                      Crack · Pothole
                    </p>
                  </div>
                  <AlertTriangle className="size-5 text-amber-400" />
                </div>
              </div>

              <div className="mt-5 grid grid-cols-3 gap-3">
                <div className="rounded-xl bg-white/5 p-3">
                  <p className="text-xs text-slate-400">Detection</p>
                  <p className="mt-1 text-sm font-semibold text-white">
                    YOLO
                  </p>
                </div>
                <div className="rounded-xl bg-white/5 p-3">
                  <p className="text-xs text-slate-400">Damage types</p>
                  <p className="mt-1 text-sm font-semibold text-white">
                    4 classes
                  </p>
                </div>
                <div className="rounded-xl bg-white/5 p-3">
                  <p className="text-xs text-slate-400">Priority</p>
                  <p className="mt-1 text-sm font-semibold text-white">
                    Estimate
                  </p>
                </div>
              </div>
            </div>

            <div className="absolute -bottom-6 -left-6 hidden items-center gap-3 rounded-2xl border border-slate-200 bg-white p-4 shadow-xl sm:flex">
              <div className="flex size-10 items-center justify-center rounded-xl bg-emerald-50">
                <MapPin className="size-5 text-emerald-700" />
              </div>
              <div>
                <p className="text-sm font-semibold">Road inspection</p>
                <p className="text-xs text-slate-500">
                  Detect · Record · Review
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Why road safety */}
      <section id="why-road-safety" className="scroll-mt-20 bg-slate-950 py-20 text-white">
        <div className="mx-auto grid max-w-7xl gap-12 px-6 lg:grid-cols-2 lg:px-10">
          <div>
            <p className="text-sm font-semibold uppercase tracking-widest text-blue-300">
              Why road safety matters
            </p>
            <h2 className="mt-4 text-3xl font-bold leading-tight tracking-tight sm:text-4xl">
              Road damage is more than a surface problem.
            </h2>
          </div>

          <div className="space-y-5 text-base leading-8 text-slate-300">
            <p>
              Damaged road surfaces can affect vehicle control, ride
              quality, and the safety of people travelling by two-wheelers,
              cars, buses, and other vehicles.
            </p>
            <p>
              Finding road defects early helps maintenance teams identify
              locations that may need inspection and repair before damage
              becomes more extensive.
            </p>
            <p>
              RoadSight AI supports this process by helping turn road
              images into structured damage observations that can be
              reviewed and prioritised.
            </p>
          </div>
        </div>
      </section>

      {/* What it detects */}
      <section className="py-20">
        <div className="mx-auto max-w-7xl px-6 lg:px-10">
          <div className="max-w-2xl">
            <p className="text-sm font-semibold uppercase tracking-widest text-blue-700">
              What RoadSight AI detects
            </p>
            <h2 className="mt-4 text-3xl font-bold tracking-tight sm:text-4xl">
              Identify common road surface damage
            </h2>
            <p className="mt-4 leading-7 text-slate-600">
              The computer vision model analyses uploaded road images and
              detects four damage categories.
            </p>
          </div>

          <div className="mt-10 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            {damageTypes.map((damage, index) => (
              <div
                key={damage}
                className="rounded-2xl border border-slate-200 bg-slate-50 p-6"
              >
                <div className="mb-5 flex size-10 items-center justify-center rounded-xl bg-white text-sm font-bold text-slate-700 shadow-sm">
                  0{index + 1}
                </div>
                <h3 className="font-semibold">{damage}</h3>
                <p className="mt-2 text-sm leading-6 text-slate-500">
                  Road surface defect category identified from the image.
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* How it works */}
      <section id="how-it-works" className="scroll-mt-20 border-y border-slate-200 bg-slate-50 py-20">
        <div className="mx-auto max-w-7xl px-6 lg:px-10">
          <div className="mx-auto max-w-2xl text-center">
            <p className="text-sm font-semibold uppercase tracking-widest text-blue-700">
              How it works
            </p>
            <h2 className="mt-4 text-3xl font-bold tracking-tight sm:text-4xl">
              From road image to inspection record
            </h2>
          </div>

          <div className="mt-12 grid gap-6 md:grid-cols-3">
            {[
              {
                number: "01",
                title: "Upload an image",
                description:
                  "Choose a road image to start an inspection using the analysis page.",
              },
              {
                number: "02",
                title: "Detect road damage",
                description:
                  "The computer vision model identifies damage categories and locations in the image.",
              },
              {
                number: "03",
                title: "Review the result",
                description:
                  "View detections, priority estimates, and the saved inspection record.",
              },
            ].map((step) => (
              <div
                key={step.number}
                className="rounded-2xl border border-slate-200 bg-white p-7"
              >
                <p className="text-sm font-bold text-blue-700">
                  STEP {step.number}
                </p>
                <h3 className="mt-4 text-xl font-semibold">{step.title}</h3>
                <p className="mt-3 leading-7 text-slate-600">
                  {step.description}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Features */}
      <section id="features" className="scroll-mt-20 py-20">
        <div className="mx-auto max-w-7xl px-6 lg:px-10">
          <div className="max-w-2xl">
            <p className="text-sm font-semibold uppercase tracking-widest text-blue-700">
              RoadSight AI platform
            </p>
            <h2 className="mt-4 text-3xl font-bold tracking-tight sm:text-4xl">
              A practical tool for road inspection workflows
            </h2>
          </div>

          <div className="mt-10 grid gap-6 md:grid-cols-3">
            {features.map((feature) => {
              const Icon = feature.icon;

              return (
                <div
                  key={feature.title}
                  className="rounded-2xl border border-slate-200 p-7 transition hover:-translate-y-1 hover:shadow-lg"
                >
                  <div className="flex size-12 items-center justify-center rounded-xl bg-blue-50 text-blue-700">
                    <Icon className="size-6" />
                  </div>
                  <h3 className="mt-5 text-lg font-semibold">
                    {feature.title}
                  </h3>
                  <p className="mt-3 leading-7 text-slate-600">
                    {feature.description}
                  </p>
                </div>
              );
            })}
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="px-6 pb-20 lg:px-10">
        <div className="mx-auto max-w-7xl overflow-hidden rounded-3xl bg-blue-700 px-8 py-12 text-white sm:px-12 sm:py-16">
          <div className="flex flex-col justify-between gap-8 md:flex-row md:items-center">
            <div className="max-w-2xl">
              <h2 className="text-3xl font-bold tracking-tight sm:text-4xl">
                Start your road inspection
              </h2>
              <p className="mt-4 leading-7 text-blue-100">
                Upload a road image and explore how RoadSight AI identifies
                damage and organises inspection results.
              </p>
            </div>

            <Link
              href="/inspections"
              className="inline-flex shrink-0 items-center justify-center gap-2 rounded-xl bg-white px-6 py-3.5 text-sm font-semibold text-blue-800 transition hover:bg-blue-50"
            >
              <Upload className="size-4" />
              Analyse an Image
              <ArrowRight className="size-4" />
            </Link>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-slate-200">
        <div className="mx-auto flex max-w-7xl flex-col gap-4 px-6 py-8 text-sm text-slate-500 sm:flex-row sm:items-center sm:justify-between lg:px-10">
          <div className="flex items-center gap-2">
            <ShieldCheck className="size-4" />
            <span className="font-semibold text-slate-700">
              RoadSight AI
            </span>
            <span>· Road intelligence</span>
          </div>

          <p>
            AI-assisted road damage detection for inspection and maintenance
            review.
          </p>
        </div>
      </footer>
    </main>
  );
}
