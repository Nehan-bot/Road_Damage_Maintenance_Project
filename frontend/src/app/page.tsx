
import {
  Activity,
  AlertTriangle,
  Camera,
  CheckCircle2,
  ChevronRight,
  LayoutDashboard,
  MapPin,
  Settings,
  ShieldCheck,
  Upload,
} from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";

const navigation = [
  { label: "Overview", icon: LayoutDashboard, active: true },
  { label: "Road inspections", icon: Camera, active: false },
  { label: "Priority map", icon: MapPin, active: false },
  { label: "Model monitoring", icon: Activity, active: false },
  { label: "Settings", icon: Settings, active: false },
];

const stats = [
  {
    title: "Road inspections",
    value: "1,284",
    change: "+12.8%",
    description: "Compared with last month",
    icon: Camera,
  },
  {
    title: "Damage detected",
    value: "3,642",
    change: "+8.2%",
    description: "Across inspected road images",
    icon: AlertTriangle,
  },
  {
    title: "High priority",
    value: "186",
    change: "Needs attention",
    description: "Requires maintenance review",
    icon: ShieldCheck,
  },
];

export default function Home() {
  return (
    <main className="min-h-screen bg-[#f6f7f9] text-slate-900">
      <div className="flex min-h-screen">
        <aside className="hidden w-64 shrink-0 border-r border-slate-200 bg-white p-5 md:flex md:flex-col">
          <div className="flex items-center gap-3 px-2 py-3">
            <div className="flex size-10 items-center justify-center rounded-xl bg-slate-900 text-white">
              <ShieldCheck className="size-5" />
            </div>
            <div>
              <p className="text-lg font-bold tracking-tight">RoadSight AI</p>
              <p className="text-xs text-slate-500">Road intelligence</p>
            </div>
          </div>

          <p className="mb-3 mt-10 px-3 text-xs font-semibold uppercase tracking-wider text-slate-400">
            Workspace
          </p>

          <nav className="space-y-1">
            {navigation.map((item) => {
              const Icon = item.icon;

              return (
                <button
                  key={item.label}
                  className={`flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-sm font-medium transition ${
                    item.active
                      ? "bg-slate-900 text-white"
                      : "text-slate-600 hover:bg-slate-100"
                  }`}
                >
                  <Icon className="size-4" />
                  {item.label}
                </button>
              );
            })}
          </nav>

          <div className="mt-auto rounded-2xl bg-slate-50 p-4">
            <div className="mb-2 flex items-center gap-2 text-sm font-semibold">
              <CheckCircle2 className="size-4 text-emerald-600" />
              System status
            </div>
            <p className="text-xs leading-5 text-slate-500">
              Dashboard interface is running. Model API connection will be added
              next.
            </p>
          </div>
        </aside>

        <section className="min-w-0 flex-1">
          <header className="flex h-20 items-center justify-between border-b border-slate-200 bg-white px-6 md:px-10">
            <div>
              <p className="text-sm text-slate-500">Workspace / Overview</p>
              <h1 className="text-xl font-semibold tracking-tight">
                Road condition dashboard
              </h1>
            </div>

            <Badge variant="secondary" className="gap-2 rounded-full px-3 py-1.5">
              <span className="size-2 rounded-full bg-emerald-500" />
              Prototype
            </Badge>
          </header>

          <div className="mx-auto max-w-7xl space-y-8 p-6 md:p-10">
            <div className="flex flex-col justify-between gap-4 sm:flex-row sm:items-end">
              <div>
                <p className="text-sm font-medium text-slate-500">
                  Monday, Road Operations
                </p>
                <h2 className="mt-2 text-3xl font-semibold tracking-tight">
                  Good morning
                </h2>
                <p className="mt-2 text-slate-500">
                  Monitor road damage and identify maintenance priorities.
                </p>
              </div>

              <Button className="gap-2 rounded-xl">
                <Upload className="size-4" />
                New inspection
              </Button>
            </div>

            <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
              {stats.map((stat) => {
                const Icon = stat.icon;

                return (
                  <Card key={stat.title} className="rounded-2xl border-slate-200 shadow-sm">
                    <CardHeader className="flex flex-row items-start justify-between space-y-0 pb-3">
                      <div>
                        <CardDescription>{stat.title}</CardDescription>
                        <CardTitle className="mt-2 text-3xl font-semibold">
                          {stat.value}
                        </CardTitle>
                      </div>
                      <div className="rounded-xl bg-slate-100 p-2.5">
                        <Icon className="size-5 text-slate-600" />
                      </div>
                    </CardHeader>
                    <CardContent>
                      <p className="text-sm font-medium text-emerald-600">
                        {stat.change}
                      </p>
                      <p className="mt-1 text-xs text-slate-500">
                        {stat.description}
                      </p>
                    </CardContent>
                  </Card>
                );
              })}
            </div>

            <div className="grid gap-6 xl:grid-cols-[1.5fr_1fr]">
              <Card className="rounded-2xl border-slate-200 shadow-sm">
                <CardHeader>
                  <CardTitle>Recent inspections</CardTitle>
                  <CardDescription>
                    Latest road assessment activity
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="rounded-xl border border-dashed border-slate-300 p-8 text-center">
                    <div className="mx-auto mb-3 flex size-12 items-center justify-center rounded-xl bg-slate-100">
                      <Camera className="size-5 text-slate-500" />
                    </div>
                    <p className="font-medium">Inspection history is coming</p>
                    <p className="mt-1 text-sm text-slate-500">
                      Your uploaded images and model predictions will appear here.
                    </p>
                  </div>
                </CardContent>
              </Card>

              <Card className="rounded-2xl border-slate-200 shadow-sm">
                <CardHeader>
                  <CardTitle>Maintenance priorities</CardTitle>
                  <CardDescription>
                    Road segments requiring review
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex items-center gap-3 rounded-xl bg-amber-50 p-4">
                    <AlertTriangle className="size-5 text-amber-600" />
                    <div className="flex-1">
                      <p className="text-sm font-semibold">High priority</p>
                      <p className="text-xs text-slate-500">
                        Review severe damage first
                      </p>
                    </div>
                    <ChevronRight className="size-4 text-slate-400" />
                  </div>

                  <Separator />

                  <div className="flex items-center gap-3 rounded-xl bg-slate-50 p-4">
                    <Activity className="size-5 text-slate-600" />
                    <div className="flex-1">
                      <p className="text-sm font-semibold">Model activity</p>
                      <p className="text-xs text-slate-500">
                        YOLO26s integration is next
                      </p>
                    </div>
                    <Badge variant="outline">Pending</Badge>
                  </div>
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
