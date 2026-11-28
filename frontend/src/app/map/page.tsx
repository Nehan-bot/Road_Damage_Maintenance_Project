
"use client";

import "leaflet/dist/leaflet.css";

import { useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import {
  AlertTriangle,
  ArrowLeft,
  RefreshCw,
} from "lucide-react";

type Inspection = {
  id: number;
  filename: string;
  priority?: string | null;
  priority_confidence?: number | null;
  detection_count?: number;
  created_at?: string;
};

type DemoLocation = {
  name: string;
  district: string;
  lat: number;
  lng: number;
};

const API_URL = "http://localhost:8000";

const DEMO_LOCATIONS: DemoLocation[] = [
  { name: "Kasaragod", district: "Kasaragod", lat: 12.4996, lng: 74.9869 },
  { name: "Kannur", district: "Kannur", lat: 11.8745, lng: 75.3704 },
  { name: "Kozhikode", district: "Kozhikode", lat: 11.2588, lng: 75.7804 },
  { name: "Wayanad", district: "Wayanad", lat: 11.6854, lng: 76.132 },
  { name: "Malappuram", district: "Malappuram", lat: 11.051, lng: 76.0711 },
  { name: "Palakkad", district: "Palakkad", lat: 10.7867, lng: 76.6548 },
  { name: "Thrissur", district: "Thrissur", lat: 10.5276, lng: 76.2144 },
  { name: "Kochi", district: "Ernakulam", lat: 9.9312, lng: 76.2673 },
  { name: "Alappuzha", district: "Alappuzha", lat: 9.4981, lng: 76.3388 },
  { name: "Kottayam", district: "Kottayam", lat: 9.5916, lng: 76.5222 },
  { name: "Kollam", district: "Kollam", lat: 8.8932, lng: 76.6141 },
  {
    name: "Thiruvananthapuram",
    district: "Thiruvananthapuram",
    lat: 8.5241,
    lng: 76.9366,
  },
];

function getDemoLocation(id: number): DemoLocation {
  return DEMO_LOCATIONS[
    Math.abs(Number(id) || 0) % DEMO_LOCATIONS.length
  ];
}

function getPriorityColor(priority?: string | null) {
  switch ((priority || "").toLowerCase()) {
    case "high":
      return "#dc2626";
    case "medium":
      return "#d97706";
    case "low":
      return "#16a34a";
    default:
      return "#64748b";
  }
}

function getPriorityLabel(priority?: string | null) {
  if (!priority) return "Unknown";

  return (
    priority.charAt(0).toUpperCase() +
    priority.slice(1).toLowerCase()
  );
}

function formatDate(value?: string) {
  if (!value) return "Date not available";

  const date = new Date(value);

  if (Number.isNaN(date.getTime())) {
    return value;
  }

  return date.toLocaleString();
}

export default function PriorityMapPage() {
  const mapContainerRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<any>(null);
  const markerLayerRef = useRef<any>(null);

  const [inspections, setInspections] = useState<Inspection[]>([]);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);
  const [mapReady, setMapReady] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const [priorityFilter, setPriorityFilter] = useState("All");

  async function loadInspections() {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_URL}/inspections`);

      if (!response.ok) {
        throw new Error("Could not load inspection history.");
      }

      const data = await response.json();

      const records: Inspection[] = Array.isArray(data)
        ? data
        : Array.isArray(data.inspections)
          ? data.inspections
          : [];

      setInspections(records);

      if (records.length > 0) {
        setSelectedId((current) =>
          current !== null &&
          records.some((item) => item.id === current)
            ? current
            : records[0].id
        );
      } else {
        setSelectedId(null);
      }
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Unexpected error while loading inspections."
      );
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadInspections();
  }, []);

  // Initialize the Leaflet map and CARTO Voyager raster basemap.
  useEffect(() => {
    let cancelled = false;

    async function initializeMap() {
      if (!mapContainerRef.current || mapRef.current) {
        return;
      }

      const L = await import("leaflet");

      if (cancelled || !mapContainerRef.current) {
        return;
      }

      const apiKey = process.env.NEXT_PUBLIC_CARTO_API_KEY;

      if (!apiKey) {
        setError(
          "CARTO API key is missing. Add NEXT_PUBLIC_CARTO_API_KEY to frontend/.env.local and restart Next.js."
        );
      }

      const map = L.map(mapContainerRef.current, {
        zoomControl: true,
        scrollWheelZoom: true,
      }).setView([10.25, 76.25], 7);

      const tileUrl = apiKey
        ? `https://basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png?key=${encodeURIComponent(apiKey)}`
        : "https://basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png";

      L.tileLayer(tileUrl, {
        attribution:
          '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
        maxZoom: 20,
        tileSize: 256,
      }).addTo(map);

      markerLayerRef.current = L.layerGroup().addTo(map);
      mapRef.current = map;

      setMapReady(true);
    }

    initializeMap();

    return () => {
      cancelled = true;

      if (mapRef.current) {
        mapRef.current.remove();
        mapRef.current = null;
        markerLayerRef.current = null;
      }
    };
  }, []);

  const mappedInspections = useMemo(
    () =>
      inspections.map((inspection) => ({
        ...inspection,
        demoLocation: getDemoLocation(inspection.id),
      })),
    [inspections]
  );

  const filteredInspections = useMemo(() => {
    const searchText = search.trim().toLowerCase();

    return mappedInspections.filter((inspection) => {
      const matchesPriority =
        priorityFilter === "All" ||
        getPriorityLabel(inspection.priority) === priorityFilter;

      const matchesSearch =
        !searchText ||
        String(inspection.id).includes(searchText) ||
        inspection.filename.toLowerCase().includes(searchText) ||
        inspection.demoLocation.name.toLowerCase().includes(searchText);

      return matchesPriority && matchesSearch;
    });
  }, [mappedInspections, search, priorityFilter]);

  const selectedInspection = mappedInspections.find(
    (inspection) => inspection.id === selectedId
  );

  const highCount = inspections.filter(
    (item) => item.priority?.toLowerCase() === "high"
  ).length;

  const mediumCount = inspections.filter(
    (item) => item.priority?.toLowerCase() === "medium"
  ).length;

  const lowCount = inspections.filter(
    (item) => item.priority?.toLowerCase() === "low"
  ).length;

  // Refresh map markers when inspections or priorities change.
  useEffect(() => {
    if (!mapReady || !mapRef.current || !markerLayerRef.current) {
      return;
    }

    let active = true;

    async function updateMarkers() {
      const L = await import("leaflet");

      if (!active || !markerLayerRef.current) {
        return;
      }

      markerLayerRef.current.clearLayers();

      mappedInspections.forEach((inspection) => {
        const location = inspection.demoLocation;
        const color = getPriorityColor(inspection.priority);

        const markerIcon = L.divIcon({
          className: "roadsight-priority-marker",
          html: `
            <div style="
              width:34px;
              height:34px;
              border-radius:50%;
              background:${color};
              border:3px solid white;
              box-shadow:0 2px 8px rgba(15,23,42,0.28);
              display:flex;
              align-items:center;
              justify-content:center;
              color:white;
              font-size:12px;
              font-weight:700;
              font-family:Arial,sans-serif;
            ">
              ${inspection.id}
            </div>
          `,
          iconSize: [34, 34],
          iconAnchor: [17, 17],
        });

        const marker = L.marker(
          [location.lat, location.lng],
          { icon: markerIcon }
        );

        const popup = document.createElement("div");
        popup.style.minWidth = "180px";
        popup.style.fontFamily = "Arial, sans-serif";

        const heading = document.createElement("strong");
        heading.textContent = `Inspection #${inspection.id}`;
        heading.style.display = "block";
        heading.style.marginBottom = "6px";

        const place = document.createElement("div");
        place.textContent = `${location.name}, ${location.district}`;
        place.style.marginBottom = "6px";

        const priority = document.createElement("div");
        priority.textContent =
          `Priority: ${getPriorityLabel(inspection.priority)}`;
        priority.style.color = color;
        priority.style.fontWeight = "700";
        priority.style.marginBottom = "6px";

        const warning = document.createElement("div");
        warning.textContent = "Illustrative demo location";
        warning.style.fontSize = "11px";
        warning.style.color = "#b45309";
        warning.style.marginBottom = "8px";

        const link = document.createElement("a");
        link.href = `/history/${inspection.id}`;
        link.textContent = "View inspection details";
        link.style.fontWeight = "600";
        link.style.color = "#1d4ed8";

        popup.append(heading, place, priority, warning, link);

        marker.bindPopup(popup);

        marker.on("click", () => {
          setSelectedId(inspection.id);
        });

        marker.addTo(markerLayerRef.current);
      });
    }

    updateMarkers();

    return () => {
      active = false;
    };
  }, [mapReady, mappedInspections]);

  function focusInspection(
    inspection: (typeof mappedInspections)[number]
  ) {
    setSelectedId(inspection.id);

    if (mapRef.current) {
      mapRef.current.setView(
        [
          inspection.demoLocation.lat,
          inspection.demoLocation.lng,
        ],
        10,
        { animate: true }
      );
    }
  }

  return (
    <main className="min-h-screen bg-slate-50 p-4 md:p-8">
      <div className="mx-auto max-w-7xl space-y-6">
        <header className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <Link
              href="/dashboard"
              className="mb-3 inline-flex items-center gap-2 text-sm text-slate-500 hover:text-slate-900"
            >
              <ArrowLeft className="size-4" />
              Back to dashboard
            </Link>

            <p className="text-sm font-medium text-slate-500">
              RoadSight AI / Map
            </p>

            <h1 className="mt-1 text-3xl font-semibold tracking-tight text-slate-900">
              Road inspection map
            </h1>

            <p className="mt-2 text-sm text-slate-500">
              Explore inspection records and model-estimated priority.
            </p>
          </div>

          <button
            onClick={loadInspections}
            disabled={loading}
            className="inline-flex items-center gap-2 rounded-xl border border-slate-200 bg-white px-4 py-2.5 text-sm font-medium text-slate-700 shadow-sm hover:bg-slate-50 disabled:opacity-50"
          >
            <RefreshCw
              className={`size-4 ${loading ? "animate-spin" : ""}`}
            />
            Refresh
          </button>
        </header>

        <div className="flex items-start gap-3 rounded-2xl border border-amber-200 bg-amber-50 p-4 text-amber-900">
          <AlertTriangle className="mt-0.5 size-5 shrink-0" />

          <div>
            <p className="font-semibold">
              Demo locations — not GPS-verified
            </p>
            <p className="mt-1 text-sm leading-6">
              Map markers use illustrative Kerala locations assigned
              from inspection IDs. They do not represent the actual
              capture location of uploaded road images.
            </p>
          </div>
        </div>

        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <SummaryCard label="Total inspections" value={inspections.length} />
          <SummaryCard label="High priority" value={highCount} color="text-red-600" />
          <SummaryCard label="Medium priority" value={mediumCount} color="text-amber-600" />
          <SummaryCard label="Low priority" value={lowCount} color="text-green-600" />
        </div>

        {error && (
          <div className="rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-700">
            {error}
            <p className="mt-1">
              Check the frontend environment file and backend status.
            </p>
          </div>
        )}

        <div className="grid gap-6 lg:grid-cols-[minmax(0,1.7fr)_minmax(300px,0.8fr)]">
          <section className="overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm">
            <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-100 p-5">
              <div>
                <h2 className="font-semibold text-slate-900">
                  Kerala
                </h2>
                <p className="mt-1 text-xs text-slate-500">
                  CARTO Voyager · Illustrative inspection markers
                </p>
              </div>

              <div className="flex flex-wrap gap-3 text-xs text-slate-600">
                <Legend color="#dc2626" label="High" />
                <Legend color="#d97706" label="Medium" />
                <Legend color="#16a34a" label="Low" />
              </div>
            </div>

            <div className="relative">
              <div
                ref={mapContainerRef}
                className="h-[520px] w-full bg-slate-100"
              />

              {!mapReady && (
                <div className="absolute inset-0 flex items-center justify-center bg-slate-100 text-sm text-slate-500">
                  Loading map...
                </div>
              )}
            </div>

            <div className="border-t border-slate-100 p-4 text-xs text-slate-500">
              Map tiles © OpenStreetMap contributors and CARTO.
              Marker coordinates are demonstration data.
            </div>
          </section>

          <aside className="space-y-4">
            <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
              <h2 className="font-semibold text-slate-900">
                Selected inspection
              </h2>

              {!selectedInspection ? (
                <p className="mt-4 text-sm text-slate-500">
                  Select an inspection marker or list item.
                </p>
              ) : (
                <div className="mt-4 space-y-4">
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <p className="text-xs text-slate-500">
                        Inspection ID
                      </p>
                      <p className="text-2xl font-semibold text-slate-900">
                        #{selectedInspection.id}
                      </p>
                    </div>

                    <span
                      className="rounded-full px-3 py-1 text-xs font-semibold text-white"
                      style={{
                        backgroundColor: getPriorityColor(
                          selectedInspection.priority
                        ),
                      }}
                    >
                      {getPriorityLabel(selectedInspection.priority)}
                    </span>
                  </div>

                  <div>
                    <p className="text-xs text-slate-500">
                      Demo location
                    </p>
                    <p className="mt-1 font-medium text-slate-800">
                      {selectedInspection.demoLocation.name},{" "}
                      {selectedInspection.demoLocation.district}
                    </p>
                  </div>

                  <div>
                    <p className="text-xs text-slate-500">
                      Image filename
                    </p>
                    <p className="mt-1 break-all text-sm text-slate-800">
                      {selectedInspection.filename}
                    </p>
                  </div>

                  <div>
                    <p className="text-xs text-slate-500">
                      Detected damage objects
                    </p>
                    <p className="mt-1 text-lg font-semibold text-slate-900">
                      {selectedInspection.detection_count ?? "—"}
                    </p>
                  </div>

                  <div>
                    <p className="text-xs text-slate-500">
                      Inspection date
                    </p>
                    <p className="mt-1 text-sm text-slate-800">
                      {formatDate(selectedInspection.created_at)}
                    </p>
                  </div>

                  <Link
                    href={`/history/${selectedInspection.id}`}
                    className="block rounded-xl bg-slate-900 px-4 py-3 text-center text-sm font-medium text-white hover:bg-slate-800"
                  >
                    View inspection details
                  </Link>
                </div>
              )}
            </section>

            <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
              <h2 className="font-semibold text-slate-900">
                Inspection records
              </h2>

              <div className="mt-4 space-y-3">
                <input
                  value={search}
                  onChange={(event) => setSearch(event.target.value)}
                  placeholder="Search ID, image or demo place..."
                  className="w-full rounded-xl border border-slate-200 px-3 py-2.5 text-sm outline-none focus:border-slate-400"
                />

                <select
                  value={priorityFilter}
                  onChange={(event) =>
                    setPriorityFilter(event.target.value)
                  }
                  className="w-full rounded-xl border border-slate-200 bg-white px-3 py-2.5 text-sm outline-none focus:border-slate-400"
                >
                  <option>All</option>
                  <option>High</option>
                  <option>Medium</option>
                  <option>Low</option>
                  <option>Unknown</option>
                </select>
              </div>

              <div className="mt-4 max-h-[340px] space-y-2 overflow-y-auto">
                {filteredInspections.map((inspection) => (
                  <button
                    key={inspection.id}
                    onClick={() => focusInspection(inspection)}
                    className={`flex w-full items-center justify-between gap-3 rounded-xl border p-3 text-left transition ${
                      selectedId === inspection.id
                        ? "border-slate-400 bg-slate-50"
                        : "border-slate-100 hover:bg-slate-50"
                    }`}
                  >
                    <div className="flex min-w-0 items-center gap-3">
                      <span
                        className="flex size-8 shrink-0 items-center justify-center rounded-full text-xs font-bold text-white"
                        style={{
                          backgroundColor: getPriorityColor(
                            inspection.priority
                          ),
                        }}
                      >
                        {inspection.id}
                      </span>

                      <div className="min-w-0">
                        <p className="truncate text-sm font-medium text-slate-800">
                          Inspection #{inspection.id}
                        </p>
                        <p className="truncate text-xs text-slate-500">
                          {inspection.demoLocation.name} · Demo
                        </p>
                      </div>
                    </div>

                    <span className="shrink-0 text-xs font-medium text-slate-600">
                      {getPriorityLabel(inspection.priority)}
                    </span>
                  </button>
                ))}

                {filteredInspections.length === 0 && (
                  <p className="py-5 text-center text-sm text-slate-500">
                    No matching inspections.
                  </p>
                )}
              </div>
            </section>
          </aside>
        </div>

        <p className="text-center text-xs text-slate-400">
          RoadSight AI · Priority values are exploratory model estimates.
          Locations shown are illustrative demo data.
        </p>
      </div>
    </main>
  );
}

function SummaryCard({
  label,
  value,
  color = "text-slate-900",
}: {
  label: string;
  value: number;
  color?: string;
}) {
  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
      <p className="text-sm text-slate-500">{label}</p>
      <p className={`mt-2 text-3xl font-semibold ${color}`}>
        {value}
      </p>
    </div>
  );
}

function Legend({
  color,
  label,
}: {
  color: string;
  label: string;
}) {
  return (
    <span className="inline-flex items-center gap-1.5">
      <span
        className="size-2.5 rounded-full"
        style={{ backgroundColor: color }}
      />
      {label}
    </span>
  );
}
