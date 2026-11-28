
"use client";

import { useState } from "react";
import { Camera, ImagePlus, Upload, X } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

export default function InspectionsPage() {
  const [image, setImage] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);

  function handleImageChange(event: React.ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];

    if (!file) return;

    if (!file.type.startsWith("image/")) {
      alert("Please select an image file.");
      return;
    }

    setImage(file);
    setPreview(URL.createObjectURL(file));
  }

  function clearImage() {
    setImage(null);
    setPreview(null);
  }

  return (
    <main className="min-h-screen bg-[#f6f7f9] p-6 md:p-10">
      <div className="mx-auto max-w-4xl space-y-8">
        <div>
          <p className="text-sm font-medium text-slate-500">
            RoadSight AI / Inspections
          </p>
          <h1 className="mt-2 text-3xl font-semibold tracking-tight">
            New road inspection
          </h1>
          <p className="mt-2 text-slate-500">
            Upload a road image to prepare it for damage detection.
          </p>
        </div>

        <Card className="rounded-2xl border-slate-200 shadow-sm">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Camera className="size-5" />
              Upload road image
            </CardTitle>
            <CardDescription>
              Choose a clear image of the road surface.
            </CardDescription>
          </CardHeader>

          <CardContent>
            {!preview ? (
              <label className="flex min-h-72 cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed border-slate-300 bg-slate-50 p-8 text-center transition hover:border-slate-500">
                <div className="mb-4 flex size-14 items-center justify-center rounded-2xl bg-white shadow-sm">
                  <ImagePlus className="size-6 text-slate-600" />
                </div>
                <p className="font-semibold">Choose an image to upload</p>
                <p className="mt-2 text-sm text-slate-500">
                  Select a JPG, PNG, or other image from your computer
                </p>
                <input
                  type="file"
                  accept="image/*"
                  className="hidden"
                  onChange={handleImageChange}
                />
              </label>
            ) : (
              <div className="space-y-4">
                <div className="relative overflow-hidden rounded-xl border bg-white">
                  <img
                    src={preview}
                    alt="Selected road for inspection"
                    className="max-h-[480px] w-full object-contain"
                  />
                  <Button
                    variant="secondary"
                    size="icon"
                    className="absolute right-3 top-3"
                    onClick={clearImage}
                    aria-label="Remove selected image"
                  >
                    <X className="size-4" />
                  </Button>
                </div>

                <p className="truncate text-sm text-slate-600">
                  Selected file: <span className="font-medium">{image?.name}</span>
                </p>

                <Button disabled className="w-full gap-2 rounded-xl">
                  <Upload className="size-4" />
                  Run damage detection
                </Button>

                <p className="text-center text-xs text-slate-500">
                  Model API is not connected yet. Detection will be enabled after
                  we integrate the FastAPI backend.
                </p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </main>
  );
}
