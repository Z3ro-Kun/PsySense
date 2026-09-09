import { useState, type ChangeEvent, type DragEvent } from "react";
import "./PhotoDropzone.css";

interface PhotoDropzoneProps {
  images: File[];
  onImagesChange: (images: File[]) => void;
}

export function PhotoDropzone({ images, onImagesChange }: PhotoDropzoneProps) {
  const [dragOver, setDragOver] = useState(false);

  const addFiles = (files: FileList | null) => {
    if (!files) return;
    const imageFiles = Array.from(files).filter((f) => f.type.startsWith("image/"));
    onImagesChange([...images, ...imageFiles]);
  };

  const removeImage = (index: number) => onImagesChange(images.filter((_, i) => i !== index));

  const handleDrop = (evt: DragEvent<HTMLDivElement>) => {
    evt.preventDefault();
    setDragOver(false);
    addFiles(evt.dataTransfer.files);
  };

  return (
    <>
      <div
        className={`photo-dropzone${dragOver ? " photo-dropzone-active" : ""}`}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
      >
        <p>Drag photos here, or</p>
        <label className="btn btn-secondary" style={{ display: "inline-block", cursor: "pointer" }}>
          Choose files
          <input
            type="file" accept="image/*" multiple hidden
            onChange={(e: ChangeEvent<HTMLInputElement>) => addFiles(e.target.files)}
          />
        </label>
      </div>

      {images.length > 0 && (
        <ul className="photo-dropzone-file-list">
          {images.map((file, i) => (
            <li key={i}>
              {file.name}
              <button type="button" onClick={() => removeImage(i)} aria-label={`Remove ${file.name}`}>
                ×
              </button>
            </li>
          ))}
        </ul>
      )}
    </>
  );
}
