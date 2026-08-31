import { useState, useEffect } from "react";

function Analysis() {
  const [mode, setMode] = useState("single");

  const [t1Image, setT1Image] = useState(null);
  const [t2Image, setT2Image] = useState(null);

  const [t1Preview, setT1Preview] = useState(null);
  const [t2Preview, setT2Preview] = useState(null);

  const [query, setQuery] = useState("");
  const [error, setError] = useState("");
  const [uploading, setUploading] = useState(false);

  // =========================
  // CREATE IMAGE PREVIEWS
  // =========================

  useEffect(() => {
    if (!t1Image) {
      setT1Preview(null);
      return;
    }

    const url = URL.createObjectURL(t1Image);
    setT1Preview(url);

    return () => URL.revokeObjectURL(url);
  }, [t1Image]);

  useEffect(() => {
    if (!t2Image) {
      setT2Preview(null);
      return;
    }

    const url = URL.createObjectURL(t2Image);
    setT2Preview(url);

    return () => URL.revokeObjectURL(url);
  }, [t2Image]);


  // =========================
  // IMAGE SELECTION
  // =========================

  const handleImageUpload = (event, setImage) => {
    const file = event.target.files?.[0];

    if (!file) return;

    setError("");

    if (!file.type.startsWith("image/")) {
      setError("Please select a valid image file.");
      return;
    }

    if (file.size > 20 * 1024 * 1024) {
      setError("Image must be smaller than 20 MB.");
      return;
    }

    setImage(file);
  };


  // =========================
  // ANALYZE
  // =========================

  const handleAnalyze = async () => {
    setError("");

    if (!t1Image) {
      setError("Please upload an image.");
      return;
    }

    if (mode === "bi-temporal" && !t2Image) {
      setError("Please upload both T1 and T2 images.");
      return;
    }

    if (!query.trim()) {
      setError("Please enter a question for AKASHA.");
      return;
    }

    try {
      setUploading(true);

      // Create form data
      const formData = new FormData();

      formData.append("mode", mode);
      formData.append("query", query);
      formData.append("t1_image", t1Image);

      if (mode === "bi-temporal" && t2Image) {
        formData.append("t2_image", t2Image);
      }

      // Send request to FastAPI
      const response = await fetch(
        "http://127.0.0.1:8000/api/analyze",
        {
          method: "POST",
          body: formData,
        }
      );

      // Check response
      if (!response.ok) {
        throw new Error(
          `Backend error: ${response.status}`
        );
      }

      const result = await response.json();

      console.log("AKASHA Backend Response:", result);

      alert("Successfully connected to AKASHA backend!");

    } catch (err) {
      console.error("API Error:", err);

      setError(
        err.message || "Could not connect to the backend."
      );

    } finally {
      setUploading(false);
    }
  };


  // =========================
  // REMOVE IMAGES
  // =========================

  const removeT1Image = () => {
    setT1Image(null);
  };

  const removeT2Image = () => {
    setT2Image(null);
  };


  // =========================
  // UPLOAD BOX
  // =========================

  const UploadBox = ({
    title,
    image,
    preview,
    setImage,
    onRemove,
  }) => {
    return (
      <div className="upload-card">

        <div className="upload-title">
          {title}
        </div>

        <label className="upload-area">

          {preview ? (

            <div className="image-preview-container">

              <img
                src={preview}
                alt="Satellite preview"
                className="satellite-preview"
              />

              <div className="preview-overlay">
                Click to replace image
              </div>

            </div>

          ) : (

            <>
              <div className="upload-icon">
                ↑
              </div>

              <strong>
                Upload satellite image
              </strong>

              <small>
                PNG, JPG or JPEG • Max 20 MB
              </small>
            </>

          )}

          <input
            type="file"
            accept="image/png,image/jpeg,image/jpg"
            onChange={(event) =>
              handleImageUpload(event, setImage)
            }
          />

        </label>

        {image && (
          <div className="selected-file-info">

            <span>
              {image.name}
            </span>

            <button
              type="button"
              className="remove-image"
              onClick={onRemove}
            >
              Remove
            </button>

          </div>
        )}

      </div>
    );
  };


  // =========================
  // UI
  // =========================

  return (
    <section className="analysis-section">

      {/* HEADER */}

      <div className="analysis-header">

        <div className="hero-badge">
          AI SATELLITE ANALYSIS
        </div>

        <h2>
          What would you like to
          <span> explore?</span>
        </h2>

        <p>
          Upload satellite imagery and ask AKASHA about
          what you see, or compare two time periods to
          detect changes.
        </p>

      </div>


      {/* MODE */}

      <div className="mode-selector">

        <button
          type="button"
          className={`mode-card ${
            mode === "single" ? "selected" : ""
          }`}
          onClick={() => {
            setMode("single");
            setT2Image(null);
            setError("");
          }}
        >

          <div className="mode-icon">
            ◉
          </div>

          <div>
            <h3>
              Single Image
            </h3>

            <p>
              Understand one satellite image
            </p>
          </div>

        </button>


        <button
          type="button"
          className={`mode-card ${
            mode === "bi-temporal" ? "selected" : ""
          }`}
          onClick={() => {
            setMode("bi-temporal");
            setError("");
          }}
        >

          <div className="mode-icon">
            ◉
          </div>

          <div>
            <h3>
              Bi-Temporal
            </h3>

            <p>
              Compare images across time
            </p>
          </div>

        </button>

      </div>


      {/* UPLOADS */}

      <div
        className={`upload-grid ${
          mode === "single" ? "single-upload" : ""
        }`}
      >

        <UploadBox
          title={
            mode === "single"
              ? "Satellite Image"
              : "T1 — Earlier Image"
          }
          image={t1Image}
          preview={t1Preview}
          setImage={setT1Image}
          onRemove={removeT1Image}
        />


        {mode === "bi-temporal" && (
          <UploadBox
            title="T2 — Later Image"
            image={t2Image}
            preview={t2Preview}
            setImage={setT2Image}
            onRemove={removeT2Image}
          />
        )}

      </div>


      {/* QUESTION */}

      <div className="query-section">

        <label htmlFor="query">
          Ask AKASHA
        </label>

        <textarea
          id="query"
          value={query}
          onChange={(event) =>
            setQuery(event.target.value)
          }
          placeholder={
            mode === "single"
              ? "Example: What objects are visible in this image?"
              : "Example: What changed between these two images?"
          }
          rows={4}
        />

      </div>


      {/* ERROR */}

      {error && (
        <div className="error-message">
          ⚠ {error}
        </div>
      )}


      {/* BUTTON */}

      <button
        type="button"
        className="analyze-button"
        onClick={handleAnalyze}
        disabled={uploading}
      >
        {uploading ? "Analyzing..." : "Analyze with AKASHA"}
        <span>{uploading ? "..." : "→"}</span>
      </button>

    </section>
  );
}

export default Analysis;