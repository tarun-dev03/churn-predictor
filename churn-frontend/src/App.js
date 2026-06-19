import React, { useState } from "react";
import "./App.css";

function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");

  const handleUpload = async () => {
    if (!file) {
      setError("Please select a CSV file before running prediction.");
      return;
    }

    setError("");
    setIsLoading(true);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        throw new Error("Prediction request failed. Try again.");
      }

      const data = await res.json();
      setResult(data.predictions || []);
    } catch (uploadError) {
      setError(uploadError.message || "Something went wrong while predicting.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <main className="app-shell">
      <section className="hero">
        <p className="kicker">Retention Intelligence</p>
        <h1>Churn Predictor Dashboard</h1>
        <p className="hero-copy">
          Upload your customer data and get instant churn predictions from your
          ML model.
        </p>
      </section>

      <section className="card">
        <label className="file-label" htmlFor="file-upload">
          Select dataset
        </label>
        <input
          id="file-upload"
          className="file-input"
          type="file"
          onChange={(e) => setFile(e.target.files[0])}
        />
        <p className="file-meta">
          {file ? `Selected: ${file.name}` : "No file selected"}
        </p>

        <button
          className="predict-button"
          onClick={handleUpload}
          disabled={isLoading}
        >
          {isLoading ? "Running Prediction..." : "Upload & Predict"}
        </button>

        {error && <p className="status error">{error}</p>}
      </section>

      <section className="card">
        <div className="results-header">
          <h2>Predictions</h2>
          <span className="badge">{result.length} records</span>
        </div>
        {result.length === 0 ? (
          <p className="status">No predictions yet. Upload a file to begin.</p>
        ) : (
          <ul className="results-grid">
            {result.map((r, i) => (
              <li key={i} className="result-item">
                <span>Customer #{i + 1}</span>
                <strong>{r}</strong>
              </li>
            ))}
          </ul>
        )}
      </section>
    </main>
  );
}

export default App;