import { useState } from "react";
import { trainModel } from "../api";

export default function TrainPanel({ onTrained }) {
  
  const [filepath, setFilepath] = useState("./emaildata/email_classification_dataset.csv");
  const [strategy, setStrategy] = useState("kmeans");
  const [useAiLabels, setUseAiLabels] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  async function handleTrain() {
    setLoading(true);
    setError(null);
    try {
      const result = await trainModel({ filepath, strategy, useAiLabels });
      onTrained(result);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  /// The main UI for training is here
  return (
    <div style={{ border: "1px solid #ccc", padding: "1rem", borderRadius: "8px" }}>
      <h2>Train Model</h2>
      <label style={{ display: "block", marginBottom: "0.5rem" }}>
        Data file path:
        <input
          type="text"
          value={filepath}
          onChange={(e) => setFilepath(e.target.value)}
          style={{ width: "100%" }}
        />
      </label>

      <label style={{ display: "block", marginBottom: "0.5rem" }}>
        Strategy:
        <select value={strategy} onChange={(e) => setStrategy(e.target.value)}>
          <option value="kmeans">KMeans</option>
          <option value="user_defined">User Defined</option>
        </select>
      </label>

      <label style={{ display: "block", marginBottom: "0.5rem" }}>
        <input
          type="checkbox"
          checked={useAiLabels}
          onChange={(e) => setUseAiLabels(e.target.checked)}
        />
        {" "}Use AI-generated cluster labels (slower, requires Ollama)
      </label>

      <button onClick={handleTrain} disabled={loading}>
        {loading ? "Training... this may take a moment" : "Train Model"}
      </button>

      {error && <p style={{ color: "red" }}>{error}</p>}
    </div>
  );
}
