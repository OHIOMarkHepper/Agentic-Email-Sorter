import { useState } from "react";
import { relabelCluster } from "../api";

export default function ClusterList({ clusters, onClustersChanged }) {
  const [editingId, setEditingId] = useState(null);
  const [newLabel, setNewLabel] = useState("");
  const [error, setError] = useState(null);

  if (!clusters || Object.keys(clusters).length === 0) {
    return <p>No clusters yet. Train a model first.</p>;
  }

  async function handleRelabel(id) {
    setError(null);
    try {
      await relabelCluster(id, newLabel);
      setEditingId(null);
      setNewLabel("");
      onClustersChanged();
    } catch (err) {
      setError(err.message);
    }
  }

  return (
    <div style={{ border: "1px solid #ccc", padding: "1rem", borderRadius: "8px" }}>
      <h2>Clusters</h2>
      {error && <p style={{ color: "red" }}>{error}</p>}
      <ul>
        {Object.entries(clusters).map(([id, name]) => (
          <li key={id} style={{ marginBottom: "0.5rem" }}>
            <strong>[{id}]</strong>{" "}
            {editingId === id ? (
              <>
                <input
                  type="text"
                  value={newLabel}
                  onChange={(e) => setNewLabel(e.target.value)}
                  autoFocus
                />
                <button onClick={() => handleRelabel(id)}>Save</button>
                <button onClick={() => setEditingId(null)}>Cancel</button>
              </>
            ) : (
              <>
                {name}{" "}
                <button onClick={() => { setEditingId(id); setNewLabel(name); }}>
                  Rename
                </button>
              </>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}
