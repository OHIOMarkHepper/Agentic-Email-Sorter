import { useState } from "react";
import { getClusters } from "./api";
import TrainPanel from "./components/TrainPanel";
import ClusterList from "./components/ClusterList";
import EmailClassifier from "./components/EmailClassifier";
import DatabasePanel from "./components/DatabasePanel";

export default function App() {
  const [clusters, setClusters] = useState(null);
  const [trained, setTrained] = useState(true);

  async function refreshClusters() {
    try {
      const data = await getClusters();
      setClusters(data);
    } catch (err) {
      console.error("Failed to fetch clusters:", err.message);
    }
  }

  async function handleTrained() {
    setTrained(true);
    await refreshClusters();
  }

  return (
    <div style={{ maxWidth: "700px", margin: "2rem auto", fontFamily: "sans-serif" }}>
      <h1>AgentMail</h1>

      <div style={{ display: "flex", flexDirection: "row", gap: "1rem" }}>
        <TrainPanel onTrained={handleTrained} />
        <DatabasePanel />
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: "1rem", marginTop: "1rem" }}>
        {trained && (
          <>
            <ClusterList clusters={clusters} onClustersChanged={refreshClusters} />
            <EmailClassifier />
          </>
        )}
      </div>
    </div>
  );
}