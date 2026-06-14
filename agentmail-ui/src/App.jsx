import { useState } from "react";
import { getClusters } from "./api";
import TrainPanel from "./components/TrainPanel";
import ClusterList from "./components/ClusterList";
import EmailClassifier from "./components/EmailClassifier";

export default function App() {
  const [clusters, setClusters] = useState(null);
  const [trained, setTrained] = useState(false);

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

      <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
        <TrainPanel onTrained={handleTrained} />

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
