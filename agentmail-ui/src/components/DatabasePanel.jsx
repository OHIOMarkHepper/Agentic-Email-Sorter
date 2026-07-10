
import { useState } from "react";
import { saveEmails, getEmailsByCluster, getDbClusters } from "../api";

export default function DatabasePanel() {
  const [data, setData] = useState(null);
  const [emails, setEmails] = useState([]);
    return (
        <div style={{ border: "1px solid #ccc", padding: "1rem", borderRadius: "8px" }}>
            <h2>Database Panel</h2>
        </div>
    );
}