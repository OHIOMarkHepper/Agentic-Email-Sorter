import { useState } from "react";
import { classifyEmail } from "../api";

export default function EmailClassifier() {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  async function handleClassify() {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await classifyEmail(text);
      setResult(res.label);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={{ border: "1px solid #ccc", padding: "1rem", borderRadius: "8px" }}>
      <h2>Classify an Email</h2>
      <textarea
        rows={6}
        style={{ width: "100%" }}
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Paste email text here..."
      />
       <p>
       </p>
      <button onClick={handleClassify} disabled={loading || !text.trim()}>
        {loading ? "Classifying..." : "Classify"}
      </button>

      {result && <p>Predicted cluster: <strong>{result}</strong></p>}
      {error && <p style={{ color: "red" }}>{error}</p>}
    </div>
  );
}
