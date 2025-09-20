import React, { useState } from 'react';

// Usage: <ClientExample apiUrl="http://127.0.0.1:8000/measure-hand" />
export default function ClientExample({ apiUrl = 'http://127.0.0.1:8000/measure-hand' }) {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleFile = (e) => {
    setFile(e.target.files[0]);
    setResult(null);
    setError(null);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) return setError('Please select an image file');

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const res = await fetch(apiUrl, {
        method: 'POST',
        body: formData,
      });

      const json = await res.json();

      if (!res.ok) {
        setError(json.error || `Request failed with status ${res.status}`);
      } else {
        setResult(json);
      }
    } catch (err) {
      setError(String(err));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: 800, margin: '0 auto' }}>
      <h3>Hand Measurement API Client</h3>
      <form onSubmit={handleSubmit}>
        <input type="file" accept="image/*" onChange={handleFile} />
        <button type="submit" disabled={loading} style={{ marginLeft: 8 }}>
          {loading ? 'Uploading...' : 'Upload & Measure'}
        </button>
      </form>

      {error && <div style={{ color: 'red', marginTop: 12 }}>{error}</div>}

      {result && (
        <div style={{ marginTop: 16 }}>
          <h4>Result JSON</h4>
          <pre style={{ maxHeight: 300, overflow: 'auto', background: '#f5f5f5', padding: 8 }}>{JSON.stringify(result, null, 2)}</pre>

          {result.annotated_image_b64 && (
            <div style={{ marginTop: 12 }}>
              <h4>Annotated Image</h4>
              <img
                src={`data:image/png;base64,${result.annotated_image_b64}`}
                alt="Annotated"
                style={{ maxWidth: '100%', height: 'auto', border: '1px solid #ccc' }}
              />
            </div>
          )}
        </div>
      )}
    </div>
  );
}
