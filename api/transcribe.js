// api/transcribe.js
import formidable from "formidable";

export const config = {
  api: { bodyParser: false }
};

function parseForm(req) {
  return new Promise((resolve, reject) => {
    const form = new formidable.IncomingForm();
    form.parse(req, (err, fields, files) => {
      if (err) reject(err); else resolve({ fields, files });
    });
  });
}

export default async function handler(req, res) {
  if (req.method !== "POST") return res.status(405).json({ error: "POST only" });
  try {
    const { files } = await parseForm(req);
    const audio = files?.file;
    if (!audio) return res.status(400).json({ error: "No audio file uploaded" });

    // stream the file to OpenAI's transcription endpoint via multipart/form-data
    const form = new FormData();
    form.append("file", await fsReadFileAsBlob(audio.filepath || audio.path), audio.originalFilename || "audio.webm");
    // optional: model name
    form.append("model", "whisper-1");

    const resp = await fetch("https://api.openai.com/v1/audio/transcriptions", {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${process.env.OPENAI_API_KEY}`
      },
      body: form
    });
    const data = await resp.json();
    return res.status(200).json(data);
  } catch (err) {
    console.error("transcribe error:", err);
    return res.status(500).json({ error: "Transcription failed", details: String(err) });
  }
}

// helper to read file as Blob for fetch. Node 18+ supports Blob.from, but we implement safely:
import fs from "fs/promises";
import path from "path";

async function fsReadFileAsBlob(filepath) {
  const buffer = await fs.readFile(filepath);
  // build a File-like object accepted by form-data append:
  const b = buffer;
  // create a simple Blob using Web Streams if available, otherwise pass Buffer (node-fetch / undici supports Buffer)
  return b;
}
