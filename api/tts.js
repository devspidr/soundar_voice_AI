// api/tts.js
export default async function handler(req, res) {
  if (req.method !== "POST") return res.status(405).json({ error: "POST only" });
  const { text } = req.body || {};
  if (!text) return res.status(400).json({ error: "No text provided" });

  try {
    const openaiResp = await fetch("https://api.openai.com/v1/audio/speech", {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${process.env.OPENAI_API_KEY}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        model: "gpt-4o-mini-tts",
        voice: "alloy",
        input: text
      })
    });

    if (!openaiResp.ok) {
      const errText = await openaiResp.text();
      return res.status(500).json({ error: "TTS API error", details: errText });
    }

    const arrayBuffer = await openaiResp.arrayBuffer();
    const audioBuf = Buffer.from(arrayBuffer);
    res.setHeader("Content-Type", "audio/mpeg");
    res.setHeader("Content-Length", audioBuf.length);
    return res.status(200).send(audioBuf);
  } catch (err) {
    console.error("tts error:", err);
    return res.status(500).json({ error: "TTS failed", details: String(err) });
  }
}
