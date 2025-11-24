// api/voice_text.js
export default async function handler(req, res) {
  if (req.method !== "POST") return res.status(405).json({ error: "POST only" });

  try {
    const body = req.body || {};
    const transcript = body.transcript || "";
    if (!transcript) return res.status(400).json({ error: "No transcript provided" });

    // Build the conversation messages (you may adapt persona as needed).
    const messages = [
      { role: "system", content: "You are Soundar's friendly voice assistant. Keep replies helpful and concise." },
      { role: "user", content: transcript }
    ];

    // Call OpenAI Chat Completions
    const openaiResp = await fetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${process.env.OPENAI_API_KEY}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        model: "gpt-4o-mini",
        messages,
        max_tokens: 300,
        temperature: 0.7
      })
    });

    if (!openaiResp.ok) {
      const txt = await openaiResp.text();
      console.error("OpenAI error:", openaiResp.status, txt);
      return res.status(502).json({ error: "OpenAI API error", details: txt });
    }

    const data = await openaiResp.json();
    const reply = data?.choices?.[0]?.message?.content ?? (data?.error || "No reply");

    return res.status(200).json({ text: reply });
  } catch (err) {
    console.error("voice_text error:", err);
    return res.status(500).json({ error: "Internal server error", details: String(err) });
  }
}
