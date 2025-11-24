// api/chat.js
export default async function handler(req, res) {
  if (req.method !== "POST") return res.status(405).json({error:"POST only"});
  const { message } = req.body || {};
  if (!message) return res.status(400).json({ error: "No message provided" });

  try {
    const openaiResp = await fetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${process.env.OPENAI_API_KEY}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        model: "gpt-4o-mini",
        messages: [
          { role: "system", content: "You are Soundar's friendly voice assistant." },
          { role: "user", content: message }
        ],
        max_tokens: 400
      })
    });

    const data = await openaiResp.json();
    const reply = data?.choices?.[0]?.message?.content ?? (data?.error || "No reply");
    return res.status(200).json({ reply });
  } catch (err) {
    console.error("chat error:", err);
    return res.status(500).json({ error: "Chat failed", details: String(err) });
  }
}
