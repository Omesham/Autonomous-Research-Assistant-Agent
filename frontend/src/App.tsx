import { useState } from "react";


type Message = {
  role: "user" | "assistant";
  content: string;
  bullets?: string[];
  followups?: string[];
  sources?: { title: string; url: string }[];
};

function App() {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);

  const sendMessage = async () => {
    if (!input.trim()) return;

    const userMessage: Message = { role: "user", content: input };
    setMessages(prev => [...prev, userMessage]);

    const response = await fetch("http://127.0.0.1:8000/research", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ topic: input })
    });

    const data = await response.json();

    const assistantMessage: Message = {
      role: "assistant",
      content: data.summary,
      bullets: data.bullet_points,
      followups: data.follow_up_questions,
      sources: data.sources
    };

    setMessages(prev => [...prev, assistantMessage]);
    setInput("");
  };

  return (
    <div style={{ padding: 20, maxWidth: 800, margin: "auto" }}>
      <h1>Research Agent</h1>

      <div style={{ marginBottom: 20 }}>
        {messages.map((msg, index) => (
          <div key={index} style={{ marginBottom: 15 }}>
            <strong>{msg.role === "user" ? "You" : "Assistant"}:</strong>
            <p>{msg.content}</p>

            {msg.bullets && (
              <ul>
                {msg.bullets.map((b, i) => (
                  <li key={i}>{b}</li>
                ))}
              </ul>
            )}

            {msg.followups && (
              <div>
                <strong>Follow-up Questions:</strong>
                <ul>
                  {msg.followups.map((f, i) => (
                    <li key={i}>{f}</li>
                  ))}
                </ul>
              </div>
            )}

            {msg.sources && (
              <div>
                <strong>Sources:</strong>
                <ul>
                  {msg.sources.map((s, i) => (
                    <li key={i}>
                      <a href={s.url} target="_blank" rel="noreferrer">
                        {s.title}
                      </a>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        ))}
      </div>

      <input
        value={input}
        onChange={e => setInput(e.target.value)}
        style={{ width: "70%", padding: 8 }}
        placeholder="Ask a research question..."
      />
      <button onClick={sendMessage} style={{ padding: "8px 15px" }}>
        Send
      </button>
    </div>
  );
}

export default App;