interface Props {
  message: string;
  onReply: (reply: string) => void;
  loading: boolean;
}

export default function ClarificationBubble({ message, onReply, loading }: Props) {
  return (
    <div className="clarification-section">
      <div className="chat-bubble ai">
        <span className="bubble-icon">✦</span>
        <p>{message}</p>
      </div>
      <div className="chat-input-row">
        <input
          className="chat-input"
          placeholder="Rispondi qui..."
          autoFocus
          onKeyDown={e => {
            if (e.key === 'Enter' && (e.target as HTMLInputElement).value.trim()) {
              onReply((e.target as HTMLInputElement).value.trim());
            }
          }}
          disabled={loading}
        />
        <button
          className="chat-send-btn"
          disabled={loading}
          onClick={e => {
            const input = (e.currentTarget.previousSibling as HTMLInputElement);
            if (input.value.trim()) onReply(input.value.trim());
          }}
        >
          {loading ? <span className="spinner small" /> : '↵'}
        </button>
      </div>
    </div>
  );
}
