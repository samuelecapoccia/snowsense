import { useState, useRef } from 'react';
import type { KeyboardEvent } from 'react';

const SUGGESTIONS: { label: string; query: string }[] = [
  {
    label: 'Famiglia con bambini vicino a Roma',
    query: 'Siamo una famiglia con bambini piccoli e cerchiamo un resort sciistico non troppo lontano da Roma, con tante piste facili e un ambiente adatto ai bambini.',
  },
  {
    label: 'Piste expert in Austria',
    query: 'Sono uno sciatore esperto e cerco piste tecniche e impegnative in Austria, possibilmente con dislivello elevato e fuori pista accessibile.',
  },
  {
    label: 'Budget freestyle in Francia',
    query: 'Cerco un resort economico in Francia con un buon snowpark per fare freestyle, il budget per il giornaliero deve essere basso.',
  },
  {
    label: 'Snowpark & vita notturna',
    query: 'Voglio un resort con uno snowpark ben attrezzato e possibilità di sciare di notte, cercando anche un buon après-ski e vita notturna.',
  },
  {
    label: 'Ghiacciaio e fuori pista',
    query: 'Vorrei sciare su un ghiacciaio con quota elevata e possibilità di fare fuori pista, sono uno sciatore di livello intermedio-avanzato.',
  },
];

interface Props {
  onSearch: (query: string) => void;
  loading: boolean;
  initialValue?: string;
}

export default function SearchBar({ onSearch, loading, initialValue = '' }: Props) {
  const [value, setValue] = useState(initialValue);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const submit = () => { if (value.trim() && !loading) onSearch(value.trim()); };

  const handleKey = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submit(); }
  };

  const resize = (el: HTMLTextAreaElement) => {
    el.style.height = 'auto';
    el.style.height = `${el.scrollHeight}px`;
  };

  const handleSuggestion = (query: string) => {
    setValue(query);
    const el = textareaRef.current;
    if (el) { el.focus(); setTimeout(() => resize(el), 0); }
  };

  return (
    <div className="search-section">
      <div className="search-box">
        <textarea
          ref={textareaRef}
          className="search-input"
          placeholder="Descrivi la tua vacanza ideale — livello, gruppo, atmosfera..."
          value={value}
          onChange={e => { setValue(e.target.value); resize(e.target); }}
          onKeyDown={handleKey}
          rows={2}
          disabled={loading}
        />
        <button
          className="search-btn"
          onClick={submit}
          disabled={loading || !value.trim()}
        >
          {loading ? <span className="spinner" /> : 'Cerca →'}
        </button>
      </div>

      <div className="suggestions">
        {SUGGESTIONS.map(s => (
          <button
            key={s.label}
            className="suggestion-chip"
            onClick={() => handleSuggestion(s.query)}
            disabled={loading}
          >
            {s.label}
          </button>
        ))}
      </div>
    </div>
  );
}
