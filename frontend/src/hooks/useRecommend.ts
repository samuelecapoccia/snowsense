import { useState } from 'react';
import type { ApiResponse } from '../types/api';
import { loadProfile, saveProfile, updateProfile, getProfileContext } from '../lib/userProfile';

const API_BASE = 'http://localhost:8000/api';

export type AppState =
  | { phase: 'idle' }
  | { phase: 'loading' }
  | { phase: 'clarification'; message: string; missing: string[] }
  | { phase: 'results'; data: Extract<ApiResponse, { status: 'ok' }>; query: string }
  | { phase: 'error'; message: string };

export function useRecommend() {
  const [state, setState] = useState<AppState>({ phase: 'idle' });

  const search = async (query: string) => {
    if (!query.trim()) return;
    setState({ phase: 'loading' });

    const profile = loadProfile();
    const profileCtx = getProfileContext(profile);

    try {
      const res = await fetch(`${API_BASE}/recommend`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query,
          strategy: 'few_shot',
          user_profile: profileCtx,
        }),
      });

      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data: ApiResponse = await res.json();

      if (data.status === 'clarification_needed') {
        setState({ phase: 'clarification', message: data.message, missing: data.missing_fields });
        return;
      }

      // Update profile
      const updated = updateProfile(
        profile,
        data.intent as unknown as import('../lib/userProfile').Intent,
        data.resorts as unknown as import('../lib/userProfile').ResortResult[],
        query
      );
      saveProfile(updated);

      setState({ phase: 'results', data, query });
    } catch (err) {
      setState({ phase: 'error', message: err instanceof Error ? err.message : 'Errore sconosciuto' });
    }
  };

  const reset = () => setState({ phase: 'idle' });

  return { state, search, reset };
}
