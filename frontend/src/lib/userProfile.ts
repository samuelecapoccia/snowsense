/**
 * Componente 5 — Personalizzazione Implicita (lato frontend)
 *
 * Gestisce il profilo utente anonimo nel localStorage.
 * Nessun login, nessun server, nessun account.
 *
 * Il profilo viene:
 *   - caricato ad ogni sessione
 *   - aggiornato dopo ogni ricerca (updateProfile)
 *   - inviato al backend come contesto opzionale (getProfileContext)
 *
 * Uso:
 *   import { loadProfile, updateProfile, getProfileContext } from "@/lib/userProfile";
 *
 *   const profile = loadProfile();
 *   const updatedProfile = updateProfile(profile, intent, resorts);
 *   saveProfile(updatedProfile);
 *
 *   // Nel payload API:
 *   const payload = { query, user_profile: getProfileContext(profile) };
 */

const STORAGE_KEY = "snowsense_user_profile";
const MAX_HISTORY  = 20;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type SkiLevel      = "beginner" | "intermediate" | "expert";
export type Budget        = "low" | "medium" | "high";
export type ExperienceType = "family" | "apres_ski" | "sport" | "panoramic";

export interface Intent {
  level?:            SkiLevel | null;
  budget?:           Budget | null;
  city_reference?:   string | null;
  country?:          string | null;
  experience_type?:  ExperienceType | null;
  snowpark?:         boolean | null;
  night_ski?:        boolean | null;
  needs_clarification?: boolean;
}

export interface ResortResult {
  Resort:       string;
  Country:      string;
  is_exploration?: boolean;
  [key: string]: unknown;
}

export interface QueryHistoryEntry {
  query:     string;
  timestamp: string; // ISO
}

export interface UserProfile {
  level_counts:      Partial<Record<SkiLevel, number>>;
  budget_counts:     Partial<Record<Budget, number>>;
  country_counts:    Record<string, number>;
  experience_counts: Partial<Record<ExperienceType, number>>;
  snowpark_pref:     boolean | null;
  night_ski_pref:    boolean | null;
  query_history:     QueryHistoryEntry[];
  session_count:     number;
  last_updated:      string | null; // ISO
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

export function emptyProfile(): UserProfile {
  return {
    level_counts:      {},
    budget_counts:     {},
    country_counts:    {},
    experience_counts: {},
    snowpark_pref:     null,
    night_ski_pref:    null,
    query_history:     [],
    session_count:     0,
    last_updated:      null,
  };
}

// ---------------------------------------------------------------------------
// Persistence
// ---------------------------------------------------------------------------

export function loadProfile(): UserProfile {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return emptyProfile();
    const parsed = JSON.parse(raw) as Partial<UserProfile>;
    // Merge with empty to ensure all fields exist after schema changes
    return { ...emptyProfile(), ...parsed };
  } catch {
    return emptyProfile();
  }
}

export function saveProfile(profile: UserProfile): void {
  try {
    const updated: UserProfile = {
      ...profile,
      last_updated: new Date().toISOString(),
    };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(updated));
  } catch {
    // localStorage quota exceeded or unavailable — fail silently
  }
}

export function clearProfile(): void {
  localStorage.removeItem(STORAGE_KEY);
}

// ---------------------------------------------------------------------------
// Update after search
// ---------------------------------------------------------------------------

/**
 * Aggiorna il profilo con i dati della ricerca appena completata.
 * Chiamare dopo ogni risposta dal backend, prima di saveProfile().
 *
 * I resort con is_exploration=true non influenzano i conteggi paese
 * (altrimenti distorcerebbero il profilo dominante).
 */
export function updateProfile(
  profile: UserProfile,
  intent: Intent,
  resorts: ResortResult[],
  query: string,
): UserProfile {
  const p: UserProfile = {
    level_counts:      { ...profile.level_counts },
    budget_counts:     { ...profile.budget_counts },
    country_counts:    { ...profile.country_counts },
    experience_counts: { ...profile.experience_counts },
    snowpark_pref:     profile.snowpark_pref,
    night_ski_pref:    profile.night_ski_pref,
    query_history:     [...profile.query_history],
    session_count:     profile.session_count,
    last_updated:      profile.last_updated,
  };

  // Conteggi categorici
  if (intent.level) {
    p.level_counts[intent.level] = (p.level_counts[intent.level] ?? 0) + 1;
  }
  if (intent.budget) {
    p.budget_counts[intent.budget] = (p.budget_counts[intent.budget] ?? 0) + 1;
  }
  if (intent.experience_type) {
    p.experience_counts[intent.experience_type] =
      (p.experience_counts[intent.experience_type] ?? 0) + 1;
  }

  // Paesi dai resort raccomandati (escludi exploration)
  for (const resort of resorts) {
    if (!resort.is_exploration && resort.Country) {
      p.country_counts[resort.Country] = (p.country_counts[resort.Country] ?? 0) + 1;
    }
  }

  // Preferenze booleane: majority voting semplice (ultimo valore esplicito vince)
  if (intent.snowpark !== undefined && intent.snowpark !== null) {
    p.snowpark_pref = intent.snowpark;
  }
  if (intent.night_ski !== undefined && intent.night_ski !== null) {
    p.night_ski_pref = intent.night_ski;
  }

  // Cronologia query (max MAX_HISTORY)
  p.query_history = [
    ...p.query_history.slice(-(MAX_HISTORY - 1)),
    { query, timestamp: new Date().toISOString() },
  ];

  p.session_count += 1;

  return p;
}

// ---------------------------------------------------------------------------
// Context for API
// ---------------------------------------------------------------------------

/**
 * Serializza il profilo nel formato atteso dal backend (FastAPI).
 * Restituisce null se il profilo ha meno di 2 sessioni (contesto insufficiente).
 */
export function getProfileContext(profile: UserProfile): UserProfile | null {
  if (profile.session_count < 2) return null;
  return profile;
}

// ---------------------------------------------------------------------------
// Read helpers
// ---------------------------------------------------------------------------

/** Categoria dominante (più frequente) in un conteggio, o null. */
export function dominant<T extends string>(
  counts: Partial<Record<T, number>>
): T | null {
  const entries = Object.entries(counts) as [T, number][];
  if (entries.length === 0) return null;
  return entries.reduce((a, b) => (b[1] > a[1] ? b : a))[0];
}

/** Top-N categorie per frequenza. */
export function topN<T extends string>(
  counts: Partial<Record<T, number>>,
  n: number
): T[] {
  return (Object.entries(counts) as [T, number][])
    .sort((a, b) => b[1] - a[1])
    .slice(0, n)
    .map(([k]) => k);
}
