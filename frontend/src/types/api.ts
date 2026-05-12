export interface ResortScores {
  embedding: number;
  mlp: number;
  distance: number;
  final: number;
}

export interface Resort {
  resort_id: number;
  Resort: string;
  Country: string;
  HighestPoint: number;
  LowestPoint: number;
  DayPassPriceAdult: number;
  TotalSlope: number;
  BeginnerSlope: number;
  IntermediateSlope: number;
  DifficultSlope: number;
  Snowparks: string;
  NightSki: string;
  TotalLifts: number;
  LiftCapacity: number;
  SnowCannons: number;
  Latitude: number | null;
  Longitude: number | null;
  distance_km: number | null;
  scores: ResortScores;
  explanation: string | null;
  is_exploration: boolean;
  exploration_reason: string | null;
}

export interface ExtractedIntent {
  level: string | null;
  budget: string | null;
  city_reference: string | null;
  country: string | null;
  experience_type: string | null;
  snowpark: boolean | null;
  night_ski: boolean | null;
  needs_clarification: boolean;
  missing_fields: string[];
}

export interface RecommendResponse {
  status: 'ok';
  intent: ExtractedIntent;
  resorts: Resort[];
}

export interface ClarificationResponse {
  status: 'clarification_needed';
  missing_fields: string[];
  message: string;
}

export type ApiResponse = RecommendResponse | ClarificationResponse;
