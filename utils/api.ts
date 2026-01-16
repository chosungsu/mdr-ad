// 백엔드 API 클라이언트 유틸리티 (Vite 기준)
// - Base URL: `VITE_API_BASE` (예: http://localhost:8000)
// - 엔드포인트 오버라이드: `VITE_API_ENDPOINTS_JSON`
//   예: {"health":"/health","logs":"/logs","realtime":"/realtime/data","modelsList":"/models/list"}

type ApiEndpoints = {
  health: string;
  logs: string;
  realtimeScores: string;
  modelsList: string;
};

const DEFAULT_ENDPOINTS: ApiEndpoints = {
  health: "/health",
  logs: "/logs",
  realtimeScores: "/realtime/scores",
  modelsList: "/models/list",
};

function parseEndpointsJson(raw: string | undefined): Partial<ApiEndpoints> | null {
  if (!raw) return null;
  try {
    const parsed = JSON.parse(raw) as Partial<ApiEndpoints>;
    return parsed && typeof parsed === "object" ? parsed : null;
  } catch {
    return null;
  }
}

// Vite에서는 환경변수가 import.meta.env에 들어갑니다.
const env = (import.meta as any).env as Record<string, string | undefined> | undefined;

// Environment-based API base URL:
// - Local development: http://localhost:8000 (direct to backend, no proxy)
// - Production (Render): /api (through Nginx proxy)
// Override with VITE_API_BASE environment variable
export const API_BASE_URL =
  env?.VITE_API_BASE?.replace(/\/$/, "") || 
  (env?.MODE === "development" ? "http://localhost:8000" : "/api");

export const API_ENDPOINTS: ApiEndpoints = {
  ...DEFAULT_ENDPOINTS,
  ...(parseEndpointsJson(env?.VITE_API_ENDPOINTS_JSON) || {}),
};

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  // Increase timeout for production (Render + Cloudflare can be slow)
  // Local: 2.5s, Production: 10s
  const timeoutMs = Number(env?.VITE_API_TIMEOUT_MS || (env?.MODE === "development" ? "2500" : "10000"));
  const controller = new AbortController();
  const timeoutId = window.setTimeout(() => controller.abort(), timeoutMs);

  try {
    const res = await fetch(`${API_BASE_URL}${path}`, {
      ...init,
      signal: controller.signal,
      headers: {
        ...(init?.headers || {}),
        ...(init?.body instanceof FormData
          ? {}
          : { "Content-Type": "application/json" }),
      },
    }).finally(() => {
      window.clearTimeout(timeoutId);
    });

    if (!res.ok) {
      const text = await res.text();
      throw new Error(text || res.statusText);
    }
    return (await res.json()) as T;
  } catch (error: any) {
    // Better error messages for debugging
    if (error.name === 'AbortError') {
      throw new Error(`Request timeout after ${timeoutMs}ms: ${API_BASE_URL}${path}`);
    }
    throw error;
  }
}

export const backendApi = {
  // 서버 상태 health 조회
  getHealth: () => {
    return apiFetch<{
      status: string;
      model_loaded: boolean;
      timestamp: string;
      model_path: string | null;
      system?: {
        cpu_percent?: number;
        memory_percent?: number;
        memory_available_gb?: number;
        disk_percent?: number;
        disk_free_gb?: number;
        platform?: string;
        python_version?: string;
      };
      uptime_seconds?: number;
      error?: string;
    }>(API_ENDPOINTS.health);
  },
  // 저장된 모델 목록 조회
  getModelList: () => {
    return apiFetch<{
      success: boolean;
      models: string[];
      model_dir: string;
      count: number;
    }>(API_ENDPOINTS.modelsList);
  },
  // 모델별 anomaly score 조회 (mscv ae / tcad)
  getRealtimeScores: () => {
    return apiFetch<{
      timestamp: string;
      cursor: number;
      scores: { mdrad: number | null };
      status: "success" | "error";
      error?: string | null;
    }>(API_ENDPOINTS.realtimeScores);
  },
  // 시스템 로그 조회
  getSystemLogs: (opts?: { limit?: number; cursor?: number; wrap?: boolean }) => {
    const search = new URLSearchParams();
    if (opts?.limit !== undefined) search.set("limit", String(opts.limit));
    if (opts?.cursor !== undefined) search.set("cursor", String(opts.cursor));
    if (opts?.wrap !== undefined) search.set("wrap", opts.wrap ? "true" : "false");
    const qs = search.toString();
    return apiFetch<{
      success: boolean;
      logs: Array<{
        id?: number;
        timestamp: string;
        level: "info" | "warning" | "error" | "success";
        message: string;
        source: string;
      }>;
      count: number;
      message: string;
      last_id?: number;
      next_cursor?: number;
      wrapped?: boolean;
    }>(`${API_ENDPOINTS.logs}${qs ? `?${qs}` : ""}`);
  },
};
