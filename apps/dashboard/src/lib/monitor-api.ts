import type { NodeRole, RunAction, RunState, TelemetryEventType } from '@/types/dashboard'

const DEFAULT_BASE_PATH = '/monitor'
const DEFAULT_RUN_ID = 'local-run'

export interface MonitorTelemetryEvent {
  event_id: string
  ts: string
  run_id: string
  round: number | null
  node_id: string
  role: NodeRole
  event_type: TelemetryEventType
  status: string
  latency_ms: number | null
  payload_bytes: number | null
  metrics: Record<string, unknown> | null
  details: Record<string, unknown> | null
}

export interface MonitorRunStateSnapshot {
  run_id: string
  state: RunState
  updated_at: string
}

export interface MonitorSnapshotResponse {
  run: MonitorRunStateSnapshot
  recent_events: MonitorTelemetryEvent[]
}

export type MonitorWsMessage =
  | { type: 'event'; payload: MonitorTelemetryEvent }
  | { type: 'run_state'; payload: MonitorRunStateSnapshot }

function trimTrailingSlash(value: string) {
  return value.length > 1 ? value.replace(/\/+$/, '') : value
}

function parseRunState(value: string): RunState {
  if (value === 'idle' || value === 'running' || value === 'paused' || value === 'stopped' || value === 'error') {
    return value
  }
  return 'error'
}

function resolveRunIdFromUrl() {
  const params = new URLSearchParams(window.location.search)
  return params.get('run_id') ?? params.get('runId')
}

export function resolveMonitorRunId() {
  return (resolveRunIdFromUrl() ?? import.meta.env.VITE_MONITOR_RUN_ID ?? DEFAULT_RUN_ID).trim()
}

export function resolveMonitorBaseUrl() {
  const base = (import.meta.env.VITE_MONITOR_BASE_URL ?? DEFAULT_BASE_PATH).trim() || DEFAULT_BASE_PATH
  return trimTrailingSlash(base)
}

function snapshotUrl(baseUrl: string, runId: string, limit: number) {
  const url = new URL(`${trimTrailingSlash(baseUrl)}/snapshot`, window.location.origin)
  url.searchParams.set('run_id', runId)
  url.searchParams.set('limit', String(limit))
  return url.toString()
}

export async function fetchMonitorSnapshot(
  {
    baseUrl,
    runId,
    limit,
    signal,
  }: {
    baseUrl: string
    runId: string
    limit: number
    signal?: AbortSignal
  },
) {
  const response = await fetch(snapshotUrl(baseUrl, runId, limit), {
    method: 'GET',
    signal,
  })
  if (!response.ok) {
    throw new Error(`snapshot request failed with status ${response.status}`)
  }
  const data = (await response.json()) as MonitorSnapshotResponse
  data.run.state = parseRunState(data.run.state)
  return data
}

interface ControlResponse {
  run_id: string
  state: string
}

export async function postMonitorControl(
  {
    baseUrl,
    runId,
    action,
    reason,
    signal,
  }: {
    baseUrl: string
    runId: string
    action: RunAction
    reason?: string
    signal?: AbortSignal
  },
) {
  const url = new URL(`${trimTrailingSlash(baseUrl)}/control/${action}`, window.location.origin)
  const response = await fetch(url.toString(), {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      run_id: runId,
      reason,
    }),
    signal,
  })
  if (!response.ok) {
    let detail = `${response.status}`
    try {
      const payload = (await response.json()) as { detail?: string }
      detail = payload.detail ?? detail
    } catch {
      // keep status-only detail for non-json responses
    }
    throw new Error(`control ${action} failed: ${detail}`)
  }
  const data = (await response.json()) as ControlResponse
  return {
    run_id: data.run_id,
    state: parseRunState(data.state),
  }
}

function toAbsoluteUrl(input: string) {
  return new URL(input, window.location.origin)
}

export function resolveMonitorWsUrl(baseUrl: string) {
  const explicit = import.meta.env.VITE_MONITOR_WS_URL
  if (explicit && explicit.trim()) {
    return explicit.trim()
  }

  const httpUrl = toAbsoluteUrl(baseUrl)
  const wsScheme = httpUrl.protocol === 'https:' ? 'wss:' : 'ws:'
  return `${wsScheme}//${httpUrl.host}${trimTrailingSlash(httpUrl.pathname)}/ws/events`
}
