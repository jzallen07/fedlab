export type RunState = 'idle' | 'running' | 'paused' | 'stopped' | 'error'
export type RunAction = 'start' | 'pause' | 'resume' | 'stop'
export type ConnectionState = 'connecting' | 'connected' | 'disconnected' | 'error'
export type NodeRole = 'server' | 'client'
export type EventLevel = 'info' | 'warn' | 'error'

export type NodeStatus = 'idle' | 'training' | 'uploading' | 'aggregating' | 'error'

export type TelemetryEventType =
  | 'node_heartbeat'
  | 'round_started'
  | 'model_dispatched'
  | 'client_train_started'
  | 'client_train_completed'
  | 'client_update_uploaded'
  | 'aggregation_started'
  | 'aggregation_completed'
  | 'round_completed'
  | 'node_error'
  | 'run_requested'
  | 'run_paused'
  | 'run_resumed'
  | 'run_stopped'

export interface ClientSnapshot {
  id: string
  label: string
  status: NodeStatus
  lastAccuracy: number
  lastLatencyMs: number
  lastLoss: number
  lastRound: number
  lastEventType: TelemetryEventType | null
}

export interface RoundPoint {
  round: number
  globalLoss: number
  globalAccuracy: number
  serverLatencyMs: number
  ts: string
}

export interface EventEntry {
  id: string
  ts: string
  type: TelemetryEventType
  level: EventLevel
  message: string
  runId: string
  nodeId: string
  role: NodeRole
  round: number | null
  status: string
  latencyMs: number | null
  payloadBytes: number | null
  metrics: Record<string, unknown> | null
  details: Record<string, unknown> | null
}

export interface NodeDetail {
  id: string
  label: string
  role: NodeRole
  status: NodeStatus
  lastEventType: TelemetryEventType | null
  lastSeenAt: string | null
  lastRound: number
  eventCount: number
  avgLatencyMs: number | null
  accuracy: number | null
  loss: number | null
  lastMessage: string | null
}
