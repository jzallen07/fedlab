import { useCallback, useEffect, useMemo, useState } from 'react'

import {
  fetchMonitorSnapshot,
  postMonitorControl,
  resolveMonitorBaseUrl,
  resolveMonitorRunId,
  resolveMonitorWsUrl,
  type MonitorTelemetryEvent,
  type MonitorWsMessage,
} from '@/lib/monitor-api'
import type {
  ClientSnapshot,
  ConnectionState,
  EventEntry,
  NodeDetail,
  NodeStatus,
  RoundPoint,
  RunAction,
  RunState,
  TelemetryEventType,
} from '@/types/dashboard'

const MAX_EVENTS = 200
const MAX_ROUNDS = 40
const SNAPSHOT_LIMIT = 200
const RECONNECT_DELAY_MS = 1500
const PING_INTERVAL_MS = 20000
const DEFAULT_CLIENT_IDS = ['client_0', 'client_1', 'client_2']
const CLIENT_LABELS = ['North Site', 'Coastal Site', 'Metro Site']

const VALID_TRANSITIONS: Record<RunState, RunAction[]> = {
  idle: ['start'],
  running: ['pause', 'stop'],
  paused: ['resume', 'stop'],
  stopped: ['start'],
  error: ['start', 'stop'],
}

interface DashboardState {
  runState: RunState
  currentRound: number
  clients: ClientSnapshot[]
  rounds: RoundPoint[]
  events: EventEntry[]
  activeClientId: string
  hydrated: boolean
}

function formatClock(ts: string) {
  const date = new Date(ts)
  if (Number.isNaN(date.getTime())) {
    return ts
  }
  return date.toLocaleTimeString('en-US', { hour12: false })
}

function asNumber(value: unknown) {
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

function numberFromMetrics(metrics: Record<string, unknown> | null, keys: string[]) {
  if (!metrics) {
    return null
  }
  for (const key of keys) {
    const value = asNumber(metrics[key])
    if (value !== null) {
      return value
    }
  }
  return null
}

function clientLabel(clientId: string) {
  const match = /(\d+)$/.exec(clientId)
  if (!match) {
    return clientId.replace(/_/g, ' ')
  }
  const idx = Number(match[1])
  return CLIENT_LABELS[idx] ?? `Site ${idx + 1}`
}

function seedClients() {
  return DEFAULT_CLIENT_IDS.map((id, idx) => ({
    id,
    label: CLIENT_LABELS[idx],
    status: 'idle' as const,
    lastAccuracy: 0,
    lastLatencyMs: 0,
    lastLoss: 0,
    lastRound: 0,
    lastEventType: null,
  }))
}

function seedState(): DashboardState {
  return {
    runState: 'idle',
    currentRound: 0,
    clients: seedClients(),
    rounds: [],
    events: [],
    activeClientId: DEFAULT_CLIENT_IDS[0],
    hydrated: false,
  }
}

function eventLevel(event: MonitorTelemetryEvent): EventEntry['level'] {
  if (event.event_type === 'node_error' || event.status === 'error') {
    return 'error'
  }
  if (
    event.event_type === 'aggregation_started' ||
    event.event_type === 'run_paused' ||
    event.event_type === 'run_stopped'
  ) {
    return 'warn'
  }
  return 'info'
}

function eventMessage(event: MonitorTelemetryEvent) {
  const roundText = event.round ? `round ${event.round}` : 'run'
  const node = event.node_id === 'operator' ? 'operator' : event.node_id
  switch (event.event_type) {
    case 'node_heartbeat':
      return `${node} heartbeat (${event.status}).`
    case 'round_started':
      return `Server started ${roundText}.`
    case 'model_dispatched':
      return `Server dispatched model for ${roundText}.`
    case 'client_train_started':
      return `${node} started local training for ${roundText}.`
    case 'client_train_completed':
      return `${node} completed local training for ${roundText}.`
    case 'client_update_uploaded':
      return `${node} uploaded model update for ${roundText}.`
    case 'aggregation_started':
      return `Server started aggregation for ${roundText}.`
    case 'aggregation_completed':
      return `Server completed aggregation for ${roundText}.`
    case 'round_completed':
      return `Server completed ${roundText}.`
    case 'run_requested':
      return `Run requested by operator.`
    case 'run_paused':
      return `Run paused by operator.`
    case 'run_resumed':
      return `Run resumed by operator.`
    case 'run_stopped':
      return `Run stopped by operator.`
    case 'node_error':
      return `${node} reported an error.`
    default:
      return `${event.event_type} from ${node}.`
  }
}

function toEventEntry(event: MonitorTelemetryEvent): EventEntry {
  return {
    id: event.event_id,
    ts: formatClock(event.ts),
    type: event.event_type,
    level: eventLevel(event),
    message: eventMessage(event),
    runId: event.run_id,
    nodeId: event.node_id,
    role: event.role,
    round: event.round,
    status: event.status,
    latencyMs: event.latency_ms,
    payloadBytes: event.payload_bytes,
    metrics: event.metrics,
    details: event.details,
  }
}

function resolveRunState(prev: RunState, event: MonitorTelemetryEvent) {
  switch (event.event_type) {
    case 'run_requested':
    case 'run_resumed':
      return 'running' as const
    case 'run_paused':
      return 'paused' as const
    case 'run_stopped':
      return 'stopped' as const
    case 'node_error':
      return event.node_id === 'server' ? ('error' as const) : prev
    default:
      return prev
  }
}

function resolveClientStatus(eventType: TelemetryEventType): NodeStatus | null {
  switch (eventType) {
    case 'client_train_started':
      return 'training'
    case 'client_update_uploaded':
      return 'uploading'
    case 'client_train_completed':
      return 'idle'
    case 'node_error':
      return 'error'
    default:
      return null
  }
}

function upsertRound(rounds: RoundPoint[], event: MonitorTelemetryEvent) {
  if (!event.round || event.round <= 0) {
    return rounds
  }
  const previous = rounds[rounds.length - 1] ?? {
    round: event.round,
    globalLoss: 0,
    globalAccuracy: 0,
    serverLatencyMs: 0,
    ts: formatClock(event.ts),
  }
  const next = [...rounds]
  const idx = next.findIndex((point) => point.round === event.round)
  const point = idx >= 0 ? { ...next[idx] } : { ...previous, round: event.round }
  const loss = numberFromMetrics(event.metrics, ['eval_loss', 'loss', 'train_loss'])
  const accuracy = numberFromMetrics(event.metrics, ['eval_accuracy', 'accuracy', 'train_accuracy', 'acc'])

  if (loss !== null) {
    point.globalLoss = Number(loss.toFixed(4))
  }
  if (accuracy !== null) {
    point.globalAccuracy = Number(accuracy.toFixed(4))
  }
  if (event.latency_ms !== null && event.latency_ms !== undefined) {
    point.serverLatencyMs = Math.round(event.latency_ms)
  }
  point.ts = formatClock(event.ts)

  if (idx >= 0) {
    next[idx] = point
  } else {
    next.push(point)
  }

  next.sort((a, b) => a.round - b.round)
  return next.slice(-MAX_ROUNDS)
}

function applyTelemetryEvent(prev: DashboardState, event: MonitorTelemetryEvent): DashboardState {
  const entry = toEventEntry(event)
  const deduped = prev.events.filter((item) => item.id !== entry.id)
  const events = [entry, ...deduped].slice(0, MAX_EVENTS)

  const clients = [...prev.clients]
  let activeClientId = prev.activeClientId

  if (event.role === 'client') {
    const existingIdx = clients.findIndex((client) => client.id === event.node_id)
    if (existingIdx < 0) {
      clients.push({
        id: event.node_id,
        label: clientLabel(event.node_id),
        status: 'idle',
        lastAccuracy: 0,
        lastLatencyMs: 0,
        lastLoss: 0,
        lastRound: 0,
        lastEventType: null,
      })
    }
    const targetIdx = clients.findIndex((client) => client.id === event.node_id)
    if (targetIdx >= 0) {
      const target = { ...clients[targetIdx] }
      const nextStatus = resolveClientStatus(event.event_type)
      if (nextStatus) {
        target.status = nextStatus
      }
      const accuracy = numberFromMetrics(event.metrics, ['eval_accuracy', 'accuracy', 'train_accuracy', 'acc'])
      if (accuracy !== null) {
        target.lastAccuracy = Number(accuracy.toFixed(4))
      }
      const loss = numberFromMetrics(event.metrics, ['eval_loss', 'loss', 'train_loss'])
      if (loss !== null) {
        target.lastLoss = Number(loss.toFixed(4))
      }
      if (event.latency_ms !== null && event.latency_ms !== undefined) {
        target.lastLatencyMs = Math.round(event.latency_ms)
      }
      if (event.round && event.round > 0) {
        target.lastRound = event.round
      }
      target.lastEventType = event.event_type
      clients[targetIdx] = target
      activeClientId = event.node_id
    }
  }

  if (event.event_type === 'aggregation_started' || event.event_type === 'round_completed') {
    for (let idx = 0; idx < clients.length; idx += 1) {
      clients[idx] = { ...clients[idx], status: 'idle' }
    }
  }

  const nextRunState = resolveRunState(prev.runState, event)
  if (nextRunState === 'paused' || nextRunState === 'stopped' || nextRunState === 'idle') {
    for (let idx = 0; idx < clients.length; idx += 1) {
      clients[idx] = { ...clients[idx], status: 'idle' }
    }
  }

  clients.sort((a, b) => a.id.localeCompare(b.id))

  return {
    ...prev,
    runState: nextRunState,
    currentRound: Math.max(prev.currentRound, event.round ?? 0),
    clients,
    rounds: upsertRound(prev.rounds, event),
    events,
    activeClientId,
    hydrated: true,
  }
}

function hydrateFromSnapshot(snapshot: {
  run: { state: RunState }
  recent_events: MonitorTelemetryEvent[]
}) {
  const orderedEvents = [...snapshot.recent_events].sort(
    (a, b) => new Date(a.ts).getTime() - new Date(b.ts).getTime(),
  )
  let state: DashboardState = {
    ...seedState(),
    runState: snapshot.run.state,
    hydrated: true,
  }

  for (const event of orderedEvents) {
    state = applyTelemetryEvent(state, event)
  }

  if (state.runState === 'paused' || state.runState === 'stopped' || state.runState === 'idle') {
    state = {
      ...state,
      clients: state.clients.map((client) => ({ ...client, status: 'idle' })),
    }
  }

  return state
}

function deriveServerStatus(runState: RunState, events: EventEntry[]): NodeStatus {
  const latestServer = events.find((event) => event.role === 'server')
  if (runState === 'error' || latestServer?.type === 'node_error') {
    return 'error'
  }
  if (runState === 'paused' || runState === 'stopped' || runState === 'idle') {
    return 'idle'
  }
  if (latestServer?.type === 'aggregation_started') {
    return 'aggregating'
  }
  return runState === 'running' ? 'training' : 'idle'
}

function coalesceNodeId(nodeId: string, role: EventEntry['role']) {
  if (nodeId === 'operator' && role === 'server') {
    return 'server'
  }
  return nodeId
}

function buildNodeDetails(runState: RunState, clients: ClientSnapshot[], events: EventEntry[]) {
  const details: Record<string, NodeDetail> = {}
  details.server = {
    id: 'server',
    label: 'Aggregation Server',
    role: 'server',
    status: deriveServerStatus(runState, events),
    lastEventType: null,
    lastSeenAt: null,
    lastRound: 0,
    eventCount: 0,
    avgLatencyMs: null,
    accuracy: null,
    loss: null,
    lastMessage: null,
  }

  for (const client of clients) {
    details[client.id] = {
      id: client.id,
      label: client.label,
      role: 'client',
      status: client.status,
      lastEventType: client.lastEventType,
      lastSeenAt: null,
      lastRound: client.lastRound,
      eventCount: 0,
      avgLatencyMs: client.lastLatencyMs > 0 ? client.lastLatencyMs : null,
      accuracy: client.lastAccuracy > 0 ? client.lastAccuracy : null,
      loss: client.lastLoss > 0 ? client.lastLoss : null,
      lastMessage: null,
    }
  }

  const latencySums: Record<string, { total: number; count: number }> = {}
  for (const event of events) {
    const nodeId = coalesceNodeId(event.nodeId, event.role)
    if (!details[nodeId]) {
      details[nodeId] = {
        id: nodeId,
        label: nodeId.replace(/_/g, ' '),
        role: event.role,
        status: event.type === 'node_error' ? 'error' : 'idle',
        lastEventType: null,
        lastSeenAt: null,
        lastRound: 0,
        eventCount: 0,
        avgLatencyMs: null,
        accuracy: null,
        loss: null,
        lastMessage: null,
      }
    }
    const detail = details[nodeId]
    detail.eventCount += 1
    if (detail.lastSeenAt === null) {
      detail.lastSeenAt = event.ts
      detail.lastEventType = event.type
      detail.lastMessage = event.message
      if (event.round !== null) {
        detail.lastRound = Math.max(detail.lastRound, event.round)
      }
      if (event.type === 'node_error') {
        detail.status = 'error'
      }
    }
    if (event.latencyMs !== null) {
      const current = latencySums[nodeId] ?? { total: 0, count: 0 }
      latencySums[nodeId] = {
        total: current.total + event.latencyMs,
        count: current.count + 1,
      }
    }
  }

  for (const [nodeId, accumulator] of Object.entries(latencySums)) {
    details[nodeId].avgLatencyMs = Math.round(accumulator.total / accumulator.count)
  }

  return details
}

function canRunAction(runState: RunState, action: RunAction) {
  return VALID_TRANSITIONS[runState].includes(action)
}

function parseWsMessage(raw: string): MonitorWsMessage | null {
  try {
    const parsed = JSON.parse(raw) as { type?: string; payload?: unknown }
    if (
      parsed.type === 'event' &&
      typeof parsed.payload === 'object' &&
      parsed.payload !== null
    ) {
      return parsed as MonitorWsMessage
    }
    if (
      parsed.type === 'run_state' &&
      typeof parsed.payload === 'object' &&
      parsed.payload !== null
    ) {
      return parsed as MonitorWsMessage
    }
    return null
  } catch {
    return null
  }
}

export function useDashboardLive() {
  const [state, setState] = useState<DashboardState>(seedState)
  const [connectionState, setConnectionState] = useState<ConnectionState>('connecting')
  const [connectionError, setConnectionError] = useState<string | null>(null)
  const [controlPendingAction, setControlPendingAction] = useState<RunAction | null>(null)
  const [controlError, setControlError] = useState<string | null>(null)

  const monitorBaseUrl = useMemo(() => resolveMonitorBaseUrl(), [])
  const monitorWsUrl = useMemo(() => resolveMonitorWsUrl(monitorBaseUrl), [monitorBaseUrl])
  const runId = useMemo(() => resolveMonitorRunId(), [])

  useEffect(() => {
    const controller = new AbortController()
    let ignore = false

    const hydrate = async () => {
      try {
        const snapshot = await fetchMonitorSnapshot({
          baseUrl: monitorBaseUrl,
          runId,
          limit: SNAPSHOT_LIMIT,
          signal: controller.signal,
        })
        if (ignore) {
          return
        }
        setState(hydrateFromSnapshot(snapshot))
      } catch (error) {
        if (ignore || controller.signal.aborted) {
          return
        }
        const message = error instanceof Error ? error.message : 'snapshot hydration failed'
        setConnectionError(message)
        setState((prev) => ({
          ...prev,
          hydrated: true,
        }))
      }
    }

    void hydrate()
    return () => {
      ignore = true
      controller.abort()
    }
  }, [monitorBaseUrl, runId])

  useEffect(() => {
    let active = true
    let socket: WebSocket | null = null
    let reconnectTimer: number | null = null
    let pingTimer: number | null = null

    const clearTimers = () => {
      if (reconnectTimer !== null) {
        window.clearTimeout(reconnectTimer)
      }
      if (pingTimer !== null) {
        window.clearInterval(pingTimer)
      }
      reconnectTimer = null
      pingTimer = null
    }

    const connect = () => {
      if (!active) {
        return
      }
      setConnectionState('connecting')
      socket = new WebSocket(monitorWsUrl)

      socket.onopen = () => {
        if (!socket || socket.readyState !== WebSocket.OPEN) {
          return
        }
        setConnectionState('connected')
        setConnectionError(null)
        socket.send('subscribe')
        pingTimer = window.setInterval(() => {
          if (socket && socket.readyState === WebSocket.OPEN) {
            socket.send('ping')
          }
        }, PING_INTERVAL_MS)
      }

      socket.onmessage = (message) => {
        const parsed = parseWsMessage(message.data)
        if (!parsed) {
          return
        }
        if (parsed.type === 'event') {
          if (parsed.payload.run_id !== runId) {
            return
          }
          setState((prev) => applyTelemetryEvent(prev, parsed.payload))
          return
        }
        if (parsed.type === 'run_state') {
          if (parsed.payload.run_id !== runId) {
            return
          }
          setState((prev) => ({
            ...prev,
            runState: parsed.payload.state,
          }))
        }
      }

      socket.onerror = () => {
        setConnectionState('error')
      }

      socket.onclose = () => {
        clearTimers()
        if (!active) {
          return
        }
        setConnectionState('disconnected')
        reconnectTimer = window.setTimeout(() => {
          connect()
        }, RECONNECT_DELAY_MS)
      }
    }

    connect()

    return () => {
      active = false
      clearTimers()
      if (socket && (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING)) {
        socket.close()
      }
    }
  }, [monitorWsUrl, runId])

  const requestControl = useCallback(
    async (action: RunAction) => {
      if (!canRunAction(state.runState, action) || controlPendingAction !== null) {
        return
      }
      setControlPendingAction(action)
      setControlError(null)
      try {
        const response = await postMonitorControl({
          baseUrl: monitorBaseUrl,
          runId,
          action,
          reason: 'dashboard-control',
        })
        setState((prev) => ({
          ...prev,
          runState: response.state,
        }))
      } catch (error) {
        setControlError(error instanceof Error ? error.message : 'control request failed')
      } finally {
        setControlPendingAction(null)
      }
    },
    [controlPendingAction, monitorBaseUrl, runId, state.runState],
  )

  const nodeDetails = useMemo(
    () => buildNodeDetails(state.runState, state.clients, state.events),
    [state.runState, state.clients, state.events],
  )

  const actionAvailability = useMemo(
    () => ({
      start: canRunAction(state.runState, 'start') && controlPendingAction === null,
      pause: canRunAction(state.runState, 'pause') && controlPendingAction === null,
      resume: canRunAction(state.runState, 'resume') && controlPendingAction === null,
      stop: canRunAction(state.runState, 'stop') && controlPendingAction === null,
    }),
    [controlPendingAction, state.runState],
  )

  return {
    runId,
    runState: state.runState,
    currentRound: state.currentRound,
    clients: state.clients,
    rounds: state.rounds,
    events: state.events,
    activeClientId: state.activeClientId,
    hydrated: state.hydrated,
    connectionState,
    connectionError,
    controlPendingAction,
    controlError,
    controls: {
      start: () => requestControl('start'),
      pause: () => requestControl('pause'),
      resume: () => requestControl('resume'),
      stop: () => requestControl('stop'),
    },
    actionsEnabled: actionAvailability,
    nodeDetails,
  }
}
