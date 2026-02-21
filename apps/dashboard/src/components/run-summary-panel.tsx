import { Activity, Network, PauseCircle, PlayCircle } from 'lucide-react'

import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import type { ClientSnapshot, ConnectionState, RunAction, RunState } from '@/types/dashboard'

interface RunSummaryPanelProps {
  runState: RunState
  runId: string
  currentRound: number
  clients: ClientSnapshot[]
  connectionState: ConnectionState
  connectionError: string | null
  pendingAction: RunAction | null
  actionError: string | null
  actionsEnabled: Record<RunAction, boolean>
  onStart: () => void
  onPause: () => void
  onResume: () => void
  onStop: () => void
}

function stateBadgeVariant(state: RunState) {
  if (state === 'running') {
    return 'success' as const
  }
  if (state === 'paused') {
    return 'warn' as const
  }
  if (state === 'error') {
    return 'danger' as const
  }
  return 'muted' as const
}

function connectionVariant(state: ConnectionState) {
  if (state === 'connected') {
    return 'success' as const
  }
  if (state === 'error') {
    return 'danger' as const
  }
  if (state === 'disconnected') {
    return 'warn' as const
  }
  return 'muted' as const
}

export function RunSummaryPanel({
  runState,
  runId,
  currentRound,
  clients,
  connectionState,
  connectionError,
  pendingAction,
  actionError,
  actionsEnabled,
  onStart,
  onPause,
  onResume,
  onStop,
}: RunSummaryPanelProps) {
  const trainingClients = clients.filter((client) => client.status === 'training').length
  const uploadingClients = clients.filter((client) => client.status === 'uploading').length

  return (
    <Card className="border-teal-900/10">
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span>Run Control</span>
          <Badge variant={stateBadgeVariant(runState)}>{runState}</Badge>
        </CardTitle>
        <CardDescription>Control commands target monitor API run lifecycle endpoints.</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex flex-wrap items-center gap-2">
          <Badge variant="muted">Run {runId}</Badge>
          <Badge variant={connectionVariant(connectionState)}>ws:{connectionState}</Badge>
          {pendingAction ? <Badge variant="warn">pending:{pendingAction}</Badge> : null}
        </div>

        {connectionError ? (
          <p className="rounded-lg border border-red-700/20 bg-red-50/80 p-2 text-xs text-red-700">
            Connection error: {connectionError}
          </p>
        ) : null}
        {actionError ? (
          <p className="rounded-lg border border-red-700/20 bg-red-50/80 p-2 text-xs text-red-700">
            Control action failed: {actionError}
          </p>
        ) : null}

        <div className="grid gap-2 text-sm sm:grid-cols-2">
          <div className="rounded-lg border border-border bg-muted/40 p-3">
            <p className="font-mono text-[11px] uppercase text-muted-foreground">Round</p>
            <p className="mt-1 text-lg font-semibold">{currentRound}</p>
          </div>
          <div className="rounded-lg border border-border bg-muted/40 p-3">
            <p className="font-mono text-[11px] uppercase text-muted-foreground">Nodes Active</p>
            <p className="mt-1 text-lg font-semibold">{clients.length + 1}</p>
          </div>
          <div className="rounded-lg border border-border bg-muted/40 p-3">
            <p className="font-mono text-[11px] uppercase text-muted-foreground">Training</p>
            <p className="mt-1 text-lg font-semibold">{trainingClients}</p>
          </div>
          <div className="rounded-lg border border-border bg-muted/40 p-3">
            <p className="font-mono text-[11px] uppercase text-muted-foreground">Uploading</p>
            <p className="mt-1 text-lg font-semibold">{uploadingClients}</p>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-2">
          <Button className="gap-2" disabled={!actionsEnabled.start} onClick={onStart} variant="default">
            <PlayCircle className="h-4 w-4" /> Start
          </Button>
          <Button
            className="gap-2"
            disabled={!actionsEnabled.pause}
            onClick={onPause}
            variant="outline"
          >
            <PauseCircle className="h-4 w-4" /> Pause
          </Button>
          <Button
            className="gap-2"
            disabled={!actionsEnabled.resume}
            onClick={onResume}
            variant="secondary"
          >
            <Activity className="h-4 w-4" /> Resume
          </Button>
          <Button className="gap-2" disabled={!actionsEnabled.stop} onClick={onStop} variant="danger">
            <Network className="h-4 w-4" /> Stop
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}
