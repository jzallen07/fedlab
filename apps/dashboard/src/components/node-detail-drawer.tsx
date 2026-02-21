import { X } from 'lucide-react'
import { useMemo } from 'react'

import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import type { EventEntry, NodeDetail } from '@/types/dashboard'

interface NodeDetailDrawerProps {
  node: NodeDetail | null
  events: EventEntry[]
  open: boolean
  onClose: () => void
}

function statusVariant(status: NodeDetail['status']) {
  if (status === 'error') {
    return 'danger' as const
  }
  if (status === 'training' || status === 'uploading' || status === 'aggregating') {
    return 'warn' as const
  }
  return 'muted' as const
}

function coalesceNodeId(nodeId: string, role: EventEntry['role']) {
  if (nodeId === 'operator' && role === 'server') {
    return 'server'
  }
  return nodeId
}

export function NodeDetailDrawer({ node, events, open, onClose }: NodeDetailDrawerProps) {
  const nodeEvents = useMemo(() => {
    if (!node) {
      return []
    }
    return events.filter((event) => coalesceNodeId(event.nodeId, event.role) === node.id).slice(0, 12)
  }, [events, node])

  if (!node) {
    return null
  }

  return (
    <div
      className={[
        'fixed inset-y-0 right-0 z-40 w-full max-w-md transform border-l border-border bg-background/95 backdrop-blur transition-transform duration-300 sm:w-[420px]',
        open ? 'translate-x-0' : 'translate-x-full',
      ].join(' ')}
    >
      <div className="flex h-full flex-col">
        <div className="flex items-center justify-between border-b border-border px-4 py-3">
          <div>
            <p className="font-mono text-[11px] uppercase tracking-wide text-muted-foreground">Node Detail</p>
            <h2 className="text-base font-semibold">{node.label}</h2>
          </div>
          <Button onClick={onClose} size="icon" variant="ghost">
            <X className="h-4 w-4" />
          </Button>
        </div>

        <div className="space-y-4 overflow-y-auto p-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                <span>Runtime State</span>
                <Badge variant={statusVariant(node.status)}>{node.status}</Badge>
              </CardTitle>
              <CardDescription>Most recent telemetry for this node.</CardDescription>
            </CardHeader>
            <CardContent className="grid grid-cols-2 gap-2 text-sm">
              <div className="rounded-md border border-border bg-muted/40 p-2">
                <p className="font-mono text-[10px] uppercase text-muted-foreground">Role</p>
                <p className="mt-1 font-semibold">{node.role}</p>
              </div>
              <div className="rounded-md border border-border bg-muted/40 p-2">
                <p className="font-mono text-[10px] uppercase text-muted-foreground">Last Round</p>
                <p className="mt-1 font-semibold">{node.lastRound}</p>
              </div>
              <div className="rounded-md border border-border bg-muted/40 p-2">
                <p className="font-mono text-[10px] uppercase text-muted-foreground">Events</p>
                <p className="mt-1 font-semibold">{node.eventCount}</p>
              </div>
              <div className="rounded-md border border-border bg-muted/40 p-2">
                <p className="font-mono text-[10px] uppercase text-muted-foreground">Avg Latency</p>
                <p className="mt-1 font-semibold">
                  {node.avgLatencyMs !== null ? `${Math.round(node.avgLatencyMs)}ms` : 'n/a'}
                </p>
              </div>
              <div className="rounded-md border border-border bg-muted/40 p-2">
                <p className="font-mono text-[10px] uppercase text-muted-foreground">Accuracy</p>
                <p className="mt-1 font-semibold">
                  {node.accuracy !== null ? `${(node.accuracy * 100).toFixed(1)}%` : 'n/a'}
                </p>
              </div>
              <div className="rounded-md border border-border bg-muted/40 p-2">
                <p className="font-mono text-[10px] uppercase text-muted-foreground">Loss</p>
                <p className="mt-1 font-semibold">{node.loss !== null ? node.loss.toFixed(4) : 'n/a'}</p>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Recent Node Events</CardTitle>
              <CardDescription>Last 12 events for focused debugging.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-2">
              {nodeEvents.length === 0 ? (
                <p className="rounded-md border border-border bg-muted/30 p-3 text-sm text-muted-foreground">
                  No events yet for this node.
                </p>
              ) : (
                nodeEvents.map((event) => (
                  <div className="rounded-md border border-border bg-white/80 p-3" key={event.id}>
                    <div className="flex items-center justify-between gap-2">
                      <span className="font-mono text-[11px] uppercase text-muted-foreground">{event.ts}</span>
                      <Badge variant={event.level === 'error' ? 'danger' : event.level === 'warn' ? 'warn' : 'success'}>
                        {event.type}
                      </Badge>
                    </div>
                    <p className="mt-2 text-sm">{event.message}</p>
                  </div>
                ))
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}
