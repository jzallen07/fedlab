import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import type { EventEntry } from '@/types/dashboard'

interface EventStreamPanelProps {
  events: EventEntry[]
  selectedNodeId: string | null
  onSelectNode: (nodeId: string) => void
}

function coalesceNodeId(nodeId: string, role: EventEntry['role']) {
  if (nodeId === 'operator' && role === 'server') {
    return 'server'
  }
  return nodeId
}

function levelBadge(level: EventEntry['level']) {
  if (level === 'error') {
    return 'danger' as const
  }
  if (level === 'warn') {
    return 'warn' as const
  }
  return 'success' as const
}

export function EventStreamPanel({ events, selectedNodeId, onSelectNode }: EventStreamPanelProps) {
  return (
    <Card className="border-teal-900/10">
      <CardHeader>
        <CardTitle>Event Log</CardTitle>
        <CardDescription>
          Live event table with node drill-down. Click a row to open node inspection.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="max-h-[420px] overflow-auto rounded-lg border border-border/80">
          <table className="min-w-full table-auto border-collapse text-xs">
            <thead className="sticky top-0 z-10 bg-muted/90 font-mono uppercase tracking-wide text-muted-foreground">
              <tr>
                <th className="px-3 py-2 text-left">Time</th>
                <th className="px-3 py-2 text-left">Event</th>
                <th className="px-3 py-2 text-left">Node</th>
                <th className="px-3 py-2 text-left">Round</th>
                <th className="px-3 py-2 text-left">Status</th>
              </tr>
            </thead>
            <tbody>
              {events.map((event) => {
                const nodeId = coalesceNodeId(event.nodeId, event.role)
                const isSelected = selectedNodeId === nodeId
                return (
                  <tr
                    className={[
                      'cursor-pointer border-t border-border/70 bg-white/75 hover:bg-teal-50/70',
                      isSelected ? 'bg-teal-100/60' : '',
                    ].join(' ')}
                    key={event.id}
                    onClick={() => onSelectNode(nodeId)}
                  >
                    <td className="whitespace-nowrap px-3 py-2 font-mono text-[11px] text-muted-foreground">{event.ts}</td>
                    <td className="px-3 py-2">
                      <Badge variant={levelBadge(event.level)}>{event.type}</Badge>
                    </td>
                    <td className="px-3 py-2 font-mono text-[11px]">{nodeId}</td>
                    <td className="px-3 py-2">{event.round ?? '-'}</td>
                    <td className="px-3 py-2">{event.status}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
          {events.length === 0 ? (
            <div className="p-4 text-sm text-muted-foreground">No telemetry received yet.</div>
          ) : null}
        </div>
      </CardContent>
    </Card>
  )
}
