import { useMemo, useState } from 'react'
import { Cpu, GaugeCircle, Layers3 } from 'lucide-react'

import { EventStreamPanel } from '@/components/event-stream-panel'
import { MetricsPanel } from '@/components/metrics-panel'
import { NodeDetailDrawer } from '@/components/node-detail-drawer'
import { RunSummaryPanel } from '@/components/run-summary-panel'
import { TopologyGraph } from '@/components/topology-graph'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { useDashboardLive } from '@/hooks/useDashboardLive'

function App() {
  const {
    runId,
    runState,
    currentRound,
    clients,
    rounds,
    events,
    activeClientId,
    hydrated,
    connectionState,
    connectionError,
    controlPendingAction,
    controlError,
    controls,
    actionsEnabled,
    nodeDetails,
  } = useDashboardLive()
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>('server')
  const [drawerOpen, setDrawerOpen] = useState(false)
  const effectiveSelectedNodeId =
    selectedNodeId && nodeDetails[selectedNodeId] ? selectedNodeId : 'server'

  const selectedNode = useMemo(
    () => nodeDetails[effectiveSelectedNodeId] ?? null,
    [effectiveSelectedNodeId, nodeDetails],
  )

  const latestEventType = events[0]?.type ?? 'node_heartbeat'

  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="pointer-events-none fixed inset-0 -z-10 overflow-hidden">
        <div className="absolute -left-24 top-10 h-80 w-80 rounded-full bg-cyan-300/25 blur-3xl" />
        <div className="absolute right-0 top-36 h-96 w-96 rounded-full bg-orange-300/25 blur-3xl" />
        <div className="absolute bottom-0 left-1/3 h-72 w-72 rounded-full bg-teal-200/25 blur-3xl" />
      </div>

      <main className="mx-auto max-w-[1400px] space-y-4 p-4 sm:p-6 lg:p-8">
        <header className="rounded-xl border border-border/70 bg-card/90 p-5 shadow-[0_10px_40px_rgba(12,36,44,0.08)]">
          <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
            <div className="space-y-1">
              <p className="font-mono text-xs uppercase tracking-[0.18em] text-primary">FedForge Dashboard</p>
              <h1 className="text-2xl font-bold leading-tight sm:text-3xl">Realtime Federated Training Map</h1>
              <p className="text-sm text-muted-foreground">
                Live telemetry from monitor API with topology, KPIs, controls, and debugging timeline.
              </p>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <Badge variant="muted">Run {runId}</Badge>
              <Badge variant={connectionState === 'connected' ? 'success' : connectionState === 'error' ? 'danger' : 'warn'}>
                ws:{connectionState}
              </Badge>
              {!hydrated ? <Badge variant="warn">hydrating</Badge> : null}
              <Badge
                variant={
                  runState === 'running'
                    ? 'success'
                    : runState === 'paused'
                      ? 'warn'
                      : runState === 'error'
                        ? 'danger'
                        : 'muted'
                }
              >
                {runState}
              </Badge>
            </div>
          </div>
        </header>

        <section className="grid gap-4 xl:grid-cols-12">
          <Card className="xl:col-span-8 border-teal-900/10">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Layers3 className="h-4 w-4 text-primary" />
                Topology Flow
              </CardTitle>
              <CardDescription>Animated client/server state transitions by round.</CardDescription>
            </CardHeader>
            <CardContent>
              <TopologyGraph
                activeClientId={activeClientId}
                clients={clients}
                currentRound={currentRound}
                latestEventType={latestEventType}
                onSelectNode={(nodeId) => {
                  setSelectedNodeId(nodeId)
                  setDrawerOpen(true)
                }}
                runState={runState}
                selectedNodeId={effectiveSelectedNodeId}
              />
            </CardContent>
          </Card>

          <div className="space-y-4 xl:col-span-4">
            <RunSummaryPanel
              actionError={controlError}
              actionsEnabled={actionsEnabled}
              clients={clients}
              connectionError={connectionError}
              connectionState={connectionState}
              currentRound={currentRound}
              pendingAction={controlPendingAction}
              onPause={controls.pause}
              onResume={controls.resume}
              onStart={controls.start}
              onStop={controls.stop}
              runId={runId}
              runState={runState}
            />

            <Card className="border-teal-900/10">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-sm">
                  <GaugeCircle className="h-4 w-4 text-primary" />
                  Deployment Profile
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-2 text-sm text-muted-foreground">
                <div className="flex items-center justify-between">
                  <span className="flex items-center gap-2">
                    <Cpu className="h-4 w-4" /> Device
                  </span>
                  <span className="font-mono text-foreground">cpu-safe</span>
                </div>
                <div className="flex items-center justify-between">
                  <span>Train Mode</span>
                  <span className="font-mono text-foreground">head_only</span>
                </div>
                <div className="flex items-center justify-between">
                  <span>Clients</span>
                  <span className="font-mono text-foreground">{clients.length}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span>Current Round</span>
                  <span className="font-mono text-foreground">{currentRound}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span>Hydrated Events</span>
                  <span className="font-mono text-foreground">{events.length}</span>
                </div>
              </CardContent>
            </Card>
          </div>

          <div className="xl:col-span-7">
            <MetricsPanel clients={clients} currentRound={currentRound} rounds={rounds} />
          </div>

          <div className="xl:col-span-5">
            <EventStreamPanel
              events={events}
              onSelectNode={(nodeId) => {
                setSelectedNodeId(nodeId)
                setDrawerOpen(true)
              }}
              selectedNodeId={effectiveSelectedNodeId}
            />
          </div>
        </section>
      </main>

      <NodeDetailDrawer
        events={events}
        node={selectedNode}
        onClose={() => setDrawerOpen(false)}
        open={drawerOpen}
      />
    </div>
  )
}

export default App
