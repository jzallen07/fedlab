import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import type { ClientSnapshot, RoundPoint } from '@/types/dashboard'

interface MetricsPanelProps {
  rounds: RoundPoint[]
  clients: ClientSnapshot[]
  currentRound: number
}

export function MetricsPanel({ rounds, clients, currentRound }: MetricsPanelProps) {
  const latest =
    rounds[rounds.length - 1] ??
    ({
      round: currentRound,
      globalLoss: 0,
      globalAccuracy: 0,
      serverLatencyMs: 0,
      ts: '--:--:--',
    } satisfies RoundPoint)

  return (
    <div className="grid min-w-0 gap-4 xl:grid-cols-2">
      <Card className="min-w-0 border-teal-900/10">
        <CardHeader>
          <CardTitle>Round KPI Trend</CardTitle>
          <CardDescription>Loss and accuracy progression through rounds.</CardDescription>
        </CardHeader>
        <CardContent className="h-64 min-w-0">
          <ResponsiveContainer width="100%" height="100%" minWidth={0}>
            <LineChart data={rounds} margin={{ left: 4, right: 12 }}>
              <CartesianGrid strokeDasharray="4 4" stroke="#cbd5e1" />
              <XAxis dataKey="round" tickLine={false} axisLine={false} />
              <YAxis yAxisId="loss" orientation="left" tickLine={false} axisLine={false} />
              <YAxis yAxisId="acc" orientation="right" domain={[0.4, 1]} tickLine={false} axisLine={false} />
              <Tooltip />
              <Legend />
              <Line
                yAxisId="loss"
                type="monotone"
                dataKey="globalLoss"
                stroke="#0f766e"
                strokeWidth={2.5}
                dot={false}
                name="Global Loss"
              />
              <Line
                yAxisId="acc"
                type="monotone"
                dataKey="globalAccuracy"
                stroke="#ea580c"
                strokeWidth={2.5}
                dot={false}
                name="Global Accuracy"
              />
            </LineChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      <Card className="min-w-0 border-teal-900/10">
        <CardHeader>
          <CardTitle>Client Latency and Accuracy</CardTitle>
          <CardDescription>Live per-site indicators for round {currentRound}.</CardDescription>
        </CardHeader>
        <CardContent className="h-64 min-w-0">
          <ResponsiveContainer width="100%" height="100%" minWidth={0}>
            <BarChart data={clients} margin={{ left: 2, right: 12 }}>
              <CartesianGrid strokeDasharray="4 4" stroke="#cbd5e1" />
              <XAxis dataKey="label" tickLine={false} axisLine={false} tick={{ fontSize: 11 }} />
              <YAxis yAxisId="latency" orientation="left" tickLine={false} axisLine={false} />
              <YAxis yAxisId="accuracy" orientation="right" domain={[0.5, 1]} tickLine={false} axisLine={false} />
              <Tooltip />
              <Legend />
              <Bar yAxisId="latency" dataKey="lastLatencyMs" fill="#0f766e" name="Latency (ms)" radius={4} />
              <Bar
                yAxisId="accuracy"
                dataKey="lastAccuracy"
                fill="#ea580c"
                name="Accuracy"
                radius={4}
              />
            </BarChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      <Card className="xl:col-span-2 border-teal-900/10">
        <CardHeader>
          <CardTitle>Current Round Snapshot</CardTitle>
          <CardDescription>Operational summary of the latest aggregation cycle.</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-3 sm:grid-cols-3">
            <div className="rounded-lg border border-border bg-muted/50 p-3">
              <p className="font-mono text-[11px] uppercase text-muted-foreground">Global Accuracy</p>
              <p className="mt-1 text-xl font-semibold">{(latest.globalAccuracy * 100).toFixed(1)}%</p>
            </div>
            <div className="rounded-lg border border-border bg-muted/50 p-3">
              <p className="font-mono text-[11px] uppercase text-muted-foreground">Global Loss</p>
              <p className="mt-1 text-xl font-semibold">{latest.globalLoss.toFixed(3)}</p>
            </div>
            <div className="rounded-lg border border-border bg-muted/50 p-3">
              <p className="font-mono text-[11px] uppercase text-muted-foreground">Server Latency</p>
              <p className="mt-1 text-xl font-semibold">{latest.serverLatencyMs}ms</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
