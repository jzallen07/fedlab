import {
  Background,
  Controls,
  Handle,
  MarkerType,
  MiniMap,
  Position,
  ReactFlow,
  type Edge,
  type Node,
  type NodeProps,
} from 'reactflow'

import { Badge } from '@/components/ui/badge'
import { cn } from '@/lib/utils'
import type { ClientSnapshot, NodeStatus, RunState, TelemetryEventType } from '@/types/dashboard'

interface TopologyGraphProps {
  clients: ClientSnapshot[]
  activeClientId: string
  currentRound: number
  runState: RunState
  latestEventType: TelemetryEventType
  selectedNodeId: string | null
  onSelectNode: (nodeId: string) => void
}

interface TopologyNodeData {
  label: string
  role: 'server' | 'client'
  status: NodeStatus
  accent: string
  subtitle: string
  selected: boolean
}

function statusColor(status: NodeStatus) {
  switch (status) {
    case 'training':
      return 'border-teal-600 bg-teal-50 text-teal-900'
    case 'uploading':
      return 'border-amber-700 bg-amber-50 text-amber-900'
    case 'aggregating':
      return 'border-cyan-700 bg-cyan-50 text-cyan-900'
    case 'error':
      return 'border-red-700 bg-red-50 text-red-900'
    default:
      return 'border-slate-300 bg-slate-50 text-slate-900'
  }
}

function StatusNode({ data }: NodeProps<TopologyNodeData>) {
  return (
    <div
      className={cn(
        'w-44 rounded-lg border px-3 py-2 shadow-sm transition-all',
        statusColor(data.status),
        data.selected ? 'ring-2 ring-primary/70 ring-offset-2 ring-offset-background' : '',
      )}
    >
      {data.role === 'server' ? (
        <>
          <Handle id="to-clients" position={Position.Bottom} type="source" />
          <Handle id="from-clients" position={Position.Bottom} type="target" />
        </>
      ) : (
        <>
          <Handle id="from-server" position={Position.Top} type="target" />
          <Handle id="to-server" position={Position.Top} type="source" />
        </>
      )}
      <div className="flex items-center justify-between">
        <span className="font-mono text-[10px] uppercase tracking-wide opacity-80">{data.role}</span>
        <Badge
          variant={
            data.status === 'training'
              ? 'success'
              : data.status === 'uploading' || data.status === 'aggregating'
                ? 'warn'
                : data.status === 'error'
                  ? 'danger'
                  : 'muted'
          }
        >
          {data.status}
        </Badge>
      </div>
      <p className="mt-2 text-sm font-semibold leading-tight">{data.label}</p>
      <p className="mt-1 text-xs opacity-80">{data.subtitle}</p>
    </div>
  )
}

const NODE_TYPES = {
  statusNode: StatusNode,
}

function deriveServerStatus(runState: RunState, latestEventType: string): NodeStatus {
  if (latestEventType === 'node_error' || runState === 'error') {
    return 'error'
  }
  if (runState === 'stopped') {
    return 'idle'
  }
  if (latestEventType === 'aggregation_started') {
    return 'aggregating'
  }
  if (latestEventType === 'run_paused') {
    return 'idle'
  }
  return runState === 'running' ? 'training' : 'idle'
}

export function TopologyGraph({
  clients,
  activeClientId,
  currentRound,
  runState,
  latestEventType,
  selectedNodeId,
  onSelectNode,
}: TopologyGraphProps) {
  const serverStatus = deriveServerStatus(runState, latestEventType)

  const nodes: Node<TopologyNodeData>[] = [
    {
      id: 'server',
      type: 'statusNode',
      position: { x: 310, y: 40 },
      sourcePosition: Position.Bottom,
      data: {
        label: 'Aggregation Server',
        role: 'server',
        status: serverStatus,
        accent: 'server',
        subtitle: `Round ${currentRound} · ${runState}`,
        selected: selectedNodeId === 'server',
      },
      draggable: false,
      selectable: false,
    },
    ...clients.map((client, idx) => ({
      id: client.id,
      type: 'statusNode',
      position: { x: idx * 260 + 70, y: 230 },
      targetPosition: Position.Top,
      sourcePosition: Position.Top,
      data: {
        label: client.label,
        role: 'client' as const,
        status: client.status,
        accent: client.id,
        subtitle: `acc ${(client.lastAccuracy * 100).toFixed(1)}% · ${client.lastLatencyMs}ms`,
        selected: selectedNodeId === client.id,
      },
      draggable: false,
      selectable: false,
    })),
  ]

  const edges: Edge[] = clients.flatMap((client) => {
    const isActive = client.id === activeClientId
    const animated = client.status === 'uploading' || client.status === 'training' || isActive

    return [
      {
        id: `${client.id}-down`,
        source: 'server',
        target: client.id,
        sourceHandle: 'to-clients',
        targetHandle: 'from-server',
        type: 'smoothstep',
        animated,
        markerEnd: { type: MarkerType.ArrowClosed, width: 18, height: 18 },
        style: { stroke: animated ? '#0f766e' : '#94a3b8', strokeWidth: animated ? 2.3 : 1.6 },
      },
      {
        id: `${client.id}-up`,
        source: client.id,
        target: 'server',
        sourceHandle: 'to-server',
        targetHandle: 'from-clients',
        type: 'smoothstep',
        animated: client.status === 'uploading',
        markerEnd: { type: MarkerType.ArrowClosed, width: 18, height: 18 },
        style: {
          stroke: client.status === 'uploading' ? '#c2410c' : '#cbd5e1',
          strokeWidth: client.status === 'uploading' ? 2.2 : 1.2,
        },
      },
    ]
  })

  return (
    <div className="h-[340px] w-full rounded-lg border border-border/70 bg-white/80">
      <ReactFlow
        fitView
        minZoom={0.25}
        maxZoom={1.4}
        nodes={nodes}
        edges={edges}
        nodeTypes={NODE_TYPES}
        nodesConnectable={false}
        nodesDraggable={false}
        elementsSelectable={false}
        panOnScroll
        onNodeClick={(_, node) => {
          onSelectNode(node.id)
        }}
      >
        <Background color="#d8dee8" gap={24} size={1} />
        <MiniMap pannable zoomable />
        <Controls showInteractive={false} />
      </ReactFlow>
    </div>
  )
}
