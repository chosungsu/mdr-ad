export type SystemStatus = 'normal' | 'warning' | 'critical';

export interface LogEntry {
  id: string;
  timestamp: number;
  message: string;
  level: 'info' | 'warning' | 'error' | 'success';
}

export interface MetricPoint {
  time: string;
  value: number;
  baseline: number;
}

export interface DashboardContextProps {
  isRunning: boolean;
}