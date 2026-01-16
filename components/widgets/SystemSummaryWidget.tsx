import React, { useState, useEffect } from 'react';
import { CheckCircle2, AlertOctagon, XCircle, Server } from 'lucide-react';
import WidgetCard from '../WidgetCard';
import { SystemStatus } from '../../types';
import { backendApi } from '../../utils/api';
import { useRealtimeScores } from '../../utils/useRealtimeScores';

interface SystemSummaryWidgetProps {
  isRunning: boolean;
}

const SystemSummaryWidget: React.FC<SystemSummaryWidgetProps> = ({ isRunning }) => {
  const [status, setStatus] = useState<SystemStatus>('normal');
  const [uptime, setUptime] = useState(99.8);
  const [lastCheck, setLastCheck] = useState<string>('Just now');
  const [modelLoaded, setModelLoaded] = useState<boolean>(false);
  const [latestScore, setLatestScore] = useState<number | null>(null);
  const [latestLevel, setLatestLevel] = useState<SystemStatus>('normal');
  const realtimeData = useRealtimeScores(isRunning);

  useEffect(() => {
    if (!isRunning) {
      setStatus('normal');
      setLastCheck('Paused');
      setLatestScore(null);
      setLatestLevel('normal');
      return;
    }

    const updateHealth = async () => {
      try {
        const health = await backendApi.getHealth();
        setModelLoaded(health.model_loaded);
        
        // uptime은 기존대로 유지 (시스템 메모리 기반)
        if (health.system?.memory_percent !== undefined) {
          setUptime(Math.max(0, Math.min(100, 100 - health.system.memory_percent)));
        }
        
        // 마지막 체크 시간 업데이트
        const checkTime = new Date(health.timestamp);
        const now = new Date();
        const diffSeconds = Math.floor((now.getTime() - checkTime.getTime()) / 1000);
        if (diffSeconds < 5) {
          setLastCheck('Just now');
        } else if (diffSeconds < 60) {
          setLastCheck(`${diffSeconds}초 전`);
        } else {
          const diffMinutes = Math.floor(diffSeconds / 60);
          setLastCheck(`${diffMinutes}분 전`);
        }
      } catch (err: any) {
        console.error('Health 체크 실패:', err);
        // Show more detailed error message
        const errorMsg = err?.message || '연결 실패';
        if (errorMsg.includes('timeout')) {
          setLastCheck('타임아웃');
        } else if (errorMsg.includes('NetworkError') || errorMsg.includes('Failed to fetch')) {
          setLastCheck('네트워크 오류');
        } else {
          setLastCheck('연결 실패');
        }
      }
    };

    // 초기 로드
    updateHealth();

    // 4초마다 업데이트
    const interval = setInterval(updateHealth, 4000);

    return () => clearInterval(interval);
  }, [isRunning]);

  // MDRAD의 warning/critical 기준(기본값): 75% / 95% 분위수 기준
  const THRESH = { warning: 0.1915, critical: 0.8151 } as const;

  const levelOf = (score: number | null, warning: number, critical: number): SystemStatus => {
    if (score === null || Number.isNaN(score)) return 'normal';
    if (score >= critical) return 'critical';
    if (score >= warning) return 'warning';
    return 'normal';
  };

  // Update status when new realtime data arrives
  useEffect(() => {
    if (!isRunning || !realtimeData) return;

    if (realtimeData.status !== 'success') {
      setStatus('warning');
      return;
    }

    const score = realtimeData.scores?.mdrad ?? null;
    const level = levelOf(score, THRESH.warning, THRESH.critical);
    setLatestScore(score);
    setLatestLevel(level);
    setStatus(level);
  }, [realtimeData, isRunning]);

  const getStatusConfig = (s: SystemStatus) => {
    switch (s) {
      case 'normal':
        return {
          color: 'text-emerald-500',
          bg: 'bg-emerald-50 dark:bg-emerald-900/20',
          border: 'border-emerald-200 dark:border-emerald-800',
          label: '정상',
          icon: <CheckCircle2 size={32} className="text-emerald-500" />
        };
      case 'warning':
        return {
          color: 'text-amber-500',
          bg: 'bg-amber-50 dark:bg-amber-900/20',
          border: 'border-amber-200 dark:border-amber-800',
          label: '주의',
          icon: <AlertOctagon size={32} className="text-amber-500" />
        };
      case 'critical':
        return {
          color: 'text-red-500',
          bg: 'bg-red-50 dark:bg-red-900/20',
          border: 'border-red-200 dark:border-red-800',
          label: '경고',
          icon: <XCircle size={32} className="text-red-500" />
        };
    }
  };

  const config = getStatusConfig(status);

  const badgeClass = (s: SystemStatus) => {
    switch (s) {
      case 'critical':
        return 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-300';
      case 'warning':
        return 'bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-300';
      default:
        return 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-300';
    }
  };

  return (
    <WidgetCard title="Summary.ai" icon={<Server size={20} />} className="h-full min-h-[190px]">
      <div className={`h-full flex flex-col justify-between ${config.bg} ${config.border} border rounded-lg p-4 transition-colors duration-500`}>
        <div className="flex items-center gap-4">
          <div className="p-2 bg-white dark:bg-slate-900 rounded-full shadow-sm">
            {config.icon}
          </div>
          <div>
            <h4 className={`font-bold text-lg ${config.color} transition-colors duration-500`}>
              {config.label}
            </h4>
            <p className="text-xs text-slate-500 dark:text-slate-400">
              Last check: {lastCheck}
            </p>
            {modelLoaded && (
              <p className="text-xs text-emerald-600 dark:text-emerald-400 mt-1">
                ✓ Model Loaded
              </p>
            )}
          </div>
        </div>

        {/* 모델 상태 요약 (차트 색과 일치) */}
        <div className="mt-3 grid grid-cols-1 gap-2">
          <div className="flex items-center justify-between rounded-md bg-white/60 dark:bg-slate-950/20 border border-slate-200/60 dark:border-slate-800/60 px-3 py-2">
            <div className="flex items-center gap-2 min-w-0">
              <span className="inline-block h-2.5 w-2.5 rounded-full bg-emerald-500" />
              <span className="text-xs font-semibold text-slate-700 dark:text-slate-200">MDRAD</span>
              <span className={`ml-2 text-[11px] font-semibold px-2 py-0.5 rounded-full ${badgeClass(latestLevel)}`}>
                {latestLevel === 'critical' ? '경고' : latestLevel === 'warning' ? '주의' : '정상'}
              </span>
            </div>
            <span className="text-[11px] font-mono tabular-nums text-slate-600 dark:text-slate-300">
              {latestScore === null ? '-' : latestScore.toFixed(4)}
            </span>
          </div>
        </div>
      </div>
    </WidgetCard>
  );
};

export default SystemSummaryWidget;