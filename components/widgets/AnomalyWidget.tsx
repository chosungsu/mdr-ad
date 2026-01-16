import React, { useEffect, useRef, useState } from 'react';
import { AlertTriangle } from 'lucide-react';
import WidgetCard from '../WidgetCard';
import { useDashboardData } from '../../utils/useDashboardData';

interface AnomalyWidgetProps {
  isRunning: boolean;
}

// MDRAD의 warning/critical 기준(기본값): 75% / 95% 분위수 기준
const THRESH = { warning: 0.1915, critical: 0.8151 } as const;

type Level = 'normal' | 'warning' | 'critical';

function levelOf(score: number | null, warning: number, critical: number): Level {
  if (score === null || Number.isNaN(score)) return 'normal';
  if (score >= critical) return 'critical';
  if (score >= warning) return 'warning';
  return 'normal';
}

const AnomalyWidget: React.FC<AnomalyWidgetProps> = ({ isRunning }) => {
  const [counts, setCounts] = useState({
    mdrad_warning: 0,
    mdrad_critical: 0,
  });
  const lastCursorRef = useRef<number>(0);
  const dashboardData = useDashboardData(isRunning);

  useEffect(() => {
    if (!isRunning) {
      setCounts({ mdrad_warning: 0, mdrad_critical: 0 });
      lastCursorRef.current = 0;
      return;
    }
  }, [isRunning]);

  // Update counts when new dashboard data arrives (every 2 seconds)
  useEffect(() => {
    if (!isRunning || !dashboardData || dashboardData.scoresStatus !== 'success') return;
    if (!dashboardData.scores) return;

    const cursor = Number(dashboardData.cursor || 0);
    
    // wrap으로 cursor가 1로 다시 돌아가면, 카운트도 새 사이클로 간주해 리셋
    if (cursor > 0 && lastCursorRef.current > 0 && cursor < lastCursorRef.current) {
      setCounts({ mdrad_warning: 0, mdrad_critical: 0 });
    }
    
    // 같은 cursor를 중복 처리하지 않음
    if (cursor <= lastCursorRef.current) return;
    lastCursorRef.current = cursor;

    const mdrLevel = levelOf(dashboardData.scores.mdrad, THRESH.warning, THRESH.critical);

    setCounts((prev) => ({
      mdrad_warning: prev.mdrad_warning + (mdrLevel === 'warning' ? 1 : 0),
      mdrad_critical: prev.mdrad_critical + (mdrLevel === 'critical' ? 1 : 0),
    }));
  }, [dashboardData, isRunning]);

  return (
    <WidgetCard title="Detected Anomalies" icon={<AlertTriangle size={20} />} className="h-full min-h-[190px]">
      <div className="h-full flex flex-col justify-between gap-4">
        <div className="grid grid-cols-1 gap-3">
          <div className="rounded-lg border border-slate-200/60 dark:border-slate-800/60 bg-white/60 dark:bg-slate-900/30 p-3">
            <div className="text-xs font-semibold text-slate-500 dark:text-slate-400">MDRAD</div>
            <div className="mt-2 flex items-baseline justify-between">
              <span className="text-xs text-amber-600 dark:text-amber-400">주의</span>
              <span className="text-2xl font-bold text-slate-900 dark:text-white tabular-nums">{counts.mdrad_warning}</span>
            </div>
            <div className="mt-1 flex items-baseline justify-between">
              <span className="text-xs text-red-600 dark:text-red-400">경고</span>
              <span className="text-2xl font-bold text-slate-900 dark:text-white tabular-nums">{counts.mdrad_critical}</span>
            </div>
          </div>
        </div>

        <div className="text-xs text-slate-500 dark:text-slate-400">
          임계값: 주의 75%, 경고 95%
        </div>
      </div>
    </WidgetCard>
  );
};

export default AnomalyWidget;