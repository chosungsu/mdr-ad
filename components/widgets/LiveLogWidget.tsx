import React, { useEffect, useState } from 'react';
import { List, Info, AlertTriangle, XCircle, Check } from 'lucide-react';
import WidgetCard from '../WidgetCard';
import { LogEntry } from '../../types';
import { useDashboardData } from '../../utils/useDashboardData';

interface LiveLogWidgetProps {
  isRunning: boolean;
}

const LiveLogWidget: React.FC<LiveLogWidgetProps> = ({ isRunning }) => {
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [error, setError] = useState<string | null>(null);
  const dashboardData = useDashboardData(isRunning);

  // Stop 버튼 등으로 isRunning이 false가 되면 UI도 즉시 초기화
  useEffect(() => {
    if (!isRunning) {
      setLogs([]);
      setError(null);
    }
  }, [isRunning]);

  const levelAccent = (level: LogEntry["level"]) => {
    switch (level) {
      case "error":
        return "border-red-500/70";
      case "warning":
        return "border-amber-500/70";
      case "success":
        return "border-emerald-500/70";
      default:
        return "border-blue-500/70";
    }
  };

  const formatMessage = (raw: string) => {
    // {"ts": "...", "data": {...}} 형태면 "컬럼명=값" 7개를 한 줄로 출력
    try {
      const obj = JSON.parse(raw);
      const data = obj?.data;
      if (data && typeof data === "object") {
        // 보안 마스킹(col1..col7) 또는 원본 키(Pressure 등) 둘 다 지원
        const maskedKeys = ["col1", "col2", "col3", "col4", "col5", "col6", "col7"];
        const originalKeys = [
          "Pressure",
          "Power1",
          "Power2",
          "Vibration_Peak1",
          "Vibration_RMS1",
          "Vibration_Peak2",
          "Vibration_RMS2",
        ];
        const useMasked = maskedKeys.some((k) => Object.prototype.hasOwnProperty.call(data, k));
        const keys = useMasked ? maskedKeys : originalKeys;
        const line = keys.map((k) => `${k}=${String(data[k] ?? "")}`).join(", ");
        // UI에서 ellipsis로 줄이되, 비정상적으로 긴 메시지는 방어적으로 제한
        const maxLenHard = 600;
        return line.length > maxLenHard ? `${line.slice(0, maxLenHard)}…` : line;
      }
    } catch {
      // ignore
    }
    // 일반 로그도 UI에서 ellipsis로 줄임
    const maxLenHard = 600;
    return raw.length > maxLenHard ? `${raw.slice(0, maxLenHard)}…` : raw;
  };

  // Update logs when new dashboard data arrives (every 2 seconds)
  useEffect(() => {
    if (!isRunning || !dashboardData) return;

    if (dashboardData.logsSuccess && dashboardData.logs.length > 0) {
      const newLogs: LogEntry[] = dashboardData.logs.map((log) => ({
        id: log.id,
        timestamp: log.timestamp,
        message: formatMessage(log.message),
        level: log.level,
      }));

      // CSV 마지막 행까지 갔다가 wrap되면: 로그 위젯 리스트를 전부 지우고 처음부터 다시 쌓기
      if (dashboardData.wrapped) {
        setLogs(newLogs.slice(0, 200));
      } else {
        // 최신 로그가 항상 위로: 새 로그를 앞에 붙임
        setLogs((prev) => [...newLogs, ...prev].slice(0, 200));
      }
      setError(null);
    } else if (!dashboardData.logsSuccess) {
      setError(dashboardData.logsMessage);
    }
  }, [dashboardData, isRunning]);

  const getIcon = (level: string) => {
    switch (level) {
      case 'error': return <XCircle size={14} className="text-red-500" />;
      case 'warning': return <AlertTriangle size={14} className="text-amber-500" />;
      case 'success': return <Check size={14} className="text-emerald-500" />;
      default: return <Info size={14} className="text-blue-500" />;
    }
  };

  return (
    <WidgetCard title="Live System Logs" icon={<List size={20} />} className="h-full min-h-[300px]">
      <div className="flex h-full flex-col">
        {error && (
          <div className="mb-2 p-2 bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 rounded text-sm text-amber-700 dark:text-amber-400">
            주의: {error}
          </div>
        )}

        {/* 
          - 넓은 화면(lg): 카드 높이를 가득 채우도록 flex-1로 확장
          - 좁은 화면: 한 화면에 로그가 최대 5개 정도만 보이도록 max-height 제한
        */}
        {/* 좁은 화면: 5개 / 넓은 화면: 15개 정도가 보이도록 높이 제한 */}
        <div className="flex-1 min-h-0 overflow-y-auto scrollbar-thin pr-2 relative max-h-[240px] md:max-h-[280px] lg:max-h-[600px]">
          <div className="space-y-2">
            {logs.length === 0 ? (
              <div className="text-center text-slate-400 dark:text-slate-600 py-8">
                로그를 불러오는 중...
              </div>
            ) : (
              logs.map((log) => (
                <div
                  key={log.id}
                  className={[
                    "group flex items-center gap-3 rounded-lg border border-slate-200/60 dark:border-slate-800/60",
                    "bg-white/70 dark:bg-slate-900/40 backdrop-blur",
                    "px-3 py-2 transition-colors",
                    "hover:bg-white dark:hover:bg-slate-900/60",
                    "border-l-4",
                    levelAccent(log.level),
                  ].join(" ")}
                >
                  <div className="flex-shrink-0 opacity-90">{getIcon(log.level)}</div>

                  {/* 메시지는 무조건 1줄로만 (ellipsis) */}
                  <div className="min-w-0 flex-1">
                    <span
                      className={[
                        "block w-full truncate whitespace-nowrap",
                        "font-mono text-xs",
                        log.level === "error"
                          ? "text-red-700 dark:text-red-400"
                          : log.level === "warning"
                            ? "text-amber-700 dark:text-amber-400"
                            : "text-slate-700 dark:text-slate-200",
                      ].join(" ")}
                      title={log.message}
                    >
                      {log.message}
                    </span>
                  </div>

                  <span className="flex-shrink-0 text-[11px] text-slate-400 dark:text-slate-500 font-mono whitespace-nowrap tabular-nums">
                    {new Date(log.timestamp).toLocaleTimeString([], { hour12: false, hour: '2-digit', minute:'2-digit', second:'2-digit' })}
                  </span>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </WidgetCard>
  );
};

export default LiveLogWidget;