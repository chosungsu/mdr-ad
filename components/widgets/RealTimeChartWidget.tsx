import React, { useState, useEffect } from 'react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { Activity } from 'lucide-react';
import WidgetCard from '../WidgetCard';
import { backendApi } from '../../utils/api';

interface RealTimeChartWidgetProps {
  isRunning: boolean;
}

type DualMetricPoint = {
  time: string;
  mdrad: number;
};

const RealTimeChartWidget: React.FC<RealTimeChartWidgetProps> = ({ isRunning }) => {
  const [data, setData] = useState<DualMetricPoint[]>([]);

  // Stop이면 UI 초기화, Run이면 초기 데이터 재구성
  useEffect(() => {
    if (!isRunning) {
      setData([]);
      return;
    }
  }, [isRunning]);

  useEffect(() => {
    if (!isRunning) return;

    const interval = setInterval(async () => {
      try {
        const response = await backendApi.getRealtimeScores();
        if (response.status === 'success' && response.scores) {
          const now = new Date();
          const newPoint: DualMetricPoint = {
            time: now.toLocaleTimeString([], { hour12: false, minute: '2-digit', second: '2-digit' }),
            mdrad: response.scores.mdrad ?? 0,
          };

          setData(prevData => {
            const newData = [...prevData, newPoint];
            if (newData.length > 30) newData.shift(); // Keep last 30 points
            return newData;
          });
        }
      } catch (err: any) {
        console.error('모델 점수 로드 실패:', err);
        // 에러 시 UI 경고는 띄우지 않고 기존 데이터 유지
      }
    }, 1000);

    return () => {
      clearInterval(interval);
    };
  }, [isRunning]);

  return (
    <WidgetCard title="Live dashboard" icon={<Activity size={20} />} className="col-span-1 lg:col-span-2 min-h-[350px] h-full">
      <div className="h-full w-full min-h-[300px] pt-4 relative">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart
            data={data}
            margin={{ top: 5, right: 0, left: -20, bottom: 5 }}
          >
            <defs>
              <linearGradient id="colorMDRAD" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#10b981" stopOpacity={0.22}/>
                <stop offset="95%" stopColor="#10b981" stopOpacity={0}/>
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#94a3b8" opacity={0.2} />
            <XAxis 
              dataKey="time" 
              tick={{ fontSize: 12, fill: '#94a3b8' }} 
              tickLine={false}
              axisLine={false}
              minTickGap={30}
            />
            <YAxis tick={{ fontSize: 12, fill: '#94a3b8' }} tickLine={false} axisLine={false} />
            <Tooltip 
              contentStyle={{ 
                backgroundColor: 'var(--tooltip-bg)', 
                borderColor: 'var(--tooltip-border)', 
                borderRadius: '8px',
                color: 'var(--tooltip-text)'
              }}
              itemStyle={{ color: 'var(--tooltip-text)' }}
            />
            <Legend
              verticalAlign="bottom"
              align="right"
              wrapperStyle={{ paddingTop: 6 }}
            />
            <Area
              type="monotone"
              dataKey="mdrad"
              name="MDRAD"
              stroke="#10b981"
              strokeWidth={3}
              fillOpacity={1}
              fill="url(#colorMDRAD)"
              isAnimationActive={false}
            />
          </AreaChart>
        </ResponsiveContainer>
        
        {/* CSS variable injection for Recharts tooltip theme support */}
        <style>{`
          :root {
            --tooltip-bg: #ffffff;
            --tooltip-border: #e2e8f0;
            --tooltip-text: #1e293b;
          }
          .dark {
            --tooltip-bg: #0f172a;
            --tooltip-border: #1e293b;
            --tooltip-text: #f8fafc;
          }
        `}</style>
      </div>
    </WidgetCard>
  );
};

export default RealTimeChartWidget;