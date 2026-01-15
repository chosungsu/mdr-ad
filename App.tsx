import React, { useState, useEffect, Suspense, lazy } from 'react';
import Header from './components/Header';
import AnomalyWidget from './components/widgets/AnomalyWidget';
import SystemSummaryWidget from './components/widgets/SystemSummaryWidget';

// 무거운 위젯(차트, 실시간 로그)을 지연 로딩해서 초기 번들 크기 감소
const LiveLogWidget = lazy(() => import('./components/widgets/LiveLogWidget'));
const RealTimeChartWidget = lazy(() => import('./components/widgets/RealTimeChartWidget'));

const App: React.FC = () => {
  const [isRunning, setIsRunning] = useState(true);
  const [isDark, setIsDark] = useState(false);

  // Handle Dark Mode
  useEffect(() => {
    // Check system preference on load
    if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
      setIsDark(true);
    }
  }, []);

  useEffect(() => {
    if (isDark) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [isDark]);

  const toggleTheme = () => setIsDark(!isDark);
  const toggleRunning = () => setIsRunning(!isRunning);

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-950 transition-colors duration-300 pb-10">
      <Header 
        isRunning={isRunning} 
        toggleRunning={toggleRunning} 
        isDark={isDark} 
        toggleTheme={toggleTheme} 
      />

      <main className="max-w-7xl mx-auto px-4 pt-6">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">

          {/* Top Row: Metrics (좌측 2/3 영역 안에서 서로 나란히 배치) */}
          <div className="col-span-1 md:col-span-2 lg:col-span-2">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <AnomalyWidget isRunning={isRunning} />
              <SystemSummaryWidget isRunning={isRunning} />
            </div>
          </div>
          
          {/* Large Chart - left 영역(2/3 너비) */}
          <div className="col-span-1 md:col-span-2 lg:col-span-2">
            <Suspense
              fallback={
                <div className="h-full w-full flex items-center justify-center text-slate-400 dark:text-slate-600 text-sm">
                  차트 로딩 중...
                </div>
              }
            >
              <RealTimeChartWidget isRunning={isRunning} />
            </Suspense>
          </div>

          {/* Logs - 넓은 화면에서 우측 1/3, 좌측 2행과 높이 맞춤 */}
          <div className="col-span-1 md:col-span-2 lg:col-span-1 lg:row-span-2 lg:col-start-3 lg:row-start-1 h-full">
            <Suspense
              fallback={
                <div className="h-full w-full flex items-center justify-center text-slate-400 dark:text-slate-600 text-sm">
                  로그 위젯 로딩 중...
                </div>
              }
            >
              <LiveLogWidget isRunning={isRunning} />
            </Suspense>
          </div>
          
        </div>
        
        {/* Footer info */}
        <div className="mt-8 text-center text-slate-400 dark:text-slate-600 text-sm">
          <p>© {2023} Bis.ai Dashboard. All systems operational.</p>
        </div>
      </main>
    </div>
  );
};

export default App;