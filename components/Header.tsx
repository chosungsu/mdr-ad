import React from 'react';
import { Play, Pause, Sun, Moon, Activity } from 'lucide-react';

interface HeaderProps {
  isRunning: boolean;
  toggleRunning: () => void;
  isDark: boolean;
  toggleTheme: () => void;
}

const Header: React.FC<HeaderProps> = ({ isRunning, toggleRunning, isDark, toggleTheme }) => {
  return (
    <header className="sticky top-0 z-50 w-full backdrop-blur-md bg-white/80 dark:bg-slate-900/80 border-b border-slate-200 dark:border-slate-800">
      <div className="max-w-7xl mx-auto px-4 h-16 flex items-center justify-between">
        {/* Logo Section */}
        <div className="flex items-center gap-2">
          <div className="bg-brand-600 p-1.5 rounded-lg text-white">
            <Activity size={20} strokeWidth={3} />
          </div>
          <h1 className="text-xl font-bold bg-gradient-to-r from-brand-600 to-brand-400 bg-clip-text text-transparent">
            Bis.ai
          </h1>
        </div>

        {/* Controls Section */}
        <div className="flex items-center gap-4">
          <button
            onClick={toggleRunning}
            className={`flex items-center gap-2 px-4 py-2 rounded-full font-medium transition-all duration-200 ${
              isRunning
                ? 'bg-amber-100 text-amber-700 hover:bg-amber-200 dark:bg-amber-900/30 dark:text-amber-400 dark:hover:bg-amber-900/50'
                : 'bg-emerald-100 text-emerald-700 hover:bg-emerald-200 dark:bg-emerald-900/30 dark:text-emerald-400 dark:hover:bg-emerald-900/50'
            }`}
          >
            {isRunning ? (
              <>
                <Pause size={18} />
                <span>Stop</span>
              </>
            ) : (
              <>
                <Play size={18} />
                <span>Run</span>
              </>
            )}
          </button>

          <div className="h-6 w-px bg-slate-200 dark:bg-slate-700"></div>

          <button
            onClick={toggleTheme}
            className="p-2 rounded-full hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-500 dark:text-slate-400 transition-colors"
            aria-label="Toggle theme"
          >
            {isDark ? <Sun size={20} /> : <Moon size={20} />}
          </button>
        </div>
      </div>
    </header>
  );
};

export default Header;