import React from 'react';

interface WidgetCardProps {
  title: string;
  icon?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
}

const WidgetCard: React.FC<WidgetCardProps> = ({ title, icon, children, className = '' }) => {
  return (
    <div className={`bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 shadow-sm flex flex-col overflow-hidden ${className}`}>
      <div className="px-5 py-4 border-b border-slate-100 dark:border-slate-800 flex items-center justify-between">
        <div className="flex items-center gap-2">
          {icon && <div className="text-slate-400 dark:text-slate-500">{icon}</div>}
          <h3 className="font-semibold text-slate-800 dark:text-slate-200">{title}</h3>
        </div>
      </div>
      <div className="p-5 flex-1 relative">
        {children}
      </div>
    </div>
  );
};

export default WidgetCard;