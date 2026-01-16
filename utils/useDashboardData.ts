/**
 * Unified dashboard data hook
 * 
 * Single API call (/api/logs) provides both logs and scores data.
 * All widgets share this hook for optimal performance.
 * 
 * Updates every 2 seconds for smooth animation while reducing server load.
 */

import { useState, useEffect, useRef } from 'react';
import { backendApi } from './api';

export interface LogEntry {
  id: string;
  timestamp: number;
  message: string;
  level: string;
}

export interface DashboardData {
  // Logs data
  logs: LogEntry[];
  logsSuccess: boolean;
  logsMessage: string;
  cursor: number;
  wrapped: boolean;
  
  // Scores data
  scores: {
    mdrad: number | null;
  };
  scoresTimestamp: string | null;
  scoresStatus: 'success' | 'error';
  
  // Common
  timestamp: string;
}

// Singleton state shared across all hook instances
let sharedData: DashboardData | null = null;
let sharedListeners: Set<(data: DashboardData) => void> = new Set();
let sharedInterval: NodeJS.Timeout | null = null;
let activeSubscribers = 0;
let currentCursor = 0;

/**
 * Fetch dashboard data from backend and notify all listeners
 */
async function fetchAndNotify() {
  try {
    const response = await backendApi.getSystemLogs({
      limit: 1,
      cursor: currentCursor,
      wrap: true,
    });
    
    sharedData = {
      // Logs data
      logs: response.logs.map((log: any, index: number) => ({
        id: `log-${log.id ?? index}-${log.timestamp}`,
        timestamp: new Date(log.timestamp).getTime(),
        message: log.message,
        level: log.level,
      })),
      logsSuccess: response.success,
      logsMessage: response.message,
      cursor: response.next_cursor,
      wrapped: response.wrapped,
      
      // Scores data (from integrated API)
      scores: {
        mdrad: response.scores?.mdrad ?? null,
      },
      scoresTimestamp: response.scores_timestamp ?? null,
      scoresStatus: response.scores_status === 'error' ? 'error' : 'success',
      
      // Common
      timestamp: new Date().toISOString(),
    };
    
    // Update cursor for next fetch
    if (response.next_cursor !== undefined) {
      currentCursor = response.next_cursor;
    }
    
    // Notify all subscribers
    sharedListeners.forEach(listener => listener(sharedData!));
  } catch (error: any) {
    console.error('Failed to fetch dashboard data:', error);
    // Don't update sharedData on error to keep last known good state
  }
}

/**
 * Start the shared polling interval (every 2 seconds)
 */
function startPolling() {
  if (sharedInterval) return; // Already polling
  
  // Fetch immediately
  fetchAndNotify();
  
  // Then poll every 2 seconds
  sharedInterval = setInterval(fetchAndNotify, 2000);
}

/**
 * Stop the shared polling interval
 */
function stopPolling() {
  if (sharedInterval) {
    clearInterval(sharedInterval);
    sharedInterval = null;
  }
}

/**
 * Reset cursor to start from beginning
 */
export function resetDashboardCursor() {
  currentCursor = 0;
}

/**
 * Hook to get unified dashboard data (logs + scores)
 * 
 * @param isRunning - Whether the dashboard is running
 * @returns Latest dashboard data
 */
export function useDashboardData(isRunning: boolean): DashboardData | null {
  const [data, setData] = useState<DashboardData | null>(sharedData);
  const listenerRef = useRef<((data: DashboardData) => void) | null>(null);

  useEffect(() => {
    if (!isRunning) {
      // Unsubscribe when not running
      if (listenerRef.current) {
        sharedListeners.delete(listenerRef.current);
        listenerRef.current = null;
        activeSubscribers--;
      }
      
      // Stop polling if no active subscribers
      if (activeSubscribers === 0) {
        stopPolling();
        resetDashboardCursor(); // Reset cursor when stopping
      }
      
      return;
    }

    // Subscribe to updates
    activeSubscribers++;
    listenerRef.current = (newData: DashboardData) => {
      setData(newData);
    };
    sharedListeners.add(listenerRef.current);

    // Start polling if not already started
    startPolling();

    // Cleanup
    return () => {
      if (listenerRef.current) {
        sharedListeners.delete(listenerRef.current);
        listenerRef.current = null;
        activeSubscribers--;
      }
      
      // Stop polling if no active subscribers
      if (activeSubscribers === 0) {
        stopPolling();
        resetDashboardCursor(); // Reset cursor when stopping
      }
    };
  }, [isRunning]);

  return data;
}
