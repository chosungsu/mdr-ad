/**
 * Shared realtime logs hook
 * 
 * All components use this hook to get the latest logs from a single API call.
 * This prevents duplicate API requests and improves performance.
 */

import { useState, useEffect, useRef } from 'react';
import { backendApi } from './api';

export interface LogEntry {
  id: string;
  timestamp: number;
  message: string;
  level: string;
}

export interface RealtimeLogsData {
  success: boolean;
  logs: LogEntry[];
  count: number;
  message: string;
  last_id: number;
  next_cursor: number;
  wrapped: boolean;
}

// Singleton state shared across all hook instances
let sharedData: RealtimeLogsData | null = null;
let sharedListeners: Set<(data: RealtimeLogsData) => void> = new Set();
let sharedInterval: NodeJS.Timeout | null = null;
let activeSubscribers = 0;
let currentCursor = 0;

/**
 * Fetch logs from backend and notify all listeners
 */
async function fetchAndNotify() {
  try {
    const response = await backendApi.getSystemLogs({
      limit: 1,
      cursor: currentCursor,
      wrap: true,
    });
    
    sharedData = {
      success: response.success,
      logs: response.logs.map((log: any, index: number) => ({
        id: `log-${log.id ?? index}-${log.timestamp}`,
        timestamp: new Date(log.timestamp).getTime(),
        message: log.message,
        level: log.level,
      })),
      count: response.count,
      message: response.message,
      last_id: response.last_id,
      next_cursor: response.next_cursor,
      wrapped: response.wrapped,
    };
    
    // Update cursor for next fetch
    if (response.next_cursor !== undefined) {
      currentCursor = response.next_cursor;
    }
    
    // Notify all subscribers
    sharedListeners.forEach(listener => listener(sharedData!));
  } catch (error: any) {
    console.error('Failed to fetch realtime logs:', error);
    // Don't update sharedData on error to keep last known good state
  }
}

/**
 * Start the shared polling interval
 */
function startPolling() {
  if (sharedInterval) return; // Already polling
  
  // Fetch immediately
  fetchAndNotify();
  
  // Then poll every second
  sharedInterval = setInterval(fetchAndNotify, 1000);
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
export function resetLogsCursor() {
  currentCursor = 0;
}

/**
 * Hook to get realtime logs
 * 
 * @param isRunning - Whether the dashboard is running
 * @returns Latest realtime logs data
 */
export function useRealtimeLogs(isRunning: boolean): RealtimeLogsData | null {
  const [data, setData] = useState<RealtimeLogsData | null>(sharedData);
  const listenerRef = useRef<((data: RealtimeLogsData) => void) | null>(null);

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
        resetLogsCursor(); // Reset cursor when stopping
      }
      
      return;
    }

    // Subscribe to updates
    activeSubscribers++;
    listenerRef.current = (newData: RealtimeLogsData) => {
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
        resetLogsCursor(); // Reset cursor when stopping
      }
    };
  }, [isRunning]);

  return data;
}
