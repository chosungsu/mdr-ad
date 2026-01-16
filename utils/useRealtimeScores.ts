/**
 * Shared realtime scores hook
 * 
 * All widgets use this hook to get the latest scores from a single API call.
 * This prevents duplicate API requests and improves performance.
 */

import { useState, useEffect, useRef } from 'react';
import { backendApi } from './api';

export interface RealtimeScoresData {
  timestamp: string;
  cursor: number;
  scores: {
    mdrad: number | null;
  };
  status: 'success' | 'error';
  error?: string;
}

// Singleton state shared across all hook instances
let sharedData: RealtimeScoresData | null = null;
let sharedListeners: Set<(data: RealtimeScoresData) => void> = new Set();
let sharedInterval: NodeJS.Timeout | null = null;
let activeSubscribers = 0;

/**
 * Fetch scores from backend and notify all listeners
 */
async function fetchAndNotify() {
  try {
    const response = await backendApi.getRealtimeScores();
    sharedData = {
      timestamp: response.timestamp,
      cursor: response.cursor,
      scores: response.scores,
      status: response.status,
      error: response.error,
    };
    
    // Notify all subscribers
    sharedListeners.forEach(listener => listener(sharedData!));
  } catch (error: any) {
    console.error('Failed to fetch realtime scores:', error);
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
 * Hook to get realtime scores
 * 
 * @param isRunning - Whether the dashboard is running
 * @returns Latest realtime scores data
 */
export function useRealtimeScores(isRunning: boolean): RealtimeScoresData | null {
  const [data, setData] = useState<RealtimeScoresData | null>(sharedData);
  const listenerRef = useRef<((data: RealtimeScoresData) => void) | null>(null);

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
      }
      
      return;
    }

    // Subscribe to updates
    activeSubscribers++;
    listenerRef.current = (newData: RealtimeScoresData) => {
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
      }
    };
  }, [isRunning]);

  return data;
}
