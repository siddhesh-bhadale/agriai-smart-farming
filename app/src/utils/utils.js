import React, { useState, useEffect } from 'react';
import { NativeModules } from 'react-native';
const AppUsageModule = NativeModules.AppUsageModule;

export function formatTime(timeInMilliseconds) {
  const seconds = Math.floor(timeInMilliseconds / 1000);

  if (seconds < 60) {
    return `${seconds}s`;
  } else if (seconds < 3600) {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}m, ${remainingSeconds}s`;
  } else if (seconds < 86400) {
    const hours = Math.floor(seconds / 3600);
    const remainingMinutes = seconds % 3600;
    const remainingSeconds = Math.floor(remainingMinutes / 60);
    return `${hours}h, ${remainingMinutes % 60}m, ${remainingSeconds}s`;
  } else if (seconds < 2592000) { // 30 days
    const days = Math.floor(seconds / 86400);
    const remainingHours = seconds % 86400;
    const remainingMinutes = Math.floor(remainingHours / 3600);
    return `${days}d, ${remainingHours % 3600}h`;
  } else if (seconds < 31536000) { // 12 months
    const months = Math.floor(seconds / 2592000);
    const remainingDays = seconds % 2592000;
    const remainingHours = Math.floor(remainingDays / 86400);
    return `${months} months, ${remainingDays % 86400 / 24}d`;
  } else {
    const years = Math.floor(seconds / 31536000);
    const remainingMonths = seconds % 31536000;
    const remainingDays = Math.floor(remainingMonths / 2592000);
    return `${years} year, ${remainingMonths % 2592000 / 24} months, ${remainingDays % 24}d`;
  }
}

export const validateEmail = (email) => {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return emailRegex.test(email);
};

export const validatePassword = (password) => {
  // Add your password validation logic here
  return password.length >= 6;
};

export const LoadAppUsageData = () => {
  const [usageData, setUsageData] = useState([]);

  useEffect(() => {
    AppUsageModule.getAppUsageData((data) => {
      setUsageData(data);
    });
  }, []);

  return usageData;
};
