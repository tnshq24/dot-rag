// Application-wide constants and configuration

// API Base URL pointing to Python backend
export const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:5001"

export const API_ENDPOINTS = {
  CHECK_AUTH: `${API_BASE_URL}/check_auth`,
  LOGIN: `${API_BASE_URL}/login`,
  LOGOUT: `${API_BASE_URL}/logout`,
  CHAT: `${API_BASE_URL}/chat`,
  UPLOAD_PDF: `${API_BASE_URL}/upload_pdf`,
  DELETE_SESSION: `${API_BASE_URL}/delete_session`,
  USER_SESSIONS: `${API_BASE_URL}/user_sessions`,
  SESSION_MESSAGES: `${API_BASE_URL}/session_messages`,
  AVAILABLE_FILES: `${API_BASE_URL}/available_files`,
  VIEW_HIGHLIGHTS: `${API_BASE_URL}/view_highlights`,
  VIEW_PDF: `${API_BASE_URL}/view_pdf`,
  HEALTH: `${API_BASE_URL}/health`,
} as const

export const USER_TYPES = {
  ADMIN: "admin",
  USER: "user",
} as const

export const DEFAULT_USERS = {
  ADMIN: "admin@xyz.com",
  USER: "user1@xyz.com",
} as const

export const UI_CONSTANTS = {
  SIDEBAR_MIN_WIDTH: 200,
  SIDEBAR_MAX_WIDTH: 400,
  SIDEBAR_DEFAULT_WIDTH: 260,
  TEXTAREA_MAX_HEIGHT: 200,
  MESSAGE_MAX_WIDTH: "60%",
} as const

export const MESSAGES = {
  WELCOME: "Hi, I am the DoT chatbot. How can I help you?",
  LOGIN_REQUIRED: "Please login to continue",
  UPLOAD_SUCCESS: "Your PDF has been successfully processed. You can now ask questions based on its content.",
  NETWORK_ERROR: "Network error. Please try again.",
  DELETE_CONFIRM: "Are you sure you want to delete ALL chat sessions? This cannot be undone.",
  DELETE_SESSION_CONFIRM: "Delete this chat session?",
} as const

export const FILE_CONSTRAINTS = {
  MAX_FILES: 3,
  ACCEPTED_TYPES: ".pdf",
  MAX_FILE_SIZE: 10 * 1024 * 1024, // 10MB
} as const
