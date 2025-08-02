// Core application types and interfaces following Interface Segregation Principle

export interface User {
  user_id: string
  email: string
}

export interface ChatMessage {
  id: string
  content: string
  isUser: boolean
  timestamp?: string
  sourceDocuments?: SourceDocument[]
}

export interface SourceDocument {
  filename: string
  page_number: number | number[]
  content: string | string[]
  pages_content?: any
}

export interface ChatSession {
  session_id: string
  question: string
  timestamp: string
  user_id: string
}

export interface FileItem {
  value: string
  label: string
}

export interface ChatRequest {
  question: string
  user_id: string
  conversation_id: string
  session_id: string
  file_names: string[]
}

export interface ChatResponse {
  answer: string
  timestamp: string
  source_documents: SourceDocument[]
}

export interface UploadFormData {
  filename: string
  projectCode: string
  labelTag: string
}

// Modal state interfaces
export interface ModalState {
  isOpen: boolean
}



// API response types
export interface ApiResponse<T = any> {
  success: boolean
  data?: T
  error?: string
}

export interface SessionMessagesResponse {
  messages: Array<{
    question: string
    answer: string
    timestamp: string
    source_documents: SourceDocument[]
  }>
}

export interface UserSessionsResponse {
  sessions: ChatSession[]
}

export interface AvailableFilesResponse {
  files: FileItem[]
}

// Additional types for enhanced functionality
export interface PdfHighlightRequest {
  filename: string
  page_number: number[]
  content: string[]
  pages_content?: any
}

export interface UploadResult {
  filename: string
  status: string
}

export interface UploadResponse {
  results: UploadResult[]
  metadata: {
    filename: string
    project_code: string
    label_tag: string
  }
}

export interface SessionMessage {
  question: string
  answer: string
  timestamp: string
  rephrased_question?: string
  retrieved_documents?: SourceDocument[]
  source_documents: SourceDocument[]
}

export interface HealthResponse {
  status: string
  pipeline_initialized: boolean
}
