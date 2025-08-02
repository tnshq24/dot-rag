// API service layer following Dependency Inversion Principle

import { API_ENDPOINTS } from "@/constants"
import { TokenManager } from "@/utils/auth"
import type {
  ApiResponse,
  ChatRequest,
  ChatResponse,
  ChatSession,
  SessionMessagesResponse,
  UserSessionsResponse,
  AvailableFilesResponse,
  SourceDocument,
  UploadFormData,
  UploadResponse,
  PdfHighlightRequest,
  HealthResponse,
} from "@/types"
import type { AuthState, LoginFormData } from "@/utils/auth"

/**
 * Base API service class implementing common HTTP operations
 */
class BaseApiService {
  protected async request<T>(url: string, options: RequestInit = {}): Promise<ApiResponse<T>> {
    try {
      const headers: Record<string, string> = {
        "Content-Type": "application/json",
        ...TokenManager.getAuthHeader(),
        ...options.headers,
      }

      const response = await fetch(url, {
        headers,
        ...options,
      })

      const data = await response.json()

      if (response.ok) {
        return { success: true, data }
      } else {
        return { success: false, error: data.error || "Request failed" }
      }
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : "Network error",
      }
    }
  }

  protected async uploadRequest<T>(url: string, formData: FormData): Promise<ApiResponse<T>> {
    try {
      const headers: Record<string, string> = {
        ...TokenManager.getAuthHeader(),
      }

      const response = await fetch(url, {
        method: "POST",
        headers,
        body: formData,
      })

      const data = await response.json()

      if (response.ok) {
        return { success: true, data }
      } else {
        return { success: false, error: data.error || "Upload failed" }
      }
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : "Upload error",
      }
    }
  }
}

/**
 * Authentication service
 */
export class AuthService extends BaseApiService {
  async checkAuthStatus(): Promise<ApiResponse<AuthState>> {
    // If no token or invalid token, return not authenticated
    if (!TokenManager.isAuthenticated()) {
      TokenManager.removeToken()
      return {
        success: true,
        data: {
          isAuthenticated: false,
          user: null,
          isAdmin: false,
        },
      }
    }

    const result = await this.request<any>(API_ENDPOINTS.CHECK_AUTH)

    if (result.success && result.data) {
      return {
        success: true,
        data: {
          isAuthenticated: result.data.authenticated,
          user: result.data.authenticated
            ? {
              user_id: result.data.user_id,
              email: result.data.email,
            }
            : null,
          isAdmin: result.data.isadmin || false,
        },
      }
    }

    // If check_auth fails, clear token and return not authenticated
    TokenManager.removeToken()
    return {
      success: true,
      data: {
        isAuthenticated: false,
        user: null,
        isAdmin: false,
      },
    }
  }

  async login(credentials: LoginFormData): Promise<ApiResponse<AuthState>> {
    const result = await this.request<any>(API_ENDPOINTS.LOGIN, {
      method: "POST",
      body: JSON.stringify(credentials),
    })

    if (result.success && result.data && result.data.token) {
      // Store the JWT token
      TokenManager.setToken(result.data.token)
      
      return {
        success: true,
        data: {
          isAuthenticated: true,
          user: {
            user_id: result.data.user_id,
            email: result.data.email,
          },
          isAdmin: credentials.email === "admin@xyz.com",
        },
      }
    }

    return result
  }

  async logout(): Promise<ApiResponse<void>> {
    const result = await this.request<void>(API_ENDPOINTS.LOGOUT, {
      method: "POST",
    })
    
    // Always clear the token on logout
    TokenManager.removeToken()
    
    return result
  }
}

/**
 * Chat service
 */
export class ChatService extends BaseApiService {
  async sendMessage(request: ChatRequest): Promise<ApiResponse<ChatResponse>> {
    return this.request<ChatResponse>(API_ENDPOINTS.CHAT, {
      method: "POST",
      body: JSON.stringify(request),
    })
  }

  async getUserSessions(): Promise<ApiResponse<ChatSession[]>> {
    const result = await this.request<UserSessionsResponse>(API_ENDPOINTS.USER_SESSIONS)

    if (result.success && result.data) {
      return {
        success: true,
        data: result.data.sessions,
      }
    }

    return { success: false, error: result.error }
  }

  async getSessionMessages(sessionId: string): Promise<ApiResponse<SessionMessagesResponse>> {
    return this.request<SessionMessagesResponse>(
      `${API_ENDPOINTS.SESSION_MESSAGES}?session_id=${encodeURIComponent(sessionId)}`,
    )
  }

  async deleteSession(sessionId: string): Promise<ApiResponse<void>> {
    return this.request<void>(API_ENDPOINTS.DELETE_SESSION, {
      method: "POST",
      body: JSON.stringify({ session_id: sessionId }),
    })
  }
}

/**
 * File service
 */
export class FileService extends BaseApiService {
  async getAvailableFiles(): Promise<ApiResponse<AvailableFilesResponse>> {
    return this.request<AvailableFilesResponse>(API_ENDPOINTS.AVAILABLE_FILES)
  }

  async uploadPDF(files: FileList, formData: UploadFormData): Promise<ApiResponse<UploadResponse>> {
    const uploadFormData = new FormData()

    for (let i = 0; i < files.length; i++) {
      uploadFormData.append("pdfs", files[i])
    }

    uploadFormData.append("field1", formData.filename)
    uploadFormData.append("field2", formData.projectCode)
    uploadFormData.append("field3", formData.labelTag)

    return this.uploadRequest<UploadResponse>(API_ENDPOINTS.UPLOAD_PDF, uploadFormData)
  }

  async viewHighlights(document: SourceDocument): Promise<{ blob: Blob; pageNumber?: string } | null> {
    try {
      const headers: Record<string, string> = {
        "Content-Type": "application/json",
        ...TokenManager.getAuthHeader(),
      }

      const response = await fetch(API_ENDPOINTS.VIEW_HIGHLIGHTS, {
        method: "POST",
        headers,
        body: JSON.stringify(document),
      })

      if (response.ok) {
        const contentType = response.headers.get("content-type")
        if (contentType && contentType.includes("application/pdf")) {
          const blob = await response.blob()
          const pageNumber = response.headers.get("X-Page-Number")
          return { blob, pageNumber: pageNumber || undefined }
        }
      }

      return null
    } catch (error) {
      console.error("Error fetching highlighted PDF:", error)
      return null
    }
  }

  async viewPDF(blobName: string): Promise<Blob | null> {
    try {
      const headers: Record<string, string> = {
        ...TokenManager.getAuthHeader(),
      }

      const encodedBlobName = blobName.replace("/", "@")
      const response = await fetch(`${API_ENDPOINTS.VIEW_PDF}/${encodedBlobName}`, {
        headers,
      })

      if (response.ok) {
        const contentType = response.headers.get("content-type")
        if (contentType && contentType.includes("application/pdf")) {
          return await response.blob()
        }
      }

      return null
    } catch (error) {
      console.error("Error fetching PDF:", error)
      return null
    }
  }

  async checkHealth(): Promise<ApiResponse<HealthResponse>> {
    return this.request<HealthResponse>(API_ENDPOINTS.HEALTH)
  }
}

// Service instances - Singleton pattern
export const authService = new AuthService()
export const chatService = new ChatService()
export const fileService = new FileService()
