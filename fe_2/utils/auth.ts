/**
 * Authentication utility module following Single Responsibility Principle
 * Handles JWT token management and validation
 */

export class TokenManager {
  private static readonly TOKEN_KEY = 'auth_token'

  /**
   * Get the stored JWT token
   */
  static getToken(): string | null {
    if (typeof window === 'undefined') return null
    return localStorage.getItem(this.TOKEN_KEY)
  }

  /**
   * Store the JWT token
   */
  static setToken(token: string): void {
    if (typeof window === 'undefined') return
    localStorage.setItem(this.TOKEN_KEY, token)
  }

  /**
   * Remove the stored JWT token
   */
  static removeToken(): void {
    if (typeof window === 'undefined') return
    localStorage.removeItem(this.TOKEN_KEY)
  }

  /**
   * Check if a JWT token is valid and not expired
   */
  static isTokenValid(token: string): boolean {
    try {
      const payload = JSON.parse(atob(token.split('.')[1]))
      return payload.exp * 1000 > Date.now()
    } catch {
      return false
    }
  }

  /**
   * Get authorization header with Bearer token if available
   */
  static getAuthHeader(): Record<string, string> {
    const token = this.getToken()
    if (token && this.isTokenValid(token)) {
      return { "Authorization": `Bearer ${token}` }
    }
    return {}
  }

  /**
   * Check if user is currently authenticated
   */
  static isAuthenticated(): boolean {
    const token = this.getToken()
    return token !== null && this.isTokenValid(token)
  }
}

/**
 * Authentication state interface
 */
export interface AuthState {
  isAuthenticated: boolean
  user: {
    user_id: string
    email: string
  } | null
  isAdmin: boolean
}

/**
 * Login form data interface
 */
export interface LoginFormData {
  email: string
  password: string
} 