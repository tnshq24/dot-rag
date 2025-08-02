"use client"

// Authentication hook following Single Responsibility Principle

import { useState, useEffect, useCallback } from "react"
import { authService } from "@/services/api"
import type { AuthState, LoginFormData } from "@/utils/auth"

interface UseAuthReturn extends AuthState {
  login: (credentials: LoginFormData) => Promise<{ success: boolean; error?: string }>
  logout: () => Promise<void>
  loading: boolean
}

export function useAuth(): UseAuthReturn {
  const [authState, setAuthState] = useState<AuthState>({
    isAuthenticated: false,
    user: null,
    isAdmin: false,
  })
  const [loading, setLoading] = useState(true)

  const checkAuthStatus = useCallback(async () => {
    setLoading(true)
    try {
      const result = await authService.checkAuthStatus()
      if (result.success && result.data) {
        setAuthState(result.data)
      } else {
        setAuthState({
          isAuthenticated: false,
          user: null,
          isAdmin: false,
        })
      }
    } catch (error) {
      console.error("Auth check failed:", error)
      setAuthState({
        isAuthenticated: false,
        user: null,
        isAdmin: false,
      })
    } finally {
      setLoading(false)
    }
  }, [])

  const login = useCallback(async (credentials: LoginFormData) => {
    try {
      const result = await authService.login(credentials)
      if (result.success && result.data) {
        setAuthState(result.data)
        return { success: true }
      } else {
        return { success: false, error: result.error }
      }
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : "Login failed",
      }
    }
  }, [])

  const logout = useCallback(async () => {
    try {
      await authService.logout()
      setAuthState({
        isAuthenticated: false,
        user: null,
        isAdmin: false,
      })
    } catch (error) {
      console.error("Logout failed:", error)
      // Still clear local state even if API call fails
      setAuthState({
        isAuthenticated: false,
        user: null,
        isAdmin: false,
      })
    }
  }, [])

  useEffect(() => {
    checkAuthStatus()
  }, [checkAuthStatus])

  return {
    ...authState,
    login,
    logout,
    loading,
  }
}
