"use client"

// Login modal component following Single Responsibility Principle

import type React from "react"
import { useState } from "react"
import { Modal } from "@/components/common/Modal"
import { Button } from "@/components/common/Button"
import { Input } from "@/components/common/Input"
import { DEFAULT_USERS } from "@/constants"
import type { LoginFormData } from "@/types"

interface LoginModalProps {
  isOpen: boolean
  onClose: () => void
  onLogin: (credentials: LoginFormData) => Promise<{ success: boolean; error?: string }>
}

export function LoginModal({ isOpen, onClose, onLogin }: LoginModalProps) {
  const [activeTab, setActiveTab] = useState<"admin" | "user">("admin")
  const [formData, setFormData] = useState<LoginFormData>({
    email: DEFAULT_USERS.ADMIN,
    password: "",
  })
  const [error, setError] = useState("")
  const [loading, setLoading] = useState(false)

  const handleTabChange = (tab: "admin" | "user") => {
    setActiveTab(tab)
    setFormData({
      email: tab === "admin" ? DEFAULT_USERS.ADMIN : DEFAULT_USERS.USER,
      password: "",
    })
    setError("")
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()

    if (!formData.email || !formData.password) {
      setError("Please enter both email and password")
      return
    }

    setLoading(true)
    setError("")

    try {
      const result = await onLogin(formData)
      if (result.success) {
        onClose()
        setFormData({ email: DEFAULT_USERS.ADMIN, password: "" })
      } else {
        setError(result.error || "Login failed")
      }
    } catch (error) {
      setError("Network error. Please try again.")
    } finally {
      setLoading(false)
    }
  }

  const handleClose = () => {
    onClose()
    setError("")
    setFormData({ email: DEFAULT_USERS.ADMIN, password: "" })
  }

  return (
    <Modal isOpen={isOpen} onClose={handleClose} title="Login" size="sm">
      <div className="space-y-4">
        {/* Tabs */}
        <div className="flex gap-2">
          <Button
            type="button"
            variant={activeTab === "admin" ? "primary" : "secondary"}
            size="sm"
            onClick={() => handleTabChange("admin")}
            className="flex-1"
          >
            Admin
          </Button>
          <Button
            type="button"
            variant={activeTab === "user" ? "primary" : "secondary"}
            size="sm"
            onClick={() => handleTabChange("user")}
            className="flex-1"
          >
            User
          </Button>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className="space-y-4">
          <Input
            label="Email"
            type="email"
            value={formData.email}
            onChange={(e) => setFormData((prev) => ({ ...prev, email: e.target.value }))}
            required
            disabled={loading}
          />

          <Input
            label="Password"
            type="password"
            value={formData.password}
            onChange={(e) => setFormData((prev) => ({ ...prev, password: e.target.value }))}
            required
            disabled={loading}
          />

          {error && (
            <p className="text-sm text-red-600" role="alert">
              {error}
            </p>
          )}

          <Button type="submit" className="w-full" loading={loading} disabled={loading}>
            Login
          </Button>
        </form>
      </div>
    </Modal>
  )
}
