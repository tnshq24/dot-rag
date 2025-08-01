"use client"

// Main sidebar component following Single Responsibility Principle

import type React from "react"
import { useState, useRef, useEffect } from "react"
import { Plus, Trash2, Menu, X } from "lucide-react"
import { Button } from "@/components/common/Button"
import { ChatHistory } from "./ChatHistory"
import { UserSection } from "./UserSection"
import { clamp } from "@/utils"
import { UI_CONSTANTS } from "@/constants"
import { cn } from "@/lib/utils"
import type { ChatSession, User } from "@/types"

interface SidebarProps {
  sessions: ChatSession[]
  activeSessionId: string | null
  user: User | null
  isAuthenticated: boolean
  onNewChat: () => void
  onSessionSelect: (sessionId: string) => void
  onSessionDelete: (sessionId: string) => void
  onDeleteAll: () => void
  onLogin: () => void
  onLogout: () => void
  className?: string
}

export function Sidebar({
  sessions,
  activeSessionId,
  user,
  isAuthenticated,
  onNewChat,
  onSessionSelect,
  onSessionDelete,
  onDeleteAll,
  onLogin,
  onLogout,
  className,
}: SidebarProps) {
  const [isCollapsed, setIsCollapsed] = useState(false)
  const [width, setWidth] = useState(UI_CONSTANTS.SIDEBAR_DEFAULT_WIDTH)
  const [isResizing, setIsResizing] = useState(false)

  const sidebarRef = useRef<HTMLDivElement>(null)
  const startXRef = useRef(0)
  const startWidthRef = useRef(0)

  const handleDeleteAll = () => {
    if (confirm("Are you sure you want to delete ALL chat sessions? This cannot be undone.")) {
      onDeleteAll()
    }
  }

  const handleMouseDown = (e: React.MouseEvent) => {
    setIsResizing(true)
    startXRef.current = e.clientX
    startWidthRef.current = width
    document.body.style.cursor = "ew-resize"
    document.body.style.userSelect = "none"
  }

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!isResizing) return

      const deltaX = e.clientX - startXRef.current
      const newWidth = clamp(
        startWidthRef.current + deltaX,
        UI_CONSTANTS.SIDEBAR_MIN_WIDTH,
        UI_CONSTANTS.SIDEBAR_MAX_WIDTH,
      )
      setWidth(newWidth)
    }

    const handleMouseUp = () => {
      setIsResizing(false)
      document.body.style.cursor = ""
      document.body.style.userSelect = ""
    }

    if (isResizing) {
      document.addEventListener("mousemove", handleMouseMove)
      document.addEventListener("mouseup", handleMouseUp)
    }

    return () => {
      document.removeEventListener("mousemove", handleMouseMove)
      document.removeEventListener("mouseup", handleMouseUp)
    }
  }, [isResizing])

  return (
    <>
      {/* Mobile toggle button */}
      <Button
        variant="ghost"
        size="sm"
        className="fixed top-4 left-4 z-50 md:hidden"
        onClick={() => setIsCollapsed(!isCollapsed)}
      >
        {isCollapsed ? <Menu className="h-4 w-4" /> : <X className="h-4 w-4" />}
      </Button>

      {/* Sidebar */}
      <div
        ref={sidebarRef}
        className={cn(
          "relative flex flex-col bg-muted/30 border-r transition-all duration-300",
          "md:relative md:translate-x-0",
          isCollapsed ? "fixed inset-y-0 left-0 z-40 translate-x-0" : "fixed inset-y-0 left-0 z-40 -translate-x-full",
          "md:flex",
          className,
        )}
        style={{ width: isCollapsed ? 0 : width }}
      >
        {!isCollapsed && (
          <>
            {/* Header */}
            <div className="border-b p-4">
              <div className="flex items-center justify-between">
                <h2 className="text-lg font-semibold">Chats</h2>
                {isAuthenticated && sessions.length > 0 && (
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={handleDeleteAll}
                    className="h-8 w-8 p-0 text-red-500 hover:text-red-600"
                    title="Delete all sessions"
                  >
                    <Trash2 className="h-4 w-4" />
                  </Button>
                )}
              </div>

              {isAuthenticated && (
                <Button onClick={onNewChat} className="mt-3 w-full justify-start" variant="secondary">
                  <Plus className="mr-2 h-4 w-4" />
                  New Chat
                </Button>
              )}
            </div>

            {/* Chat History */}
            {isAuthenticated && (
              <ChatHistory
                sessions={sessions}
                activeSessionId={activeSessionId}
                onSessionSelect={onSessionSelect}
                onSessionDelete={onSessionDelete}
              />
            )}

            {/* User Section */}
            <UserSection user={user} isAuthenticated={isAuthenticated} onLogin={onLogin} onLogout={onLogout} />

            {/* Resize handle */}
            <div
              className="absolute top-0 right-0 bottom-0 w-1 cursor-ew-resize bg-transparent hover:bg-border transition-colors"
              onMouseDown={handleMouseDown}
            />
          </>
        )}
      </div>

      {/* Mobile overlay */}
      {!isCollapsed && (
        <div className="fixed inset-0 z-30 bg-black/50 md:hidden" onClick={() => setIsCollapsed(true)} />
      )}
    </>
  )
}
