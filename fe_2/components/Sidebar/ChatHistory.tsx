"use client"

// Chat history component following Single Responsibility Principle

import type React from "react"
import { MessageSquare, Trash2 } from "lucide-react"
import { Button } from "@/components/common/Button"
import { formatISTDateTime } from "@/utils"
import { cn } from "@/lib/utils"
import type { ChatSession } from "@/types"

interface ChatHistoryProps {
  sessions: ChatSession[]
  activeSessionId: string | null
  onSessionSelect: (sessionId: string) => void
  onSessionDelete: (sessionId: string) => void
}

export function ChatHistory({ sessions, activeSessionId, onSessionSelect, onSessionDelete }: ChatHistoryProps) {
  const handleDeleteClick = (e: React.MouseEvent, sessionId: string) => {
    e.stopPropagation()
    if (confirm("Delete this chat session?")) {
      onSessionDelete(sessionId)
    }
  }

  if (sessions.length === 0) {
    return (
      <div className="flex-1 flex items-center justify-center text-muted-foreground text-sm">No chat history yet</div>
    )
  }

  return (
    <div className="flex-1 overflow-y-auto space-y-1 p-2">
      {sessions.map((session) => (
        <div
          key={session.session_id}
          className={cn(
            "group flex items-center gap-3 rounded-lg p-3 cursor-pointer transition-colors",
            "hover:bg-accent/50",
            activeSessionId === session.session_id && "bg-accent",
          )}
          onClick={() => onSessionSelect(session.session_id)}
        >
          <MessageSquare className="h-4 w-4 text-muted-foreground flex-shrink-0" />

          <div className="flex-1 min-w-0">
            <p className="text-sm font-medium truncate">{session.question}</p>
            <p className="text-xs text-muted-foreground">{formatISTDateTime(session.timestamp)}</p>
          </div>

          <Button
            variant="ghost"
            size="sm"
            className="h-8 w-8 p-0 opacity-0 group-hover:opacity-100 transition-opacity"
            onClick={(e) => handleDeleteClick(e, session.session_id)}
            title="Delete chat"
          >
            <Trash2 className="h-4 w-4 text-muted-foreground hover:text-red-500" />
          </Button>
        </div>
      ))}
    </div>
  )
}
