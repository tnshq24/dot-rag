"use client"

// Chat messages container following Single Responsibility Principle

import { useEffect, useRef } from "react"
import { ChatMessage } from "./ChatMessage"
import { LoadingSpinner } from "@/components/common/LoadingSpinner"
import type { ChatMessage as ChatMessageType } from "@/types"

interface ChatMessagesProps {
  messages: ChatMessageType[]
  loading: boolean
}

export function ChatMessages({ messages, loading }: ChatMessagesProps) {
  const messagesEndRef = useRef<HTMLDivElement>(null)

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  return (
    <div className="flex-1 overflow-y-auto bg-muted/30 p-6">
      <div className="mx-auto max-w-4xl space-y-6">
        {messages.map((message) => (
          <ChatMessage key={message.id} message={message} />
        ))}

        {loading && (
          <div className="flex justify-center py-4">
            <LoadingSpinner text="Processing your question..." />
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>
    </div>
  )
}
