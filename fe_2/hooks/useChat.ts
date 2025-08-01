"use client"

// Chat management hook following Single Responsibility Principle

import { useState, useCallback, useRef } from "react"
import { chatService } from "@/services/api"
import { generateUUID } from "@/utils"
import type { ChatMessage, ChatSession, ChatRequest } from "@/types"

interface UseChatReturn {
  messages: ChatMessage[]
  sessions: ChatSession[]
  currentSessionId: string | null
  activeSessionId: string | null
  loading: boolean
  sendMessage: (content: string, fileNames?: string[]) => Promise<void>
  loadSession: (sessionId: string) => Promise<void>
  startNewChat: () => void
  deleteSession: (sessionId: string) => Promise<void>
  loadChatHistory: () => Promise<void>
  clearMessages: () => void
  addMessage: (content: string, isUser: boolean, timestamp?: string, sourceDocuments?: any[]) => void
}

export function useChat(userId?: string): UseChatReturn {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [sessions, setSessions] = useState<ChatSession[]>([])
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null)
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  const currentConversationId = useRef<string | null>(null)

  const addMessage = useCallback((content: string, isUser: boolean, timestamp?: string, sourceDocuments?: any[]) => {
    const newMessage: ChatMessage = {
      id: generateUUID(),
      content,
      isUser,
      timestamp,
      sourceDocuments,
    }

    setMessages((prev) => [...prev, newMessage])
  }, [])

  const clearMessages = useCallback(() => {
    setMessages([])
  }, [])

  const loadChatHistory = useCallback(async () => {
    if (!userId) return

    try {
      const result = await chatService.getUserSessions()
      if (result.success && result.data) {
        setSessions(result.data)
      }
    } catch (error) {
      console.error("Error loading chat history:", error)
    }
  }, [userId])

  const loadSession = useCallback(
    async (sessionId: string) => {
      try {
        const result = await chatService.getSessionMessages(sessionId)
        if (result.success && result.data) {
          clearMessages()

          result.data.messages.forEach((msg) => {
            addMessage(msg.question, true)
            addMessage(msg.answer, false, msg.timestamp, msg.source_documents)
          })

          setCurrentSessionId(sessionId)
          setActiveSessionId(sessionId)
          currentConversationId.current = generateUUID()
        }
      } catch (error) {
        console.error("Error loading session:", error)
      }
    },
    [addMessage, clearMessages],
  )

  const startNewChat = useCallback(async () => {
    if (!userId) return

    const newSessionId = generateUUID()
    const newConversationId = generateUUID()

    setCurrentSessionId(newSessionId)
    setActiveSessionId(newSessionId)
    currentConversationId.current = newConversationId

    clearMessages()
    addMessage("Hi, I am the DoT chatbot. How can I help you?", false)

    // Reload chat history to show new session
    await loadChatHistory()
  }, [userId, clearMessages, addMessage, loadChatHistory])

  const sendMessage = useCallback(
    async (content: string, fileNames: string[] = []) => {
      if (!userId || !content.trim()) return

      // Ensure we have session and conversation IDs
      if (!currentSessionId) {
        // Check if user has existing sessions
        const result = await chatService.getUserSessions()
        if (result.success && result.data && result.data.length > 0) {
          setCurrentSessionId(result.data[0].session_id)
          currentConversationId.current = generateUUID()
        } else {
          // Create new session for new user
          const newSessionId = generateUUID()
          setCurrentSessionId(newSessionId)
          currentConversationId.current = generateUUID()
        }
      }

      if (!currentConversationId.current) {
        currentConversationId.current = generateUUID()
      }

      // Add user message
      addMessage(content, true)
      setLoading(true)

      try {
        const request: ChatRequest = {
          question: content,
          user_id: userId,
          conversation_id: currentConversationId.current,
          session_id: currentSessionId!,
          file_names: fileNames,
        }

        const result = await chatService.sendMessage(request)

        if (result.success && result.data) {
          addMessage(result.data.answer, false, result.data.timestamp, result.data.source_documents)

          // Update chat history if this was the first message in a new session
          if (!activeSessionId) {
            await loadChatHistory()
            setActiveSessionId(currentSessionId)
          }
        } else {
          addMessage(result.error || "An error occurred while processing your request.", false)
        }
      } catch (error) {
        addMessage("Network error. Please try again.", false)
        console.error("Chat error:", error)
      } finally {
        setLoading(false)
      }
    },
    [userId, currentSessionId, activeSessionId, addMessage, loadChatHistory],
  )

  const deleteSession = useCallback(
    async (sessionId: string) => {
      try {
        const result = await chatService.deleteSession(sessionId)
        if (result.success) {
          await loadChatHistory()

          // Clear messages if we deleted the active session
          if (activeSessionId === sessionId) {
            clearMessages()
            setActiveSessionId(null)
            setCurrentSessionId(null)
            currentConversationId.current = null
          }
        }
      } catch (error) {
        console.error("Error deleting session:", error)
      }
    },
    [activeSessionId, clearMessages, loadChatHistory],
  )

  return {
    messages,
    sessions,
    currentSessionId,
    activeSessionId,
    loading,
    sendMessage,
    loadSession,
    startNewChat,
    deleteSession,
    loadChatHistory,
    clearMessages,
    addMessage,
  }
}
