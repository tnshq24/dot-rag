// Main application page following SOLID principles

"use client"

import { useEffect, useState } from "react"
import { Sidebar } from "@/components/Sidebar/Sidebar"
import { ChatContainer } from "@/components/Chat/ChatContainer"
import { LoginModal } from "@/components/Modals/LoginModal"
import { UploadModal } from "@/components/Modals/UploadModal"
import { FileSelectionModal } from "@/components/Modals/FileSelectionModal"
import { useAuth } from "@/hooks/useAuth"
import { useChat } from "@/hooks/useChat"
import { useFileSelection } from "@/hooks/useFileSelection"
import { MESSAGES } from "@/constants"

export default function ChatbotPage() {
  // Authentication state
  const { user, isAuthenticated, isAdmin, login, logout, loading: authLoading } = useAuth()

  // Chat state
  const {
    messages,
    sessions,
    currentSessionId,
    activeSessionId,
    loading: chatLoading,
    sendMessage,
    loadSession,
    startNewChat,
    deleteSession,
    loadChatHistory,
    clearMessages,
    addMessage,
  } = useChat(user?.user_id)

  // File selection state
  const {
    availableFiles,
    selectedFile,
    loading: filesLoading,
    loadAvailableFiles,
    selectFile,
    clearSelection,
  } = useFileSelection()

  // Modal states
  const [showLoginModal, setShowLoginModal] = useState(false)
  const [showUploadModal, setShowUploadModal] = useState(false)
  const [showFileSelectionModal, setShowFileSelectionModal] = useState(false)

  // Load chat history when user authenticates
  useEffect(() => {
    if (isAuthenticated && user) {
      loadChatHistory()
    }
  }, [isAuthenticated, user, loadChatHistory])

  // Initialize welcome message for unauthenticated users
  useEffect(() => {
    if (!isAuthenticated) {
      clearMessages()
      addMessage(MESSAGES.WELCOME, false)
    }
  }, [isAuthenticated, clearMessages, addMessage])

  // Event handlers
  const handleLogin = async (credentials: { email: string; password: string }) => {
    const result = await login(credentials)
    if (result.success) {
      setShowLoginModal(false)
    }
    return result
  }

  const handleLogout = async () => {
    await logout()
    clearSelection()
    clearMessages()
    addMessage(MESSAGES.WELCOME, false)
  }

  const handleSendMessage = async (message: string, fileNames?: string[]) => {
    if (!isAuthenticated) {
      setShowLoginModal(true)
      return
    }

    await sendMessage(message, fileNames)
  }

  const handleNewChat = async () => {
    if (!isAuthenticated) {
      setShowLoginModal(true)
      return
    }

    clearSelection()
    await startNewChat()
  }

  const handleDeleteAllSessions = async () => {
    if (!isAuthenticated) return

    // Delete all sessions
    for (const session of sessions) {
      await deleteSession(session.session_id)
    }

    clearMessages()
    clearSelection()
  }

  const handleSelectFiles = () => {
    if (!isAuthenticated) {
      setShowLoginModal(true)
      return
    }

    setShowFileSelectionModal(true)
  }

  const handleUploadFiles = () => {
    if (!isAuthenticated) {
      setShowLoginModal(true)
      return
    }

    setShowUploadModal(true)
  }

  const handleFileSelectionConfirm = () => {
    setShowFileSelectionModal(false)
  }

  const handleUploadSuccess = () => {
    // Refresh available files after successful upload
    loadAvailableFiles()
  }

  if (authLoading) {
    return (
      <div className="flex h-screen items-center justify-center">
        <div className="text-center">
          <div className="h-8 w-8 animate-spin rounded-full border-2 border-primary border-t-transparent mx-auto mb-4" />
          <p className="text-muted-foreground">Loading...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="flex h-screen bg-background">
      {/* Sidebar */}
      <Sidebar
        sessions={sessions}
        activeSessionId={activeSessionId}
        user={user}
        isAuthenticated={isAuthenticated}
        onNewChat={handleNewChat}
        onSessionSelect={loadSession}
        onSessionDelete={deleteSession}
        onDeleteAll={handleDeleteAllSessions}
        onLogin={() => setShowLoginModal(true)}
        onLogout={handleLogout}
      />

      {/* Main Chat Area */}
      <ChatContainer
        messages={messages}
        loading={chatLoading}
        user={user}
        isAuthenticated={isAuthenticated}
        isAdmin={isAdmin}
        selectedFile={selectedFile}
        onSendMessage={handleSendMessage}
        onSelectFiles={handleSelectFiles}
        onUploadFiles={handleUploadFiles}
        onRemoveFile={clearSelection}
      />

      {/* Modals */}
      <LoginModal isOpen={showLoginModal} onClose={() => setShowLoginModal(false)} onLogin={handleLogin} />

      <UploadModal
        isOpen={showUploadModal}
        onClose={() => setShowUploadModal(false)}
        onUploadSuccess={handleUploadSuccess}
      />

      <FileSelectionModal
        isOpen={showFileSelectionModal}
        onClose={() => setShowFileSelectionModal(false)}
        availableFiles={availableFiles}
        selectedFile={selectedFile}
        loading={filesLoading}
        onLoadFiles={loadAvailableFiles}
        onSelectFile={selectFile}
        onConfirm={handleFileSelectionConfirm}
      />
    </div>
  )
}
