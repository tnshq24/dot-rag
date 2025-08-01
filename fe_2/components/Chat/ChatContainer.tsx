import { ChatMessages } from "./ChatMessages"
import { ChatInput } from "./ChatInput"
import { MESSAGES } from "@/constants"
import type { ChatMessage, User } from "@/types"

interface ChatContainerProps {
  messages: ChatMessage[]
  loading: boolean
  user: User | null
  isAuthenticated: boolean
  isAdmin: boolean
  selectedFile: string | null
  onSendMessage: (message: string, fileNames?: string[]) => Promise<void>
  onSelectFiles: () => void
  onUploadFiles: () => void
  onRemoveFile: () => void
}

export function ChatContainer({
  messages,
  loading,
  user,
  isAuthenticated,
  isAdmin,
  selectedFile,
  onSendMessage,
  onSelectFiles,
  onUploadFiles,
  onRemoveFile,
}: ChatContainerProps) {
  // Show welcome message if no messages
  const displayMessages =
    messages.length === 0
      ? [
          {
            id: "welcome",
            content: MESSAGES.WELCOME,
            isUser: false,
          },
        ]
      : messages

  return (
    <div className="flex flex-1 flex-col">
      {/* Header */}
      <div className="border-b bg-background p-4">
        <div className="flex items-center justify-center">
          <img src="/logo.png" alt="DoT Logo" className="h-10" />
        </div>
      </div>

      {/* Messages */}
      <ChatMessages messages={displayMessages} loading={loading} />

      {/* Input */}
      <ChatInput
        onSendMessage={onSendMessage}
        selectedFile={selectedFile}
        onSelectFiles={onSelectFiles}
        onUploadFiles={onUploadFiles}
        onRemoveFile={onRemoveFile}
        loading={loading}
        isAuthenticated={isAuthenticated}
        isAdmin={isAdmin}
      />
    </div>
  )
}
