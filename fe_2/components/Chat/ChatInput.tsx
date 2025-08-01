"use client"

// Chat input component following Single Responsibility Principle

import type React from "react"
import { useState, useRef, type KeyboardEvent } from "react"
import { Send, Plus, Paperclip, X } from "lucide-react"
import { Button } from "@/components/common/Button"
import { Textarea } from "@/components/common/Textarea"
import { getFilenameOnly } from "@/utils"

interface ChatInputProps {
  onSendMessage: (message: string, fileNames?: string[]) => Promise<void>
  selectedFile: string | null
  onSelectFiles: () => void
  onUploadFiles: () => void
  onRemoveFile: () => void
  loading: boolean
  isAuthenticated: boolean
  isAdmin: boolean
}

export function ChatInput({
  onSendMessage,
  selectedFile,
  onSelectFiles,
  onUploadFiles,
  onRemoveFile,
  loading,
  isAuthenticated,
  isAdmin,
}: ChatInputProps) {
  const [message, setMessage] = useState("")
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()

    if (!message.trim() || loading) return

    const fileNames = selectedFile ? [selectedFile] : []
    await onSendMessage(message.trim(), fileNames)
    setMessage("")

    // Reset textarea height
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto"
    }
  }

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e as any)
    }
  }

  return (
    <div className="border-t bg-background p-6">
      <form onSubmit={handleSubmit} className="mx-auto max-w-4xl">
        {/* Selected file display */}
        {selectedFile && (
          <div className="mb-3 flex items-center gap-2 rounded-md bg-primary/10 px-3 py-2 text-sm">
            <span className="flex-1 truncate text-primary">Selected: {getFilenameOnly(selectedFile)}</span>
            <Button
              type="button"
              variant="ghost"
              size="sm"
              onClick={onRemoveFile}
              className="h-6 w-6 p-0 text-primary hover:text-primary/80"
            >
              <X className="h-3 w-3" />
            </Button>
          </div>
        )}

        {/* Input area */}
        <div className="flex items-end gap-2">
          {/* File actions */}
          {isAuthenticated && (
            <div className="flex gap-1">
              <Button
                type="button"
                variant="ghost"
                size="sm"
                onClick={onSelectFiles}
                className="h-10 w-10 p-0"
                title="Select Files"
              >
                <Plus className="h-4 w-4" />
              </Button>

              {isAdmin && (
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  onClick={onUploadFiles}
                  className="h-10 w-10 p-0"
                  title="Upload PDF"
                >
                  <Paperclip className="h-4 w-4" />
                </Button>
              )}
            </div>
          )}

          {/* Message input */}
          <div className="relative flex-1">
            <Textarea
              ref={textareaRef}
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask me anything..."
              autoResize
              maxHeight={200}
              className="min-h-[44px] pr-12 resize-none"
              disabled={loading}
            />

            {/* Send button */}
            <Button
              type="submit"
              size="sm"
              disabled={!message.trim() || loading}
              className="absolute bottom-2 right-2 h-8 w-8 p-0"
            >
              <Send className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </form>
    </div>
  )
}
