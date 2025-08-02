"use client"

// File selection modal component following Single Responsibility Principle

import { useState, useEffect } from "react"
import { FileText, X } from "lucide-react"
import { Modal } from "@/components/common/Modal"
import { Button } from "@/components/common/Button"
import { LoadingSpinner } from "@/components/common/LoadingSpinner"
import { getFilenameOnly } from "@/utils"
import { cn } from "@/lib/utils"
import type { FileItem } from "@/types"

interface FileSelectionModalProps {
  isOpen: boolean
  onClose: () => void
  availableFiles: FileItem[]
  selectedFile: string | null
  loading: boolean
  onLoadFiles: () => Promise<void>
  onSelectFile: (filename: string) => void
  onClearSelection: () => void
  onConfirm: () => void
}

export function FileSelectionModal({
  isOpen,
  onClose,
  availableFiles,
  selectedFile,
  loading,
  onLoadFiles,
  onSelectFile,
  onClearSelection,
  onConfirm,
}: FileSelectionModalProps) {
  const [localSelectedFile, setLocalSelectedFile] = useState<string | null>(selectedFile)

  useEffect(() => {
    if (isOpen) {
      onLoadFiles()
      setLocalSelectedFile(selectedFile)
    }
  }, [isOpen, onLoadFiles, selectedFile])

  const handleFileSelect = (filename: string) => {
    setLocalSelectedFile(filename)
  }

  const handleClearSelection = () => {
    setLocalSelectedFile(null)
  }

  const handleConfirm = () => {
    if (localSelectedFile) {
      onSelectFile(localSelectedFile)
    } else {
      onClearSelection()
    }
    onConfirm()
  }

  const handleCancel = () => {
    setLocalSelectedFile(selectedFile)
    onClose()
  }

  return (
    <Modal isOpen={isOpen} onClose={handleCancel} title="Select a File to Chat With" size="md">
      <div className="space-y-4">
        {/* Current Selection Display */}
        {selectedFile && (
          <div className="flex items-center justify-between p-3 bg-muted/50 rounded-lg">
            <div className="flex items-center gap-2">
              <FileText className="h-4 w-4 text-muted-foreground" />
              <span className="text-sm font-medium">Currently selected: {getFilenameOnly(selectedFile)}</span>
            </div>
            <Button
              variant="ghost"
              size="sm"
              onClick={handleClearSelection}
              className="h-6 w-6 p-0 text-muted-foreground hover:text-foreground"
              title="Clear selection"
            >
              <X className="h-3 w-3" />
            </Button>
          </div>
        )}

        {/* File List */}
        <div className="max-h-80 overflow-y-auto space-y-2">
          {loading ? (
            <div className="flex justify-center py-8">
              <LoadingSpinner text="Loading files..." />
            </div>
          ) : availableFiles.length === 0 ? (
            <div className="text-center py-8 text-muted-foreground">
              <FileText className="mx-auto h-12 w-12 mb-2 opacity-50" />
              <p>No files available</p>
            </div>
          ) : (
            availableFiles.map((file) => (
              <div
                key={file.value}
                className={cn(
                  "flex items-center gap-3 p-3 rounded-lg border cursor-pointer transition-colors",
                  "hover:bg-accent/50",
                  localSelectedFile === file.value && "bg-accent border-primary",
                )}
                onClick={() => handleFileSelect(file.value)}
              >
                <input
                  type="radio"
                  name="fileSelection"
                  checked={localSelectedFile === file.value}
                  onChange={() => handleFileSelect(file.value)}
                  className="text-primary focus:ring-primary"
                />

                <FileText className="h-4 w-4 text-muted-foreground flex-shrink-0" />

                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium truncate">{getFilenameOnly(file.value)}</p>
                  <p className="text-xs text-muted-foreground truncate">
                    {file.value} • {file.count} chunks
                  </p>
                </div>
              </div>
            ))
          )}
        </div>

        {/* Actions */}
        <div className="flex gap-2 justify-end">
          <Button variant="secondary" onClick={handleCancel} disabled={loading}>
            Cancel
          </Button>
          <Button onClick={handleConfirm} disabled={loading}>
            {localSelectedFile ? "Confirm File" : "Clear Selection"}
          </Button>
        </div>
      </div>
    </Modal>
  )
}
