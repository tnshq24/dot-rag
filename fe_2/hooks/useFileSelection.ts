"use client"

// File selection hook following Single Responsibility Principle

import { useState, useCallback } from "react"
import { fileService } from "@/services/api"
import type { FileItem } from "@/types"

interface UseFileSelectionReturn {
  availableFiles: FileItem[]
  selectedFile: string | null
  loading: boolean
  loadAvailableFiles: () => Promise<void>
  selectFile: (filename: string) => void
  clearSelection: () => void
}

export function useFileSelection(): UseFileSelectionReturn {
  const [availableFiles, setAvailableFiles] = useState<FileItem[]>([])
  const [selectedFile, setSelectedFile] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  const loadAvailableFiles = useCallback(async () => {
    setLoading(true)
    try {
      const result = await fileService.getAvailableFiles()
      if (result.success && result.data) {
        setAvailableFiles(result.data.files || [])
      } else {
        console.error("Failed to load available files:", result.error)
        setAvailableFiles([])
      }
    } catch (error) {
      console.error("Error loading available files:", error)
      setAvailableFiles([])
    } finally {
      setLoading(false)
    }
  }, [])

  const selectFile = useCallback((filename: string) => {
    setSelectedFile(filename)
  }, [])

  const clearSelection = useCallback(() => {
    setSelectedFile(null)
  }, [])

  return {
    availableFiles,
    selectedFile,
    loading,
    loadAvailableFiles,
    selectFile,
    clearSelection,
  }
}
