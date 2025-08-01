"use client"

// Upload modal component following Single Responsibility Principle

import type React from "react"
import { useState } from "react"
import { Upload } from "lucide-react"
import { Modal } from "@/components/common/Modal"
import { Button } from "@/components/common/Button"
import { Input } from "@/components/common/Input"
import { fileService } from "@/services/api"
import { FILE_CONSTRAINTS, MESSAGES } from "@/constants"
import { isValidFileType, formatFileSize } from "@/utils"
import type { UploadFormData } from "@/types"

interface UploadModalProps {
  isOpen: boolean
  onClose: () => void
  onUploadSuccess: () => void
}

export function UploadModal({ isOpen, onClose, onUploadSuccess }: UploadModalProps) {
  const [files, setFiles] = useState<FileList | null>(null)
  const [formData, setFormData] = useState<UploadFormData>({
    filename: "",
    projectCode: "",
    labelTag: "",
  })
  const [loading, setLoading] = useState(false)
  const [progress, setProgress] = useState(0)
  const [status, setStatus] = useState("")
  const [error, setError] = useState("")

  const validateForm = () => {
    return (
      files &&
      files.length > 0 &&
      files.length <= FILE_CONSTRAINTS.MAX_FILES &&
      formData.filename.trim() &&
      formData.projectCode.trim()
    )
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFiles = e.target.files
    setError("")

    if (!selectedFiles || selectedFiles.length === 0) {
      setFiles(null)
      return
    }

    // Validate file count
    if (selectedFiles.length > FILE_CONSTRAINTS.MAX_FILES) {
      setError(`Maximum ${FILE_CONSTRAINTS.MAX_FILES} files allowed`)
      return
    }

    // Validate file types and sizes
    for (let i = 0; i < selectedFiles.length; i++) {
      const file = selectedFiles[i]

      if (!isValidFileType(file, FILE_CONSTRAINTS.ACCEPTED_TYPES)) {
        setError("Only PDF files are allowed")
        return
      }

      if (file.size > FILE_CONSTRAINTS.MAX_FILE_SIZE) {
        setError(`File "${file.name}" is too large. Maximum size is ${formatFileSize(FILE_CONSTRAINTS.MAX_FILE_SIZE)}`)
        return
      }
    }

    setFiles(selectedFiles)
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()

    if (!validateForm() || !files) return

    setLoading(true)
    setProgress(30)
    setStatus("Uploading...")
    setError("")

    try {
      // Simulate progress updates
      setTimeout(() => {
        setProgress(60)
        setStatus("Indexing your document, please wait...")
      }, 1000)

      setTimeout(() => {
        setProgress(90)
      }, 2000)

      const result = await fileService.uploadPDF(files, formData)
      setProgress(100)

      if (result.success) {
        setStatus(MESSAGES.UPLOAD_SUCCESS)
        onUploadSuccess()
        setTimeout(() => {
          handleClose()
        }, 2000)
      } else {
        setError(result.error || "Upload failed")
        setProgress(0)
        setStatus("")
      }
    } catch (error) {
      setError("Network error. Please try again.")
      setProgress(0)
      setStatus("")
    } finally {
      setLoading(false)
    }
  }

  const handleClose = () => {
    onClose()
    setFiles(null)
    setFormData({ filename: "", projectCode: "", labelTag: "" })
    setProgress(0)
    setStatus("")
    setError("")
  }

  return (
    <Modal isOpen={isOpen} onClose={handleClose} title="Upload PDF Documents" size="md">
      <form onSubmit={handleSubmit} className="space-y-4">
        {/* File Input */}
        <div className="space-y-2">
          <label className="text-sm font-medium">Select PDF files to upload</label>
          <div className="relative">
            <input
              type="file"
              accept={FILE_CONSTRAINTS.ACCEPTED_TYPES}
              multiple
              onChange={handleFileChange}
              className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
              disabled={loading}
            />
            <div className="flex items-center justify-center w-full h-32 border-2 border-dashed border-muted-foreground/25 rounded-lg bg-muted/50 hover:bg-muted/70 transition-colors">
              <div className="text-center">
                <Upload className="mx-auto h-8 w-8 text-muted-foreground mb-2" />
                <p className="text-sm text-muted-foreground">
                  {files && files.length > 0 ? `${files.length} file(s) selected` : "Click to select PDF files"}
                </p>
                <p className="text-xs text-muted-foreground mt-1">
                  Maximum {FILE_CONSTRAINTS.MAX_FILES} files, {formatFileSize(FILE_CONSTRAINTS.MAX_FILE_SIZE)} each
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Form Fields */}
        <Input
          label="Filename *"
          value={formData.filename}
          onChange={(e) => setFormData((prev) => ({ ...prev, filename: e.target.value }))}
          required
          disabled={loading}
        />

        <Input
          label="Categories *"
          value={formData.projectCode}
          onChange={(e) => setFormData((prev) => ({ ...prev, projectCode: e.target.value }))}
          required
          disabled={loading}
        />

        <Input
          label="Label/Tag"
          value={formData.labelTag}
          onChange={(e) => setFormData((prev) => ({ ...prev, labelTag: e.target.value }))}
          disabled={loading}
        />

        {/* Progress Bar */}
        {loading && (
          <div className="space-y-2">
            <div className="w-full bg-muted rounded-full h-2">
              <div
                className="bg-primary h-2 rounded-full transition-all duration-300"
                style={{ width: `${progress}%` }}
              />
            </div>
            {status && <p className="text-sm text-center text-muted-foreground">{status}</p>}
          </div>
        )}

        {/* Error Message */}
        {error && (
          <p className="text-sm text-red-600" role="alert">
            {error}
          </p>
        )}

        {/* Submit Button */}
        <Button type="submit" className="w-full" disabled={!validateForm() || loading} loading={loading}>
          Upload
        </Button>
      </form>
    </Modal>
  )
}
