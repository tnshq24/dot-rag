"use client"

// Reusable Textarea component with auto-resize functionality

import React, { useEffect, useRef } from "react"
import { cn } from "@/lib/utils"

interface TextareaProps extends React.TextareaHTMLAttributes<HTMLTextAreaElement> {
  label?: string
  error?: string
  helperText?: string
  autoResize?: boolean
  maxHeight?: number
}

export const Textarea = React.forwardRef<HTMLTextAreaElement, TextareaProps>(
  ({ className, label, error, helperText, autoResize = false, maxHeight = 200, id, ...props }, ref) => {
    const textareaRef = useRef<HTMLTextAreaElement>(null)
    const inputId = id || `textarea-${Math.random().toString(36).substr(2, 9)}`

    useEffect(() => {
      if (autoResize && textareaRef.current) {
        const textarea = textareaRef.current
        const adjustHeight = () => {
          textarea.style.height = "auto"
          textarea.style.height = Math.min(textarea.scrollHeight, maxHeight) + "px"
        }

        textarea.addEventListener("input", adjustHeight)
        adjustHeight() // Initial adjustment

        return () => {
          textarea.removeEventListener("input", adjustHeight)
        }
      }
    }, [autoResize, maxHeight])

    return (
      <div className="space-y-2">
        {label && (
          <label
            htmlFor={inputId}
            className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
          >
            {label}
          </label>
        )}
        <textarea
          id={inputId}
          className={cn(
            "flex min-h-[80px] w-full rounded-md border border-input bg-background px-3 py-2 text-sm",
            "ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none",
            "focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
            "disabled:cursor-not-allowed disabled:opacity-50 resize-none",
            error && "border-red-500 focus-visible:ring-red-500",
            className,
          )}
          ref={(node) => {
            textareaRef.current = node
            if (typeof ref === "function") {
              ref(node)
            } else if (ref) {
              ref.current = node
            }
          }}
          style={{ maxHeight: autoResize ? maxHeight : undefined }}
          {...props}
        />
        {error && (
          <p className="text-sm text-red-600" role="alert">
            {error}
          </p>
        )}
        {helperText && !error && <p className="text-sm text-muted-foreground">{helperText}</p>}
      </div>
    )
  },
)

Textarea.displayName = "Textarea"
