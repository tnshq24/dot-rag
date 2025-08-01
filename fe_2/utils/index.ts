/**
 * Generates a UUID v4 string
 */
export function generateUUID(): string {
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0
    const v = c == "x" ? r : (r & 0x3) | 0x8
    return v.toString(16)
  })
}

/**
 * Extracts filename from full path
 */
export function getFilenameOnly(fullPath: string): string {
  const parts = fullPath.split("/")
  return parts[parts.length - 1]
}

/**
 * Formats timestamp to IST timezone
 */
export function formatISTDateTime(ts: string | null): string {
  if (!ts) return ""

  const utc = new Date(ts)
  // IST is UTC+5:30
  const istOffset = 5.5 * 60 // in minutes
  const ist = new Date(utc.getTime() + istOffset * 60000)

  const yyyy = ist.getFullYear()
  const mm = String(ist.getMonth() + 1).padStart(2, "0")
  const dd = String(ist.getDate()).padStart(2, "0")
  const hh = String(ist.getHours()).padStart(2, "0")
  const min = String(ist.getMinutes()).padStart(2, "0")
  const ss = String(ist.getSeconds()).padStart(2, "0")

  return `${yyyy}-${mm}-${dd} ${hh}:${min}:${ss}`
}

/**
 * Validates email format
 */
export function isValidEmail(email: string): boolean {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/
  return emailRegex.test(email)
}

/**
 * Validates file type
 */
export function isValidFileType(file: File, acceptedTypes: string): boolean {
  const fileExtension = "." + file.name.split(".").pop()?.toLowerCase()
  return acceptedTypes.includes(fileExtension)
}

/**
 * Formats file size to human readable format
 */
export function formatFileSize(bytes: number): string {
  if (bytes === 0) return "0 Bytes"

  const k = 1024
  const sizes = ["Bytes", "KB", "MB", "GB"]
  const i = Math.floor(Math.log(bytes) / Math.log(k))

  return Number.parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i]
}

/**
 * Debounce function for performance optimization
 */
export function debounce<T extends (...args: any[]) => any>(func: T, wait: number): (...args: Parameters<T>) => void {
  let timeout: NodeJS.Timeout

  return (...args: Parameters<T>) => {
    clearTimeout(timeout)
    timeout = setTimeout(() => func.apply(null, args), wait)
  }
}

/**
 * Throttle function for performance optimization
 */
export function throttle<T extends (...args: any[]) => any>(func: T, limit: number): (...args: Parameters<T>) => void {
  let inThrottle: boolean

  return (...args: Parameters<T>) => {
    if (!inThrottle) {
      func.apply(null, args)
      inThrottle = true
      setTimeout(() => (inThrottle = false), limit)
    }
  }
}

/**
 * Clamps a value between min and max
 */
export function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max)
}

/**
 * Checks if content contains "no information" patterns
 */
export function hasNoInformationPattern(content: string): boolean {
  const noInfoPatterns = [
    "The context documents provided do not contain any information",
    "The context documents provided do not include any information",
    "no information found",
    "cannot be found in the context",
    "no relevant information",
  ]

  return noInfoPatterns.some((pattern) => content.toLowerCase().includes(pattern.toLowerCase()))
}
