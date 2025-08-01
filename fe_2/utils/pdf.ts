// PDF handling utilities
import { fileService } from "@/services/api"
import type { SourceDocument } from "@/types"

/**
 * Opens a highlighted PDF in a new window
 */
export async function openHighlightedPDF(document: SourceDocument): Promise<boolean> {
    try {
        console.log('Opening highlighted PDF for:', document)

        const result = await fileService.viewHighlights(document)

        if (result) {
            const { blob, pageNumber } = result
            const url = window.URL.createObjectURL(blob)

            // Add page number to URL if available
            let finalUrl = url
            if (pageNumber) {
                finalUrl = url + '#page=' + pageNumber
                console.log('Final URL with page number:', finalUrl)
            }

            const newWindow = window.open(finalUrl, "_blank")

            if (!newWindow) {
                alert("Please allow pop-ups to view the highlighted PDF")
                return false
            }

            // Clean up the blob URL after a delay
            setTimeout(() => {
                window.URL.revokeObjectURL(url)
            }, 1000)

            return true
        } else {
            throw new Error("Failed to load highlighted PDF")
        }
    } catch (error) {
        console.error("Error opening highlighted PDF:", error)
        throw error
    }
}

/**
 * Opens a regular PDF in a new window
 */
export async function openPDF(blobName: string): Promise<boolean> {
    try {
        const blob = await fileService.viewPDF(blobName)

        if (blob) {
            const url = window.URL.createObjectURL(blob)
            const newWindow = window.open(url, "_blank")

            if (!newWindow) {
                alert("Please allow pop-ups to view the PDF")
                return false
            }

            // Clean up the blob URL after a delay
            setTimeout(() => {
                window.URL.revokeObjectURL(url)
            }, 1000)

            return true
        } else {
            throw new Error("Failed to load PDF")
        }
    } catch (error) {
        console.error("Error opening PDF:", error)
        throw error
    }
}

/**
 * Downloads a PDF blob as a file
 */
export function downloadPDFBlob(blob: Blob, filename: string): void {
    const url = window.URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = filename
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    window.URL.revokeObjectURL(url)
}

/**
 * Gets the filename from a document source
 */
export function getDocumentFilename(document: SourceDocument): string {
    return document.filename.split('/').pop() || document.filename
}