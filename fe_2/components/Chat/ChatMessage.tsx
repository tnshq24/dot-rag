"use client"
import { marked } from "marked"
import { ExternalLink } from "lucide-react"
import { cn } from "@/lib/utils"
import { formatISTDateTime, getFilenameOnly, hasNoInformationPattern } from "@/utils"
import { openHighlightedPDF } from "@/utils/pdf"
import type { ChatMessage as ChatMessageType } from "@/types"

interface ChatMessageProps {
  message: ChatMessageType
}

export function ChatMessage({ message }: ChatMessageProps) {
  const { content, isUser, timestamp, sourceDocuments } = message

  const handleDocumentClick = async (doc: any) => {
    try {
      await openHighlightedPDF(doc)
    } catch (error) {
      console.error("Error opening document:", error)
      alert("Error loading highlighted PDF: " + (error as Error).message)
    }
  }

  return (
    <div className={cn("flex gap-4 mb-6", isUser ? "flex-row-reverse justify-end" : "flex-row justify-start")}>
      {/* Avatar */}
      {!isUser && (
        <div className="flex h-10 w-10 items-center justify-center">
          <svg width="32" height="32" viewBox="0 0 512 512" className="text-primary" fill="currentColor">
            <path d="M2487 4900 c-177 -31 -320 -179 -348 -363 -28 -177 73 -360 241 -439 l70 -33 0 -217 0 -217 -687 -3 c-749 -4 -709 -1 -834 -62 -114 -56 -208 -164 -258 -296 -23 -60 -25 -81 -29 -282 l-4 -217 -206 -3 c-193 -3 -210 -5 -250 -26 -66 -35 -119 -88 -149 -150 l-28 -57 0 -560 0 -560 24 -53 c29 -65 113 -143 178 -168 38 -14 89 -18 240 -22 l193 -4 0 -186 c0 -277 31 -372 160 -503 71 -72 173 -128 268 -148 37 -8 487 -11 1495 -11 1584 0 1503 -3 1629 61 90 46 186 145 230 239 50 105 58 155 58 365 l0 183 193 4 c151 4 202 8 240 22 65 25 149 103 178 168 l24 53 0 560 0 560 -28 57 c-30 62 -83 115 -149 150 -40 21 -57 23 -250 26 l-206 3 -4 217 c-4 201 -6 222 -29 282 -62 164 -173 271 -344 333 -59 22 -70 22 -747 25 l-688 3 0 217 0 217 70 33 c126 59 215 175 241 316 20 109 -19 248 -96 342 -88 108 -257 169 -398 144z" />
          </svg>
        </div>
      )}

      {/* Message Content */}
      <div
        className={cn(
          "max-w-[70%] rounded-2xl px-6 py-4 text-sm leading-relaxed",
          isUser
            ? "bg-muted text-foreground rounded-br-sm ml-auto mr-8"
            : "bg-background text-foreground rounded-bl-sm mr-auto ml-8 border",
        )}
      >
        {/* Message text */}
        <div className="prose prose-sm max-w-none dark:prose-invert">
          {isUser ? (
            <p className="m-0">{content}</p>
          ) : (
            <div
              dangerouslySetInnerHTML={{
                __html: marked.parse(content),
              }}
            />
          )}
        </div>

        {/* Source documents */}
        {!isUser && sourceDocuments && sourceDocuments.length > 0 && !hasNoInformationPattern(content) && (
          <div className="mt-4 pt-3 border-t border-border">
            <p className="text-xs font-medium text-muted-foreground mb-2">References:</p>
            <div className="flex flex-wrap gap-2">
              {sourceDocuments.map((doc, idx) => (
                <button
                  key={idx}
                  onClick={() => handleDocumentClick(doc)}
                  className="inline-flex items-center gap-1 px-2 py-1 text-xs bg-primary/10 text-primary rounded hover:bg-primary/20 transition-colors"
                >
                  <ExternalLink className="h-3 w-3" />
                  {getFilenameOnly(doc.filename)}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Timestamp */}
        {timestamp && <div className="mt-2 text-xs text-muted-foreground">{formatISTDateTime(timestamp)}</div>}
      </div>
    </div>
  )
}
