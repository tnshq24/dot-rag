"use client"
import { LogOut, User } from "lucide-react"
import { Button } from "@/components/common/Button"
import type { User as UserType } from "@/types"

interface UserSectionProps {
  user: UserType | null
  isAuthenticated: boolean
  onLogin: () => void
  onLogout: () => void
}

export function UserSection({ user, isAuthenticated, onLogin, onLogout }: UserSectionProps) {
  if (!isAuthenticated) {
    return (
      <div className="border-t p-4 cursor-pointer hover:bg-accent/50 transition-colors" onClick={onLogin}>
        <div className="flex items-center gap-3">
          <div className="flex h-8 w-8 items-center justify-center rounded-full bg-muted">
            <User className="h-4 w-4" />
          </div>
          <div className="flex-1">
            <p className="text-sm font-medium">Not logged in</p>
            <p className="text-xs text-muted-foreground">Click to login</p>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="border-t p-4">
      <div className="flex items-center gap-3">
        <div className="flex h-8 w-8 items-center justify-center rounded-full bg-primary text-primary-foreground text-sm font-medium">
          {user?.email.charAt(0).toUpperCase()}
        </div>

        <div className="flex-1 min-w-0">
          <p className="text-sm font-medium truncate">{user?.email}</p>
          <p className="text-xs text-muted-foreground">Logged in</p>
        </div>

        <Button variant="ghost" size="sm" onClick={onLogout} className="h-8 w-8 p-0" title="Logout">
          <LogOut className="h-4 w-4" />
        </Button>
      </div>
    </div>
  )
}
