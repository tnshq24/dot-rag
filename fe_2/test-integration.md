# Integration Testing Guide

## Prerequisites

1. Python backend running on port 5001
2. All required Python dependencies installed
3. Environment variables configured

## Testing Steps

### 1. Start the Backend
```bash
cd DOT_RAG/frontend
python main.py
```

### 2. Start the Frontend
```bash
cd fe_2
pnpm install
pnpm dev
```

### 3. Test Authentication
- Visit http://localhost:3000
- Click on user section to open login modal
- Test admin login: admin@xyz.com / admin
- Test user login: user1@xyz.com / user1

### 4. Test Chat Functionality
- Send a test message
- Verify response from RAG pipeline
- Check session creation in sidebar

### 5. Test File Operations
- Upload a PDF (admin only)
- Select files for contextual chat
- Test PDF highlighting by clicking references

### 6. Test Session Management
- Create multiple chat sessions
- Switch between sessions
- Delete sessions

## Expected Behavior

✅ Login/logout works correctly
✅ Chat messages send and receive properly
✅ PDF references open with highlights
✅ File upload works (admin only)
✅ Session management functions properly
✅ UI is responsive and functional