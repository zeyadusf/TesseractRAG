@echo off
REM ═══════════════════════════════════════════════════════════════════
REM  TesseractRAG — Project Structure Setup Script
REM  Run from: G:\_TesseractRAG\TesseractRAG\
REM  Safe to run multiple times — skips existing files and folders
REM ═══════════════════════════════════════════════════════════════════

echo.
echo  ████████╗███████╗███████╗███████╗███████╗██████╗  █████╗  ██████╗████████╗
echo  ╚══██╔══╝██╔════╝██╔════╝██╔════╝██╔════╝██╔══██╗██╔══██╗██╔════╝╚══██╔══╝
echo     ██║   █████╗  ███████╗███████╗██
███╗  ██████╔╝███████║██║        ██║
echo     ██║   ██╔══╝  ╚════██║╚════██║██╔══╝  ██╔══██╗██╔══██║██║        ██║
echo     ██║   ███████╗███████║███████║███████╗██║  ██║██║  ██║╚██████╗   ██║
echo     ╚═╝   ╚══════╝╚══════╝╚══════╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝   ╚═╝
echo.
echo  Project Structure Setup ^| Phase 0
echo  ═══════════════════════════════════════════════════════════════════
echo.

REM ── Verify we are in the right directory ──────────────────────────
if not exist ".git" (
    echo  [ERROR] No .git folder found.
    echo  [ERROR] Please run this script from the project root.
    echo  [ERROR] Expected: G:\_TesseractRAG\TesseractRAG\
    echo.
    pause
    exit /b 1
)

echo  [OK] Running from correct project root: %CD%
echo.

REM ═══════════════════════════════════════════════════════════════════
REM  HELPER: Create directory only if it does not exist
REM ═══════════════════════════════════════════════════════════════════
REM  CMD's "mkdir" crashes if folder exists.
REM  We use "if not exist" to guard every single mkdir call.

REM ═══════════════════════════════════════════════════════════════════
REM  BACKEND FOLDERS
REM ═══════════════════════════════════════════════════════════════════
echo  [FOLDERS] Creating backend structure...

if not exist "backend"                              mkdir "backend"
if not exist "backend\app"                          mkdir "backend\app"
if not exist "backend\app\api"                      mkdir "backend\app\api"
if not exist "backend\app\api\v1"                   mkdir "backend\app\api\v1"
if not exist "backend\app\core"                     mkdir "backend\app\core"
if not exist "backend\app\core\ingestion"           mkdir "backend\app\core\ingestion"
if not exist "backend\app\core\retrieval"           mkdir "backend\app\core\retrieval"
if not exist "backend\app\core\generation"          mkdir "backend\app\core\generation"
if not exist "backend\app\models"                   mkdir "backend\app\models"
if not exist "backend\app\utils"                    mkdir "backend\app\utils"
if not exist "backend\tests"                        mkdir "backend\tests"
if not exist "backend\tests\unit"                   mkdir "backend\tests\unit"
if not exist "backend\tests\integration"            mkdir "backend\tests\integration"
if not exist "backend\data"                         mkdir "backend\data"
if not exist "backend\data\sessions"                mkdir "backend\data\sessions"

echo  [OK] Backend folders ready
echo.

REM ═══════════════════════════════════════════════════════════════════
REM  FRONTEND FOLDERS
REM ═══════════════════════════════════════════════════════════════════
echo  [FOLDERS] Creating frontend structure...

if not exist "frontend"                             mkdir "frontend"
if not exist "frontend\components"                  mkdir "frontend\components"

echo  [OK] Frontend folders ready
echo.

REM ═══════════════════════════════════════════════════════════════════
REM  GITHUB ACTIONS FOLDER
REM ═══════════════════════════════════════════════════════════════════
echo  [FOLDERS] Creating GitHub Actions structure...

if not exist ".github"                              mkdir ".github"
if not exist ".github\workflows"                    mkdir ".github\workflows"

echo  [OK] GitHub Actions folder ready
echo.

REM ═══════════════════════════════════════════════════════════════════
REM  HELPER MACRO: Create empty file only if it does not exist
REM  Uses: "if not exist filename ( type nul > filename )"
REM  "type nul" creates an empty file — CMD equivalent of "touch"
REM ═══════════════════════════════════════════════════════════════════

REM ═══════════════════════════════════════════════════════════════════
REM  BACKEND __init__.py FILES
REM ═══════════════════════════════════════════════════════════════════
echo  [FILES] Creating __init__.py files...

if not exist "backend\app\__init__.py"                      type nul > "backend\app\__init__.py"
if not exist "backend\app\api\__init__.py"                  type nul > "backend\app\api\__init__.py"
if not exist "backend\app\api\v1\__init__.py"               type nul > "backend\app\api\v1\__init__.py"
if not exist "backend\app\core\__init__.py"                 type nul > "backend\app\core\__init__.py"
if not exist "backend\app\core\ingestion\__init__.py"       type nul > "backend\app\core\ingestion\__init__.py"
if not exist "backend\app\core\retrieval\__init__.py"       type nul > "backend\app\core\retrieval\__init__.py"
if not exist "backend\app\core\generation\__init__.py"      type nul > "backend\app\core\generation\__init__.py"
if not exist "backend\app\models\__init__.py"               type nul > "backend\app\models\__init__.py"
if not exist "backend\app\utils\__init__.py"                type nul > "backend\app\utils\__init__.py"
if not exist "backend\tests\__init__.py"                    type nul > "backend\tests\__init__.py"
if not exist "backend\tests\unit\__init__.py"               type nul > "backend\tests\unit\__init__.py"
if not exist "backend\tests\integration\__init__.py"        type nul > "backend\tests\integration\__init__.py"

echo  [OK] __init__.py files ready
echo.

REM ═══════════════════════════════════════════════════════════════════
REM  .gitkeep FOR EMPTY FOLDERS
REM  Git does not track empty directories.
REM  .gitkeep is a convention to keep empty folders in version control.
REM ═══════════════════════════════════════════════════════════════════
echo  [FILES] Creating .gitkeep files for empty folders...

if not exist "backend\data\sessions\.gitkeep"       type nul > "backend\data\sessions\.gitkeep"
if not exist ".github\workflows\.gitkeep"           type nul > ".github\workflows\.gitkeep"

echo  [OK] .gitkeep files ready
echo.

REM ═══════════════════════════════════════════════════════════════════
REM  BACKEND SOURCE FILES (empty placeholders)
REM  Only created if they do not already exist.
REM  Existing files with content are NEVER overwritten.
REM ═══════════════════════════════════════════════════════════════════
echo  [FILES] Creating backend source file placeholders...

REM app root
if not exist "backend\app\main.py"                          type nul > "backend\app\main.py"
if not exist "backend\app\config.py"                        type nul > "backend\app\config.py"
if not exist "backend\app\dependencies.py"                  type nul > "backend\app\dependencies.py"

REM api/v1
if not exist "backend\app\api\v1\sessions.py"               type nul > "backend\app\api\v1\sessions.py"
if not exist "backend\app\api\v1\documents.py"              type nul > "backend\app\api\v1\documents.py"
if not exist "backend\app\api\v1\chat.py"                   type nul > "backend\app\api\v1\chat.py"

REM core
if not exist "backend\app\core\session_manager.py"          type nul > "backend\app\core\session_manager.py"

REM ingestion
if not exist "backend\app\core\ingestion\parser.py"         type nul > "backend\app\core\ingestion\parser.py"
if not exist "backend\app\core\ingestion\chunker.py"        type nul > "backend\app\core\ingestion\chunker.py"
if not exist "backend\app\core\ingestion\embedder.py"       type nul > "backend\app\core\ingestion\embedder.py"
if not exist "backend\app\core\ingestion\indexer.py"        type nul > "backend\app\core\ingestion\indexer.py"

REM retrieval
if not exist "backend\app\core\retrieval\bm25_retriever.py" type nul > "backend\app\core\retrieval\bm25_retriever.py"
if not exist "backend\app\core\retrieval\vector_retriever.py" type nul > "backend\app\core\retrieval\vector_retriever.py"
if not exist "backend\app\core\retrieval\hybrid_retriever.py" type nul > "backend\app\core\retrieval\hybrid_retriever.py"
if not exist "backend\app\core\retrieval\reranker.py"       type nul > "backend\app\core\retrieval\reranker.py"
if not exist "backend\app\core\retrieval\router.py"         type nul > "backend\app\core\retrieval\router.py"

REM generation
if not exist "backend\app\core\generation\context_builder.py" type nul > "backend\app\core\generation\context_builder.py"
if not exist "backend\app\core\generation\prompt_builder.py"  type nul > "backend\app\core\generation\prompt_builder.py"
if not exist "backend\app\core\generation\llm_client.py"      type nul > "backend\app\core\generation\llm_client.py"

REM models
if not exist "backend\app\models\session.py"                type nul > "backend\app\models\session.py"
if not exist "backend\app\models\document.py"               type nul > "backend\app\models\document.py"
if not exist "backend\app\models\chat.py"                   type nul > "backend\app\models\chat.py"

REM utils
if not exist "backend\app\utils\logger.py"                  type nul > "backend\app\utils\logger.py"
if not exist "backend\app\utils\text_utils.py"              type nul > "backend\app\utils\text_utils.py"

REM tests
if not exist "backend\tests\unit\test_chunker.py"           type nul > "backend\tests\unit\test_chunker.py"
if not exist "backend\tests\unit\test_retrieval.py"         type nul > "backend\tests\unit\test_retrieval.py"
if not exist "backend\tests\unit\test_context_builder.py"   type nul > "backend\tests\unit\test_context_builder.py"
if not exist "backend\tests\integration\test_sessions.py"   type nul > "backend\tests\integration\test_sessions.py"
if not exist "backend\tests\integration\test_documents.py"  type nul > "backend\tests\integration\test_documents.py"
if not exist "backend\tests\integration\test_chat.py"       type nul > "backend\tests\integration\test_chat.py"

echo  [OK] Backend source files ready
echo.

REM ═══════════════════════════════════════════════════════════════════
REM  FRONTEND SOURCE FILES
REM ═══════════════════════════════════════════════════════════════════
echo  [FILES] Creating frontend source file placeholders...

if not exist "frontend\app.py"                              type nul > "frontend\app.py"
if not exist "frontend\api_client.py"                       type nul > "frontend\api_client.py"
if not exist "frontend\components\__init__.py"              type nul > "frontend\components\__init__.py"
if not exist "frontend\components\sidebar.py"               type nul > "frontend\components\sidebar.py"
if not exist "frontend\components\chat_window.py"           type nul > "frontend\components\chat_window.py"
if not exist "frontend\components\document_panel.py"        type nul > "frontend\components\document_panel.py"
if not exist "frontend\requirements.txt"                    type nul > "frontend\requirements.txt"

echo  [OK] Frontend source files ready
echo.

REM ═══════════════════════════════════════════════════════════════════
REM  ROOT CONFIG FILES
REM ═══════════════════════════════════════════════════════════════════
echo  [FILES] Creating root config files...

if not exist "docker-compose.yml"                           type nul > "docker-compose.yml"
if not exist "backend\Dockerfile"                           type nul > "backend\Dockerfile"
if not exist "frontend\Dockerfile"                          type nul > "frontend\Dockerfile"
if not exist "backend\requirements.txt"                     type nul > "backend\requirements.txt"
if not exist "backend\.env.example"                         type nul > "backend\.env.example"
if not exist "backend\checkpoint.py"                        type nul > "backend\checkpoint.py"

echo  [OK] Root config files ready
echo.

REM ═══════════════════════════════════════════════════════════════════
REM  UPDATE .gitignore
REM  Only writes if .gitignore does not already contain our marker
REM ═══════════════════════════════════════════════════════════════════
echo  [FILES] Checking .gitignore...

findstr /C:"# TesseractRAG" .gitignore >nul 2>&1
if errorlevel 1 (
    echo. >> .gitignore
    echo # TesseractRAG >> .gitignore
    echo # Virtual environment >> .gitignore
    echo .venv/ >> .gitignore
    echo tessRAG/ >> .gitignore
    echo # Secrets — NEVER commit >> .gitignore
    echo backend/.env >> .gitignore
    echo # Python cache >> .gitignore
    echo __pycache__/ >> .gitignore
    echo *.pyc >> .gitignore
    echo *.pyo >> .gitignore
    echo # Runtime data >> .gitignore
    echo backend/data/sessions/*/ >> .gitignore
    echo !backend/data/sessions/.gitkeep >> .gitignore
    echo # ML model cache >> .gitignore
    echo .cache/ >> .gitignore
    echo # VS Code >> .gitignore
    echo .vscode/* >> .gitignore
    echo !.vscode/settings.json >> .gitignore
    echo  [OK] .gitignore updated with TesseractRAG rules
) else (
    echo  [SKIP] .gitignore already has TesseractRAG rules
)
echo.

REM ═══════════════════════════════════════════════════════════════════
REM  VERIFICATION — Print the full tree
REM ═══════════════════════════════════════════════════════════════════
echo  ═══════════════════════════════════════════════════════════════════
echo  VERIFICATION — Project Structure
echo  ═══════════════════════════════════════════════════════════════════
echo.
tree /F /A 2>nul || dir /s /b
echo.

REM ═══════════════════════════════════════════════════════════════════
REM  COUNT SUMMARY
REM ═══════════════════════════════════════════════════════════════════
echo  ═══════════════════════════════════════════════════════════════════
echo  SUMMARY
echo  ═══════════════════════════════════════════════════════════════════

REM Count Python files
set pycount=0
for /r %%f in (*.py) do set /a pycount+=1
echo  Python files created : %pycount%

REM Count directories
set dircount=0
for /d /r %%d in (*) do set /a dircount+=1
echo  Directories created  : %dircount%

echo.
echo  ═══════════════════════════════════════════════════════════════════
echo  [DONE] TesseractRAG structure is ready
echo  ═══════════════════════════════════════════════════════════════════
echo.
echo  NEXT STEPS:
echo  1. cd backend
echo  2. ..\tessRAG\Scripts\activate.bat
echo  3. Open each file in VS Code and add its content
echo  4. python checkpoint.py   ^<-- verify Phase 0
echo.
pause