# Use this file to run the n8n-nodes-browser-use on local for running the integration tests for the app in headful mode
from __future__ import annotations
import asyncio
import json
import logging
import os
import uuid
import base64
from typing import Optional
from datetime import datetime, UTC
from enum import Enum
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Response, Query, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from browser_use import Agent
from browser_use import BrowserProfile, Browser
import httpx

from browser_use.llm import (
    ChatAnthropic,
    ChatOpenAI,
    ChatGoogle,
    ChatOllama,
    ChatAzureOpenAI,
    ChatAWSBedrock,
)

from pathlib import Path
# Import our task storage abstraction    
from task_storage import get_task_storage
from task_storage.base import DEFAULT_USER_ID
from browser_use.browser.events import ScreenshotEvent

def save_screenshot_png(base64_str: str, file_path: str):
    image_data = base64.b64decode(base64_str)
    with open(file_path, 'wb') as f:
        f.write(image_data)
    print(f"Screenshot saved at {file_path}")

async def test_capture_screenshot(agent: Agent, task_media_dir: Path, task_id: str):
    screenshot_event = agent.browser_session.event_bus.dispatch(ScreenshotEvent(full_page=True))
    await screenshot_event
    result = await screenshot_event.event_result(raise_if_any=True, raise_if_none=True)
    if result and isinstance(result, str):
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        save_screenshot_png(result, f'{task_media_dir}/result_{timestamp}.png')

# Define task status enum
class TaskStatus(str, Enum):
    CREATED = "created"  # Task is initialized but not yet started
    RUNNING = "running"  # Task is currently executing
    FINISHED = "finished"  # Task has completed successfully
    STOPPED = "stopped"  # Task was manually stopped
    PAUSED = "paused"  # Task execution is temporarily paused
    FAILED = "failed"  # Task encountered an error and could not complete
    STOPPING = "stopping"  # Task is in the process of stopping (transitional state)

# Load environment variables from .env file
load_dotenv()

# Create media directory if it doesn't exist
MEDIA_DIR = Path("media")
MEDIA_DIR.mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("browser-use-bridge")

app = FastAPI(title="Browser Use Bridge API")

# Mount static files
app.mount("/media", StaticFiles(directory=str(MEDIA_DIR)), name="media")

# Custom JSON encoder for Enum serialization
class EnumJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value
        return super().default(obj)

# Configure FastAPI to use custom JSON serialization for responses
@app.middleware("http")
async def add_json_serialization(request: Request, call_next):
    response = await call_next(request)
    
    # Only attempt to modify JSON responses and check if body() method exists
    if response.headers.get("content-type") == "application/json" and hasattr(
        response, "body"
    ):
        try:
            content = await response.body()
            content_str = content.decode("utf-8")
            content_dict = json.loads(content_str)
            # Convert any Enum values to their string representation
            content_str = json.dumps(content_dict, cls=EnumJSONEncoder)
            response = Response(
                content=content_str,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type="application/json",
            )
        except Exception as e:
            logger.error(f"Error serializing JSON response: {str(e)}")
    
    return response

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize task storage
task_storage = get_task_storage()

async def get_first_websocket_debugger_url():
    url = "http://localhost:9222/json"
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        response.raise_for_status()
        data = response.json()
        if data and isinstance(data, list) and "webSocketDebuggerUrl" in data[0]:
            return data[0]["webSocketDebuggerUrl"]
        else:
            raise RuntimeError("No webSocketDebuggerUrl found in response")


# Models
class TaskRequest(BaseModel):
    task: str
    ai_provider: Optional[str] = os.environ.get(
        "DEFAULT_AI_PROVIDER", "google"  # Default to Gemini instead of OpenAI
    )  
    save_browser_data: Optional[bool] = False  # Whether to save browser cookies
    headful: Optional[bool] = None  # Override BROWSER_USE_HEADFUL setting
    use_custom_chrome: Optional[bool] = (
        None  # Whether to use custom Chrome from env vars
    )


class TaskResponse(BaseModel):
    id: str
    status: str
    live_url: str


class TaskStatusResponse(BaseModel):
    status: str
    result: Optional[str] = None
    error: Optional[str] = None


# Dependency to get user_id from headers
async def get_user_id(x_user_id: Optional[str] = Header(None)) -> str:
    """Extract user ID from header or use default"""
    return x_user_id or DEFAULT_USER_ID


# Utility functions
def get_llm(ai_provider: str):
    """Get LLM based on provider"""
    if ai_provider == "anthropic":
        return ChatAnthropic(
            model=os.environ.get("ANTHROPIC_MODEL_ID", "claude-3-opus-20240229")
        )
    elif ai_provider == "mistral":
        return LLMProvider.MISTRAL(
            model=os.environ.get("MISTRAL_MODEL_ID", "mistral-large-latest")
        )
    elif ai_provider == "google":
        return ChatGoogle(model=os.environ.get("GOOGLE_MODEL_ID", "gemini-2.5-flash"))  # use 2.5 flash here
    elif ai_provider == "ollama":
        return ChatOllama(model=os.environ.get("OLLAMA_MODEL_ID", "llama3"))
    elif ai_provider == "azure":
        return ChatAzureOpenAI(
            model=os.environ.get("AZURE_MODEL_ID", "gpt-4o"),
            azure_deployment=os.environ.get("AZURE_DEPLOYMENT_NAME"),
            api_version=os.environ.get("AZURE_API_VERSION", "2023-05-15"),
            azure_endpoint=os.environ.get("AZURE_ENDPOINT"),
        )
    elif ai_provider == "bedrock":
        return ChatAWSBedrock(
            model=os.environ.get(
                "BEDROCK_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0"
            )
        )
    else:  # default to OpenAI
        base_url = os.environ.get("OPENAI_BASE_URL")
        model = os.environ.get("OPENAI_MODEL_ID", "gpt-4o")

        if base_url:
            return ChatOpenAI(model=model, base_url=base_url)
        else:
            return ChatOpenAI(model=model)


async def execute_task(
    task_id: str, instruction: str, ai_provider: str, user_id: str = DEFAULT_USER_ID
):
    """Execute browser task in background

    Chrome paths (CHROME_PATH and CHROME_USER_DATA) are only sourced from
    environment variables for security reasons.
    """
    # Initialize browser variable outside the try block
    browser = None
    try:
        # Update task status
        task_storage.update_task_status(task_id, TaskStatus.RUNNING, user_id)

        # Get LLM
        llm = get_llm(ai_provider)

        # Get task-specific browser configuration if available
        task = task_storage.get_task(task_id, user_id)
        task_browser_config = task.get("browser_config", {}) if task else {}

        # Create task media directory up front
        task_media_dir = MEDIA_DIR / task_id
        task_media_dir.mkdir(exist_ok=True, parents=True)
        logger.info(f"Created media directory for task {task_id}: {task_media_dir}")

        # Configure browser headless/headful mode (task setting overrides env var)
        task_headful = task_browser_config.get("headful")
        if task_headful is not None:
            headful = task_headful
        else:
            headful = os.environ.get("BROWSER_USE_HEADFUL", "false").lower() == "true"
            browser_config_args = {
                "headless": not headful,
            }

        # Get Chrome path and user data directory (task settings override env vars)
        use_custom_chrome = task_browser_config.get("use_custom_chrome")

        if use_custom_chrome is False:
            # Explicitly disabled custom Chrome for this task
            chrome_path = None
            chrome_user_data = None
        else:
            # Only use environment variables for Chrome paths
            chrome_path = os.environ.get("CHROME_PATH")
            chrome_user_data = os.environ.get("CHROME_USER_DATA")

        sensitive_data = {}
        for key, value in os.environ.items():
            if key.startswith("X_") and value:
                sensitive_data[key] = value

        # Configure agent options - start with basic configuration
        agent_kwargs = {
            "task": instruction,
            "llm": llm,
            "sensitive_data": sensitive_data,
        }

        # Only configure and include browser if we need a custom browser setup
        if not headful or chrome_path:
            extra_chromium_args = []
            # Configure browser
            browser_config_args = {
                "headless": not headful,
                "chrome_instance_path": None,
            }
            # For older Chrome versions
            extra_chromium_args += ["--headless=new"]
            logger.info(
                f"Task {task_id}: Browser config args: {browser_config_args.get('headless')}"
            )
            # Add Chrome executable path if provided
            if chrome_path and chrome_path.lower() != "false":
                browser_config_args["chrome_instance_path"] = chrome_path
                logger.info(
                    f"Task {task_id}: Using custom Chrome executable: {chrome_path}"
                )

            # Add Chrome user data directory if provided
            if chrome_user_data:
                extra_chromium_args += [f"--user-data-dir={chrome_user_data}"]
                logger.info(
                    f"Task {task_id}: Using Chrome user data directory: {chrome_user_data}"
                )

            profile = BrowserProfile(headless=True)  # configure headless or other options here
            cdp_url = await get_first_websocket_debugger_url()
            browser = Browser(browser_profile=profile, cdp_url=cdp_url)

            # Add browser to agent kwargs - let Agent manage its own browser session
            agent_kwargs["browser"] = browser

        logger.info(f"Agent kwargs: {agent_kwargs}")
        # Pass the browser to Agent
        agent = Agent(**agent_kwargs)
        await agent.run(on_step_end=lambda agent_instance: asyncio.create_task(test_capture_screenshot(agent_instance, task_media_dir, task_id))) #test_capture_screenshot

        # Store agent in task
        task_storage.set_task_agent(task_id, agent, user_id)

        # Update finished timestamp and task status
        task_storage.mark_task_finished(task_id, user_id, TaskStatus.FINISHED)

    except Exception as e:
        logger.exception(f"Error executing task {task_id}")
        task_storage.update_task_status(task_id, TaskStatus.FAILED, user_id)
        task_storage.set_task_error(task_id, str(e), user_id)
        task_storage.mark_task_finished(task_id, user_id, TaskStatus.FAILED)
    finally:
        # Always close the browser, regardless of success or failure
        if browser is not None:
            logger.info(f"Closing browser for task {task_id}")
            try:
                logger.info(
                    f"Taking final screenshot for task {task_id} after completion"
                )

                # Get agent to take screenshot
                agent = task_storage.get_task_agent(task_id, user_id)
              
            except Exception as e:
                logger.error(f"Error taking final screenshot: {str(e)}")
            finally:
                if browser:
                    try:
                        await browser.close()
                    except Exception as e:
                        logger.error(
                            f"Error closing browser for task {task_id}: {str(e)}"
                        )


# API Routes
@app.post("/api/v1/run-task", response_model=TaskResponse)
async def run_task(request: TaskRequest, user_id: str = Depends(get_user_id)):
    """Start a browser automation task"""
    task_id = str(uuid.uuid4())
    now = datetime.now(UTC).isoformat() + "Z"

    # Generate live URL
    live_url = f"/live/{task_id}"

    # Initialize task record
    task_data = {
        "id": task_id,
        "task": request.task,
        "ai_provider": request.ai_provider,
        "status": TaskStatus.CREATED,
        "created_at": now,
        "finished_at": None,
        "output": None,  # Final result
        "error": None,
        "steps": [],  # Will store step information
        "agent": None,
        "save_browser_data": request.save_browser_data,
        "browser_data": None,  # Will store browser cookies if requested
        # Store browser configuration options
        "browser_config": {
            "headful": request.headful,
            "use_custom_chrome": request.use_custom_chrome,
        },
        "live_url": live_url,
    }

    # Store the task in storage
    task_storage.create_task(task_id, task_data, user_id)

    # Start task in background
    ai_provider = request.ai_provider or "google"
    asyncio.create_task(execute_task(task_id, request.task, ai_provider, user_id))

    return TaskResponse(id=task_id, status=TaskStatus.CREATED, live_url=live_url)


@app.get("/api/v1/task/{task_id}/status", response_model=TaskStatusResponse)
async def get_task_status(task_id: str, user_id: str = Depends(get_user_id)):
    """Get status of a task"""
    task = task_storage.get_task(task_id, user_id)

    agent = task_storage.get_task_agent(task_id, user_id)

    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    # Only increment steps for running tasks
    if task["status"] == TaskStatus.RUNNING:
        # Initialize steps array if not present
        current_step = len(task.get("steps", [])) + 1

        # Add step info
        step_info = {
            "step": current_step,
            "timestamp": datetime.now(UTC).isoformat() + "Z",
            "next_goal": f"Progress check {current_step}",
            "evaluation_previous_goal": "In progress",
        }

        task_storage.add_task_step(task_id, step_info, user_id)
        logger.info(f"Added step {current_step} for task {task_id}")

    try:
        _ = agent.browser_session
        # await capture_screenshot(agent, task_id, user_id)
    except (AssertionError, AttributeError):
        logger.info(
            f"BrowserSession not ready for task {task_id}, skipping screenshot."
        )

    return TaskStatusResponse(
        status=task["status"],
        result=task.get("output"),
        error=task.get("error"),
    )


@app.get("/api/v1/task/{task_id}", response_model=dict)
async def get_task(task_id: str, user_id: str = Depends(get_user_id)):
    """Get full task details"""
    task = task_storage.get_task(task_id, user_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    return task


@app.put("/api/v1/stop-task/{task_id}")
async def stop_task(task_id: str, user_id: str = Depends(get_user_id)):
    """Stop a running task"""
    task = task_storage.get_task(task_id, user_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    if task["status"] in [
        TaskStatus.FINISHED,
        TaskStatus.FAILED,
        TaskStatus.STOPPED,
    ]:
        return {"message": f"Task already in terminal state: {task['status']}"}

    # Get agent
    agent = task_storage.get_task_agent(task_id, user_id)
    if agent:
        # Call agent's stop method
        agent.stop()
        task_storage.update_task_status(task_id, TaskStatus.STOPPING, user_id)
        return {"message": "Task stopping"}
    else:
        task_storage.update_task_status(task_id, TaskStatus.STOPPED, user_id)
        task_storage.mark_task_finished(task_id, user_id, TaskStatus.STOPPED)
        return {"message": "Task stopped (no agent found)"}


@app.put("/api/v1/pause-task/{task_id}")
async def pause_task(task_id: str, user_id: str = Depends(get_user_id)):
    """Pause a running task"""
    task = task_storage.get_task(task_id, user_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    if task["status"] != TaskStatus.RUNNING:
        return {"message": f"Task not running: {task['status']}"}

    # Get agent
    agent = task_storage.get_task_agent(task_id, user_id)
    if agent:
        # Call agent's pause method
        agent.pause()
        task_storage.update_task_status(task_id, TaskStatus.PAUSED, user_id)
        return {"message": "Task paused"}
    else:
        return {"message": "Task could not be paused (no agent found)"}


@app.put("/api/v1/resume-task/{task_id}")
async def resume_task(task_id: str, user_id: str = Depends(get_user_id)):
    """Resume a paused task"""
    task = task_storage.get_task(task_id, user_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    if task["status"] != TaskStatus.PAUSED:
        return {"message": f"Task not paused: {task['status']}"}

    # Get agent
    agent = task_storage.get_task_agent(task_id, user_id)
    if agent:
        # Call agent's resume method
        agent.resume()
        task_storage.update_task_status(task_id, TaskStatus.RUNNING, user_id)
        return {"message": "Task resumed"}
    else:
        return {"message": "Task could not be resumed (no agent found)"}


@app.get("/api/v1/list-tasks")
async def list_tasks(
    user_id: str = Depends(get_user_id),
    page: int = Query(1, ge=1),
    per_page: int = Query(100, ge=1, le=1000),
):
    """List all tasks"""
    return task_storage.list_tasks(user_id, page, per_page)


@app.get("/live/{task_id}", response_class=HTMLResponse)
async def live_view(task_id: str, user_id: str = Depends(get_user_id)):
    """Get a live view of a task that can be embedded in an iframe"""
    task = task_storage.get_task(task_id, user_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Browser Use Task {task_id}</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .status {{ padding: 10px; border-radius: 4px; margin-bottom: 20px; }}
            .{TaskStatus.RUNNING} {{ background-color: #e3f2fd; }}
            .{TaskStatus.FINISHED} {{ background-color: #e8f5e9; }}
            .{TaskStatus.FAILED} {{ background-color: #ffebee; }}
            .{TaskStatus.PAUSED} {{ background-color: #fff8e1; }}
            .{TaskStatus.STOPPED} {{ background-color: #eeeeee; }}
            .{TaskStatus.CREATED} {{ background-color: #f3e5f5; }}
            .{TaskStatus.STOPPING} {{ background-color: #fce4ec; }}
            .controls {{ margin-bottom: 20px; }}
            button {{ padding: 8px 16px; margin-right: 10px; cursor: pointer; }}
            pre {{ background-color: #f5f5f5; padding: 15px; border-radius: 4px; overflow: auto; }}
            .step {{ margin-bottom: 10px; padding: 10px; border: 1px solid #ddd; border-radius: 4px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Browser Use Task</h1>
            <div id="status" class="status">Loading...</div>
            
            <div class="controls">
                <button id="pauseBtn">Pause</button>
                <button id="resumeBtn">Resume</button>
                <button id="stopBtn">Stop</button>
            </div>
            
            <h2>Result</h2>
            <pre id="result">Loading...</pre>
            
            <h2>Steps</h2>
            <div id="steps">Loading...</div>
            
            <script>
                const taskId = '{task_id}';
                const FINISHED = '{TaskStatus.FINISHED}';
                const FAILED = '{TaskStatus.FAILED}';
                const STOPPED = '{TaskStatus.STOPPED}';
                const userId = '{user_id}';
                
                // Set user ID in request headers if available
                const headers = {{}};
                if (userId && userId !== 'default') {{
                    headers['X-User-ID'] = userId;
                }}
                
                // Update status function
                function updateStatus() {{
                    fetch(`/api/v1/task/${{taskId}}/status`, {{ headers }})
                        .then(response => response.json())
                        .then(data => {{
                            // Update status element
                            const statusEl = document.getElementById('status');
                            statusEl.textContent = `Status: ${{data.status}}`;
                            statusEl.className = `status ${{data.status}}`;
                            
                            // Update result if available
                            if (data.result) {{
                                document.getElementById('result').textContent = data.result;
                            }} else if (data.error) {{
                                document.getElementById('result').textContent = `Error: ${{data.error}}`;
                            }}
                            
                            // Continue polling if not in terminal state
                            if (![FINISHED, FAILED, STOPPED].includes(data.status)) {{
                                setTimeout(updateStatus, 2000);
                            }}
                        }})
                        .catch(error => {{
                            console.error('Error fetching status:', error);
                            setTimeout(updateStatus, 5000);
                        }});
                        
                    // Also fetch full task to get steps
                    fetch(`/api/v1/task/${{taskId}}`, {{ headers }})
                        .then(response => response.json())
                        .then(data => {{
                            if (data.steps && data.steps.length > 0) {{
                                const stepsHtml = data.steps.map(step => `
                                    <div class="step">
                                        <strong>Step ${{step.step}}</strong>
                                        <p>Next Goal: ${{step.next_goal || 'N/A'}}</p>
                                        <p>Evaluation: ${{step.evaluation_previous_goal || 'N/A'}}</p>
                                    </div>
                                `).join('');
                                document.getElementById('steps').innerHTML = stepsHtml;
                            }} else {{
                                document.getElementById('steps').textContent = 'No steps recorded yet.';
                            }}
                        }})
                        .catch(error => {{
                            console.error('Error fetching task details:', error);
                        }});
                }}
                
                // Setup control buttons
                document.getElementById('pauseBtn').addEventListener('click', () => {{
                    fetch(`/api/v1/pause-task/${{taskId}}`, {{ 
                        method: 'PUT',
                        headers
                    }})
                        .then(response => response.json())
                        .then(data => alert(data.message))
                        .catch(error => console.error('Error pausing task:', error));
                }});
                
                document.getElementById('resumeBtn').addEventListener('click', () => {{
                    fetch(`/api/v1/resume-task/${{taskId}}`, {{ 
                        method: 'PUT',
                        headers
                    }})
                        .then(response => response.json())
                        .then(data => alert(data.message))
                        .catch(error => console.error('Error resuming task:', error));
                }});
                
                document.getElementById('stopBtn').addEventListener('click', () => {{
                    if (confirm('Are you sure you want to stop this task? This action cannot be undone.')) {{
                        fetch(`/api/v1/stop-task/${{taskId}}`, {{ 
                            method: 'PUT',
                            headers
                        }})
                            .then(response => response.json())
                            .then(data => alert(data.message))
                            .catch(error => console.error('Error stopping task:', error));
                    }}
                }});
                
                // Start status updates
                updateStatus();
                
                // Refresh every 5 seconds
                setInterval(updateStatus, 5000);
            </script>
        </div>
    </body>
    </html>
    """

    return HTMLResponse(content=html_content)


@app.get("/api/v1/ping")
async def ping():
    """Health check endpoint"""
    return {"status": "success", "message": "API is running"}


@app.get("/api/v1/browser-config")
async def browser_config():
    """Get current browser configuration

    Note: Chrome paths (CHROME_PATH and CHROME_USER_DATA) can only be set via
    environment variables for security reasons and cannot be overridden in task requests.
    """
    headful = os.environ.get("BROWSER_USE_HEADFUL", "false").lower() == "true"
    chrome_path = os.environ.get("CHROME_PATH", None)
    chrome_user_data = os.environ.get("CHROME_USER_DATA", None)

    return {
        "headful": headful,
        "headless": not headful,
        "chrome_path": chrome_path,
        "chrome_user_data": chrome_user_data,
        "using_custom_chrome": chrome_path is not None,
        "using_user_data": chrome_user_data is not None,
    }


@app.get("/api/v1/task/{task_id}/media")
async def get_task_media(
    task_id: str, user_id: str = Depends(get_user_id), type: Optional[str] = None
):
    """Returns links to any recordings or media generated during task execution"""
    task = task_storage.get_task(task_id, user_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    # Check if task is completed
    if task["status"] not in [
        TaskStatus.FINISHED,
        TaskStatus.FAILED,
        TaskStatus.STOPPED,
    ]:
        raise HTTPException(
            status_code=400, detail="Media only available for completed tasks"
        )

    # Check if the media directory exists and contains files
    task_media_dir = MEDIA_DIR / task_id
    media_files = []

    if task_media_dir.exists():
        media_files = list(task_media_dir.glob("*"))
        logger.info(
            f"Media directory for task {task_id} contains {len(media_files)} files: {[f.name for f in media_files]}"
        )
    else:
        logger.warning(f"Media directory for task {task_id} does not exist")

    # If we have files but no media entries, create them now
    if media_files and (not task.get("media") or len(task.get("media", [])) == 0):
        for file_path in media_files:
            if file_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                file_url = f"/media/{task_id}/{file_path.name}"
                media_entry = {
                    "url": file_url,
                    "type": "screenshot",
                    "filename": file_path.name,
                }
                task_storage.add_task_media(task_id, media_entry, user_id)

    # Get updated task with media
    task = task_storage.get_task(task_id, user_id)
    if task is not None:
        media_list = task.get("media", [])
    else:
        media_list = []

    # Filter by type if specified
    if type and isinstance(media_list, list):
        if all(isinstance(item, dict) for item in media_list):
            # Dictionary format with type info
            media_list = [item for item in media_list if item.get("type") == type]
            recordings = [item["url"] for item in media_list]
        else:
            # Just URLs without type info
            recordings = []
            logger.warning(
                f"Media list for task {task_id} doesn't contain type information"
            )
    else:
        # Return all media
        if isinstance(media_list, list):
            if media_list and all(isinstance(item, dict) for item in media_list):
                recordings = [item["url"] for item in media_list]
            else:
                recordings = media_list
        else:
            recordings = []

    logger.info(f"Returning {len(recordings)} media items for task {task_id}")
    return {"recordings": recordings}


@app.get("/api/v1/task/{task_id}/media/list")
async def list_task_media(
    task_id: str, user_id: str = Depends(get_user_id), type: Optional[str] = None
):
    """Returns detailed information about media files associated with a task"""
    # Check if the media directory exists
    task_media_dir = MEDIA_DIR / task_id

    if not task_storage.task_exists(task_id, user_id):
        raise HTTPException(status_code=404, detail="Task not found")

    if not task_media_dir.exists():
        return {
            "media": [],
            "count": 0,
            "message": f"No media found for task {task_id}",
        }

    media_info = []

    media_files = list(task_media_dir.glob("*"))
    logger.info(f"Found {len(media_files)} media files for task {task_id}")

    for file_path in media_files:
        # Determine media type based on file extension
        file_type = "unknown"
        if file_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
            file_type = "screenshot"
        elif file_path.suffix.lower() in [".mp4", ".webm"]:
            file_type = "recording"

        # Get file stats
        stats = file_path.stat()

        file_info = {
            "filename": file_path.name,
            "type": file_type,
            "size_bytes": stats.st_size,
            "created_at": datetime.fromtimestamp(stats.st_ctime).isoformat(),
            "url": f"/media/{task_id}/{file_path.name}",
        }
        media_info.append(file_info)

    # Filter by type if specified
    if type:
        media_info = [item for item in media_info if item["type"] == type]

    logger.info(f"Returning {len(media_info)} media items for task {task_id}")
    return {"media": media_info, "count": len(media_info)}


@app.get("/api/v1/media/{task_id}/{filename}")
async def get_media_file(
    task_id: str,
    filename: str,
    download: bool = Query(
        False, description="Force download instead of viewing in browser"
    ),
):
    """Serve a media file with options for viewing or downloading"""
    # Construct the file path
    file_path = MEDIA_DIR / task_id / filename

    # Check if file exists
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Media file not found")

    # Determine content type
    content_type, _ = mimetypes.guess_type(file_path)

    # Set headers based on download preference
    headers = {}
    if download:
        headers["Content-Disposition"] = f'attachment; filename="{filename}"'
    else:
        headers["Content-Disposition"] = f'inline; filename="{filename}"'

    # Return the file with appropriate headers
    return FileResponse(
        path=file_path, media_type=content_type, headers=headers, filename=filename
    )


@app.get("/api/v1/test-screenshot")
async def test_screenshot(ai_provider: str = "google"):
    """Test endpoint to verify screenshot functionality using the BrowserContext"""
    logger.info(f"Testing screenshot functionality with provider: {ai_provider}")

    browser_service = None
    browser_context = None

    try:
        # Configure browser
        headful = os.environ.get("BROWSER_USE_HEADFUL", "false").lower() == "true"
        browser_config_args = {
            "headless": not headful,
            "chrome_instance_path": None,
        }

        # Add Chrome executable path if provided
        chrome_path = os.environ.get("CHROME_PATH")
        if chrome_path:
            browser_config_args["chrome_instance_path"] = chrome_path

        logger.info(f"Creating browser with config: {browser_config_args}")
        profile = BrowserProfile(headless=False)  # configure headless or other options here
        # browser_service = Browser(browser_profile=profile, cdp_url=steel_cdp_url)
        browser_service = Browser(browser_profile=profile, cdp_url="ws://localhost:9222/devtools/page/735A5092EBC4958A93F90FD3AE84CAC6")

        # Create a BrowserContext instance which has the take_screenshot method
        # browser_context = Browser(browser=browser_service.browser, cdp_url=steel_cdp_url)
        browser_context = Browser(browser=browser_service.browser, cdp_url="ws://localhost:9222/devtools/page/735A5092EBC4958A93F90FD3AE84CAC6")

        # Start the context and navigate to example.com
        async with browser_context:
            logger.info("BrowserContext created, navigating to example.com")
            await browser_context.navigate_to("https://example.com")

            # Now call take_screenshot on the context
            logger.info("Taking screenshot using browser_context.take_screenshot")
            screenshot_b64 = await browser_context.take_screenshot(full_page=True)

            if not screenshot_b64:
                return {"error": "Screenshot returned None or empty string"}

            logger.info(f"Screenshot captured: {len(screenshot_b64)} bytes")

            # Create test directory and save screenshot
            test_dir = MEDIA_DIR / "test"
            test_dir.mkdir(exist_ok=True, parents=True)
            screenshot_path = test_dir / "test_screenshot.png"

            try:
                # Decode and save screenshot
                image_data = base64.b64decode(screenshot_b64)
                logger.info(f"Decoded base64 data: {len(image_data)} bytes")

                with open(screenshot_path, "wb") as f:
                    f.write(image_data)

                if screenshot_path.exists():
                    file_size = screenshot_path.stat().st_size
                    logger.info(
                        f"Screenshot saved: {screenshot_path} ({file_size} bytes)"
                    )

                    return {
                        "success": True,
                        "message": "Screenshot captured and saved successfully",
                        "file_size": file_size,
                        "file_path": str(screenshot_path),
                        "url": f"/media/test/test_screenshot.png",
                        "working_method": "browser_context.take_screenshot",
                    }
                else:
                    return {"error": "File not created"}
            except Exception as e:
                logger.exception("Error saving screenshot")
                return {"error": f"Error saving screenshot: {str(e)}"}
    except Exception as e:
        logger.exception("Error in screenshot test")
        return {"error": f"Test failed: {str(e)}"}
    finally:
        # Clean up resources
        if browser_context:
            try:
                await browser_context.close()
            except Exception as e:
                logger.warning(f"Error closing browser context: {str(e)}")

        if browser_service:
            try:
                await browser_service.close()
            except Exception as e:
                logger.warning(f"Error closing browser: {str(e)}")


# Run server if executed directly
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
