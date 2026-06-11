import os
import logging
from datetime import datetime, timezone as dt_timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from croniter import croniter, CroniterBadCronError
from cron_descriptor import get_description
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn

from .store import WebStore
from .telemetry import SimpleTelemetry

# OpenTelemetry FastAPI instrumentation
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize telemetry
telemetry = SimpleTelemetry(
    service_name="joke-admin-web", endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "localhost:4317")
)

# Initialize store
store = WebStore(
    telemetry=telemetry,
    host=os.getenv("POSTGRES_HOST", "localhost"),
    port=int(os.getenv("POSTGRES_PORT", "5432")),
    user=os.getenv("POSTGRES_USER", "postgres"),
    password=os.getenv("POSTGRES_PASSWORD", "postgres"),
    database=os.getenv("POSTGRES_DB", "urmom"),
)

# Initialize FastAPI app
app = FastAPI(title="Joke Management Interface", version="1.0.0")

# Instrument FastAPI with OpenTelemetry
FastAPIInstrumentor.instrument_app(app)

# Get the directory where this script is located
current_dir = os.path.dirname(__file__)
# Get the parent directory (app root where static and templates are)
app_root = os.path.dirname(current_dir)

# Mount static files
app.mount("/static", StaticFiles(directory=os.path.join(app_root, "static")), name="static")

# Initialize templates
templates = Jinja2Templates(directory=os.path.join(app_root, "templates"))


def describe_cron(cron_expression: str | None) -> str:
    """Render a cron expression as human-readable text. Returns 'One-off' for null cron."""
    if not cron_expression:
        return "One-off"
    try:
        return get_description(cron_expression)
    except Exception:
        return cron_expression


def render_in_timezone(dt: datetime | None, tz_name: str) -> str:
    """Render a UTC datetime in the given IANA timezone. Returns '—' for None."""
    if dt is None:
        return "—"
    try:
        tz = ZoneInfo(tz_name)
        return dt.astimezone(tz).strftime("%Y-%m-%d %H:%M %Z")
    except ZoneInfoNotFoundError:
        return dt.isoformat()


templates.env.filters["describe_cron"] = describe_cron
templates.env.filters["in_tz"] = render_in_timezone


@app.get("/")
async def root():
    return RedirectResponse(url="/jokes", status_code=308)


@app.get("/jokes", response_class=HTMLResponse)
async def jokes(request: Request, page: int = 1, search: str = ""):
    """Main joke management page with pagination and search"""
    try:
        page_size = 20
        offset = (page - 1) * page_size

        search_query = search.strip()
        jokes = await store.get_jokes(limit=page_size, offset=offset, search_query=search_query)
        total_count = await store.get_jokes_count(search_query=search_query)

        total_pages = (total_count + page_size - 1) // page_size

        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "jokes": jokes,
                "current_page": page,
                "total_pages": total_pages,
                "search": search,
                "total_count": total_count,
            },
        )
    except Exception as e:
        logger.error(f"Error loading jokes page: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error loading jokes")


@app.post("/edit_message/{message_id}")
async def edit_message(request: Request, message_id: int, content: str = Form(...)):
    """Edit message content via HTMX"""
    try:
        success = await store.update_message_content(message_id, content)
        if success:
            return templates.TemplateResponse(
                request, "partials/editable_content.html", {"content": content, "message_id": message_id}
            )
        else:
            raise HTTPException(status_code=400, detail="Failed to update message")
    except Exception as e:
        logger.error(f"Error updating message {message_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error updating message")


@app.get("/edit_message_form/{message_id}")
async def edit_message_form(request: Request, message_id: int):
    """Return edit form for message"""
    try:
        content = await store.get_message_content(message_id)
        if content is None:
            raise HTTPException(status_code=404, detail="Message not found")

        return templates.TemplateResponse(
            request, "partials/edit_form.html", {"content": content, "message_id": message_id}
        )
    except Exception as e:
        logger.error(f"Error getting edit form for message {message_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error getting edit form")


@app.get("/cancel_edit_message/{message_id}")
async def cancel_edit_message(request: Request, message_id: int):
    """Cancel edit and return to display mode"""
    try:
        content = await store.get_message_content(message_id)
        if content is None:
            raise HTTPException(status_code=404, detail="Message not found")

        return templates.TemplateResponse(
            request, "partials/editable_content.html", {"content": content, "message_id": message_id}
        )
    except Exception as e:
        logger.error(f"Error canceling edit for message {message_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error canceling edit")


@app.delete("/delete/{source_message_id}/{joke_message_id}")
async def delete_joke(source_message_id: int, joke_message_id: int):
    """Delete a joke pair via HTMX"""
    try:
        success = await store.delete_joke(source_message_id, joke_message_id)
        if success:
            return HTMLResponse("")  # Empty response removes the row
        else:
            raise HTTPException(status_code=400, detail="Failed to delete joke")
    except Exception as e:
        logger.error(f"Error deleting joke {source_message_id}/{joke_message_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error deleting joke")


@app.get("/users", response_class=HTMLResponse)
async def users(request: Request, page: int = 1, search: str = ""):
    """User facts management page with pagination and search"""
    try:
        page_size = 20
        offset = (page - 1) * page_size

        search_query = search.strip()
        rows = await store.get_user_facts_rows(limit=page_size, offset=offset, search_query=search_query)
        total_count = await store.get_user_facts_count(search_query=search_query)

        total_pages = (total_count + page_size - 1) // page_size

        return templates.TemplateResponse(
            request,
            "users.html",
            {
                "rows": rows,
                "current_page": page,
                "total_pages": total_pages,
                "search": search,
                "total_count": total_count,
            },
        )
    except Exception as e:
        logger.error(f"Error loading user facts page: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error loading user facts")


@app.get("/edit_user_facts_form/{guild_id}/{user_id}")
async def edit_user_facts_form(request: Request, guild_id: int, user_id: int):
    """Return edit form for user facts row"""
    try:
        memory_blob = await store.get_user_facts_blob(guild_id, user_id)
        if memory_blob is None:
            raise HTTPException(status_code=404, detail="User facts not found")

        return templates.TemplateResponse(
            request,
            "partials/user_facts_edit_form.html",
            {"memory_blob": memory_blob, "guild_id": guild_id, "user_id": user_id},
        )
    except Exception as e:
        logger.error(f"Error getting edit form for user facts {guild_id}/{user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error getting edit form")


@app.get("/cancel_edit_user_facts/{guild_id}/{user_id}")
async def cancel_edit_user_facts(request: Request, guild_id: int, user_id: int):
    """Cancel edit and return to display mode"""
    try:
        memory_blob = await store.get_user_facts_blob(guild_id, user_id)
        if memory_blob is None:
            raise HTTPException(status_code=404, detail="User facts not found")

        return templates.TemplateResponse(
            request,
            "partials/user_facts_editable_content.html",
            {"memory_blob": memory_blob, "guild_id": guild_id, "user_id": user_id},
        )
    except Exception as e:
        logger.error(f"Error canceling edit for user facts {guild_id}/{user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error canceling edit")


@app.post("/edit_user_facts/{guild_id}/{user_id}")
async def edit_user_facts(request: Request, guild_id: int, user_id: int, memory_blob: str = Form(...)):
    """Edit user facts memory_blob via HTMX"""
    try:
        success = await store.update_user_facts_blob(guild_id, user_id, memory_blob)
        if success:
            return templates.TemplateResponse(
                request,
                "partials/user_facts_editable_content.html",
                {"memory_blob": memory_blob, "guild_id": guild_id, "user_id": user_id},
            )
        else:
            raise HTTPException(status_code=400, detail="Failed to update user facts")
    except Exception as e:
        logger.error(f"Error updating user facts {guild_id}/{user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error updating user facts")


@app.delete("/delete_user_facts/{guild_id}/{user_id}")
async def delete_user_facts(guild_id: int, user_id: int):
    """Delete a user facts row via HTMX"""
    try:
        success = await store.delete_user_facts(guild_id, user_id)
        if success:
            return HTMLResponse("")  # Empty response removes the row
        else:
            raise HTTPException(status_code=400, detail="Failed to delete user facts")
    except Exception as e:
        logger.error(f"Error deleting user facts {guild_id}/{user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error deleting user facts")


@app.get("/tasks", response_class=HTMLResponse)
async def tasks(request: Request, page: int = 1, search: str = ""):
    """Scheduled task management page."""
    try:
        page_size = 20
        offset = (page - 1) * page_size

        search_query = search.strip()
        rows = await store.get_scheduled_tasks_rows(limit=page_size, offset=offset, search_query=search_query)
        total_count = await store.get_scheduled_tasks_count(search_query=search_query)

        total_pages = (total_count + page_size - 1) // page_size

        return templates.TemplateResponse(
            request,
            "tasks.html",
            {
                "rows": rows,
                "current_page": page,
                "total_pages": total_pages,
                "search": search,
                "total_count": total_count,
            },
        )
    except Exception as e:
        logger.error(f"Error loading scheduled tasks page: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error loading scheduled tasks")


@app.get("/edit_task_form/{task_id}")
async def edit_task_form(request: Request, task_id: int):
    """Return edit form for a scheduled task."""
    try:
        task = await store.get_scheduled_task(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="Task not found")

        return templates.TemplateResponse(
            request,
            "partials/task_edit_form.html",
            {"task": task},
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting edit form for task {task_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error getting edit form")


@app.get("/cancel_edit_task/{task_id}")
async def cancel_edit_task(request: Request, task_id: int):
    """Return to display mode without saving."""
    try:
        task = await store.get_scheduled_task(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="Task not found")

        return templates.TemplateResponse(
            request,
            "partials/task_editable_content.html",
            {"task": task},
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error canceling edit for task {task_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error canceling edit")


@app.post("/edit_task/{task_id}")
async def edit_task(
    request: Request,
    task_id: int,
    prompt: str = Form(...),
    cron_expression: str = Form(""),
    timezone_name: str = Form(...),
):
    """Save edits to a scheduled task. Recomputes next_run_at only when the schedule changes."""
    try:
        existing = await store.get_scheduled_task(task_id)
        if existing is None:
            raise HTTPException(status_code=404, detail="Task not found")

        cron_value = cron_expression.strip() or None
        tz_name = timezone_name.strip()

        # Validate timezone
        try:
            tz = ZoneInfo(tz_name)
        except ZoneInfoNotFoundError:
            raise HTTPException(status_code=400, detail=f"Invalid IANA timezone: {tz_name}")

        # Preserve next_run_at unless cron or timezone changed — recomputing on a
        # prompt-only edit would clobber a one-off's stored firing time or an
        # explicitly anchored first run of a recurring task (mirrors bot-side _edit).
        schedule_unchanged = cron_value == existing.cron_expression and tz_name == existing.timezone
        if schedule_unchanged or not cron_value:
            next_run_at = existing.next_run_at
        else:
            try:
                now_in_tz = datetime.now(tz)
                itr = croniter(cron_value, now_in_tz)
                next_run_at = itr.get_next(datetime).astimezone(dt_timezone.utc)
            except (CroniterBadCronError, ValueError) as e:
                raise HTTPException(status_code=400, detail=f"Invalid cron expression: {e}")

        success = await store.update_scheduled_task(
            task_id=task_id,
            prompt=prompt,
            cron_expression=cron_value,
            timezone=tz_name,
            next_run_at=next_run_at,
        )
        if not success:
            raise HTTPException(status_code=400, detail="Failed to update task")

        # Re-fetch to get fresh row (includes updated_at and next_run_at)
        updated = await store.get_scheduled_task(task_id)
        return templates.TemplateResponse(
            request,
            "partials/task_editable_content.html",
            {"task": updated},
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating task {task_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error updating task")


@app.delete("/delete_scheduled_task/{task_id}")
async def delete_scheduled_task(task_id: int):
    """Delete a scheduled task via HTMX."""
    try:
        success = await store.delete_scheduled_task(task_id)
        if success:
            return HTMLResponse("")
        raise HTTPException(status_code=400, detail="Failed to delete task")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting task {task_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error deleting task")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up database connections on shutdown"""
    await store.close()


if __name__ == "__main__":
    port = int(os.getenv("PORT", "80"))
    uvicorn.run(app, host="0.0.0.0", port=port)
