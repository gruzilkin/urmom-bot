import os
import logging
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
    service_name="joke-admin-web",
    endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "localhost:4317")
)

# Initialize store
store = WebStore(
    telemetry=telemetry,
    host=os.getenv("POSTGRES_HOST", "localhost"),
    port=int(os.getenv("POSTGRES_PORT", "5432")),
    user=os.getenv("POSTGRES_USER", "postgres"),
    password=os.getenv("POSTGRES_PASSWORD", "postgres"),
    database=os.getenv("POSTGRES_DB", "urmom")
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

@app.get("/", response_class=HTMLResponse)
async def index(request: Request, page: int = 1, search: str = ""):
    """Main joke management page with pagination and search"""
    try:
        page_size = 20
        offset = (page - 1) * page_size
        
        search_query = search.strip()
        jokes = await store.get_jokes(limit=page_size, offset=offset, search_query=search_query)
        total_count = await store.get_jokes_count(search_query=search_query)
        
        total_pages = (total_count + page_size - 1) // page_size
        
        return templates.TemplateResponse("index.html", {
            "request": request,
            "jokes": jokes,
            "current_page": page,
            "total_pages": total_pages,
            "search": search,
            "total_count": total_count
        })
    except Exception as e:
        logger.error(f"Error loading jokes page: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error loading jokes")

@app.post("/edit_message/{message_id}")
async def edit_message(request: Request, message_id: int, content: str = Form(...)):
    """Edit message content via HTMX"""
    try:
        success = await store.update_message_content(message_id, content)
        if success:
            return templates.TemplateResponse("partials/editable_content.html", {
                "request": request,
                "content": content,
                "message_id": message_id
            })
        else:
            raise HTTPException(status_code=400, detail="Failed to update message")
    except Exception as e:
        logger.error(f"Error updating message {message_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error updating message")

@app.get("/edit_message_form/{message_id}")
async def edit_message_form(request: Request, message_id: int, current_content: str):
    """Return edit form for message"""
    from urllib.parse import unquote
    decoded_content = unquote(current_content)
    
    return templates.TemplateResponse("partials/edit_form.html", {
        "request": request,
        "content": decoded_content,
        "message_id": message_id
    })

@app.get("/cancel_edit_message/{message_id}")
async def cancel_edit_message(request: Request, message_id: int):
    """Cancel edit and return to display mode"""
    try:
        content = await store.get_message_content(message_id)
        if content is None:
            raise HTTPException(status_code=404, detail="Message not found")
            
        return templates.TemplateResponse("partials/editable_content.html", {
            "request": request,
            "content": content,
            "message_id": message_id
        })
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

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up database connections on shutdown"""
    await store.close()

if __name__ == "__main__":
    port = int(os.getenv("PORT", "80"))
    uvicorn.run(app, host="0.0.0.0", port=port)