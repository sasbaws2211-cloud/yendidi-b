from fastapi import APIRouter, FastAPI
from sqlmodel import SQLModel
from middleware import register_middleware
from app.api import (
    accounts,
    admin,
    auth,
    catalog,
    favorites,
    guardian_links,
    kiosk,
    notifications,
    orders,
    parents,
    reports,
    schools,
    subscriptions,
    vendor_submissions,
    wallet,
    webhooks,
)
from app.db.session import engine

app = FastAPI(title="Y3ndidi Backend", 
              version="0.1.0",
              docs_url="/v1/docs"

      )

register_middleware(app)
router = APIRouter(prefix="/v1")

router.include_router(auth.router, prefix="/auth", tags=["auth"])
router.include_router(catalog.router, prefix="", tags=["catalog"])
router.include_router(orders.router, prefix="", tags=["orders"])
router.include_router(wallet.router, prefix="", tags=["wallet"])
router.include_router(parents.router, prefix="", tags=["parents"])
router.include_router(schools.router, prefix="", tags=["schools"])
router.include_router(reports.router, prefix="", tags=["reports"])
router.include_router(notifications.router, prefix="", tags=["notifications"])
router.include_router(subscriptions.router, prefix="", tags=["subscriptions"])
router.include_router(accounts.router, prefix="", tags=["accounts"])
router.include_router(vendor_submissions.router, prefix="", tags=["vendor-submissions"])
router.include_router(favorites.router, prefix="", tags=["favorites"])
router.include_router(admin.router, prefix="", tags=["admin"])
router.include_router(guardian_links.router, prefix="", tags=["guardian-links"])
router.include_router(kiosk.router, prefix="", tags=["kiosk"])
router.include_router(webhooks.router, prefix="/webhooks", tags=["webhooks"])

app.include_router(router)


@app.on_event("startup")
async def on_startup() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)


@app.on_event("shutdown")
async def on_shutdown() -> None:
    await engine.dispose()
