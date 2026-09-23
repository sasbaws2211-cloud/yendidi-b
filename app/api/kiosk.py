import secrets
from datetime import datetime, timedelta
from typing import List, Optional
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, Header, HTTPException, status
from pydantic import BaseModel
from sqlmodel import select

from app.api.orders import OrderCreateResponse, OrderPreviewRequest, execute_order
from app.core.dependencies import (
    get_current_kiosk,
    get_current_user,
    get_session,
    rate_limiter,
)
from app.core.security import hash_token
from app.db.models import FaceTemplate, FoodCategory, Kiosk, KioskStatus, KioskVerification, Role, School, Student, User, Wallet
from app.db.session import AsyncSession

router = APIRouter()

# `/kiosks/verify` is called by an unattended device, not a signed-in person,
# and its body is a guessable student code — the closest thing this API has
# to a login form for a stranger. `/kiosks/pair` guards a one-time code with
# the same shape. Both get the anonymous, high-value rate limiter.
VERIFY_RATE_LIMIT = rate_limiter(max_requests=30, window_seconds=60)
PAIR_RATE_LIMIT = rate_limiter(max_requests=10, window_seconds=60)
# A 4-8 digit PIN has too few combinations to leave unlimited from a device
# that already holds a valid token — 5 tries a minute is enough for a fumbled
# entry and too little for a search.
EXIT_RATE_LIMIT = rate_limiter(max_requests=5, window_seconds=60)

PAIRING_CODE_TTL_MINUTES = 15
VERIFICATION_TOKEN_TTL_SECONDS = 60


def _generate_code() -> str:
    return secrets.token_hex(4).upper()


def _generate_token() -> str:
    return secrets.token_urlsafe(32)


def _validate_exit_pin(pin: str) -> str:
    cleaned = pin.strip()
    if not cleaned.isdigit() or not (4 <= len(cleaned) <= 8):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Exit PIN must be 4-8 digits")
    return cleaned


async def _authorize_school_scope(school_id: UUID, current_user: User) -> None:
    if current_user.role == Role.super_admin:
        return
    if current_user.role == Role.school_admin and current_user.school_id == school_id:
        return
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized for this school")


class KioskCreateRequest(BaseModel):
    school_id: UUID
    label: str
    exit_pin: str


class KioskCreateResponse(BaseModel):
    id: UUID
    school_id: UUID
    label: str
    status: KioskStatus
    pairing_code: str
    pairing_code_expires_at: datetime


class KioskResponse(BaseModel):
    id: UUID
    school_id: UUID
    label: str
    status: KioskStatus
    paired_at: Optional[datetime]
    last_seen_at: Optional[datetime]
    created_at: datetime

    class Config:
        from_attributes = True


@router.post("/kiosks", response_model=KioskCreateResponse, status_code=status.HTTP_201_CREATED)
async def create_kiosk(
    request: KioskCreateRequest,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    if current_user.role != Role.school_admin and current_user.role != Role.super_admin:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only school admins may provision a kiosk")

    exit_pin = _validate_exit_pin(request.exit_pin)
    code = _generate_code()
    kiosk = Kiosk(
        id=uuid4(),
        school_id=request.school_id,
        label=request.label.strip(),
        status=KioskStatus.pending,
        pairing_code_hash=hash_token(code),
        pairing_code_expires_at=datetime.utcnow() + timedelta(minutes=PAIRING_CODE_TTL_MINUTES),
        exit_pin_hash=hash_token(exit_pin),
    )
    session.add(kiosk)
    await session.commit()
    await session.refresh(kiosk)
    return KioskCreateResponse(
        id=kiosk.id,
        school_id=kiosk.school_id,
        label=kiosk.label,
        status=kiosk.status,
        pairing_code=code,
        pairing_code_expires_at=kiosk.pairing_code_expires_at,
    )


class KioskPairRequest(BaseModel):
    pairing_code: str


class KioskPairResponse(BaseModel):
    device_token: str
    kiosk_id: UUID
    school_id: UUID
    school_name: str


@router.post("/kiosks/pair", response_model=KioskPairResponse)
async def pair_kiosk(
    request: KioskPairRequest,
    session: AsyncSession = Depends(get_session),
    _rl=Depends(PAIR_RATE_LIMIT),
):
    statement = select(Kiosk).where(Kiosk.pairing_code_hash == hash_token(request.pairing_code.strip()))
    result = await session.execute(statement)
    kiosk = result.scalar_one_or_none()
    if not kiosk or not kiosk.pairing_code_expires_at or kiosk.pairing_code_expires_at < datetime.utcnow():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid or expired pairing code")

    device_token = _generate_token()
    kiosk.device_token_hash = hash_token(device_token)
    kiosk.status = KioskStatus.active
    kiosk.paired_at = datetime.utcnow()
    kiosk.last_seen_at = datetime.utcnow()
    # Spent on first use — a pairing code is a bearer secret good for exactly
    # one device until this clears it.
    kiosk.pairing_code_hash = None
    kiosk.pairing_code_expires_at = None
    session.add(kiosk)
    await session.commit()

    # Denormalised onto the response rather than left for the client to fetch
    # separately: a kiosk token cannot call `GET /schools/{id}` (that route is
    # school_admin/super_admin only), so this is the only way the device ever
    # learns its own school's name.
    school = (await session.execute(select(School).where(School.id == kiosk.school_id))).scalar_one_or_none()

    return KioskPairResponse(
        device_token=device_token,
        kiosk_id=kiosk.id,
        school_id=kiosk.school_id,
        school_name=school.name if school else "",
    )


@router.get("/schools/{school_id}/kiosks", response_model=List[KioskResponse])
async def list_school_kiosks(
    school_id: UUID,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    await _authorize_school_scope(school_id, current_user)
    statement = select(Kiosk).where(Kiosk.school_id == school_id).order_by(Kiosk.created_at.desc())
    result = await session.execute(statement)
    return result.scalars().all()


@router.post("/kiosks/{kiosk_id}/revoke", response_model=KioskResponse)
async def revoke_kiosk(
    kiosk_id: UUID,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    statement = select(Kiosk).where(Kiosk.id == kiosk_id)
    result = await session.execute(statement)
    kiosk = result.scalar_one_or_none()
    if not kiosk:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Kiosk not found")
    await _authorize_school_scope(kiosk.school_id, current_user)

    # A tablet walking out of a corridor at 2pm needs to stop working
    # immediately, not at the next token expiry — clear the hash as well as
    # flipping status so a leaked hash from a backup can't be replayed either.
    kiosk.status = KioskStatus.revoked
    kiosk.revoked_at = datetime.utcnow()
    kiosk.device_token_hash = None
    session.add(kiosk)
    await session.commit()
    await session.refresh(kiosk)
    return kiosk


class KioskExitRequest(BaseModel):
    pin: str


class KioskExitResponse(BaseModel):
    ok: bool = True


@router.post("/kiosks/exit", response_model=KioskExitResponse)
async def exit_kiosk(
    request: KioskExitRequest,
    kiosk: Kiosk = Depends(get_current_kiosk),
    session: AsyncSession = Depends(get_session),
    _rl=Depends(EXIT_RATE_LIMIT),
):
    """Lets whoever is standing at a paired device take it out of service
    without an admin's JWT, using the PIN set when the kiosk was provisioned
    (SPEC_KIOSK_AND_VERIFICATION.md §2.1). Ends the kiosk exactly like
    `POST /kiosks/{id}/revoke` — same status, same cleared token — because the
    PIN is device-held and shared among on-site staff, not a personal
    credential worth a softer outcome, and re-entering service needs a new
    kiosk from an admin either way, same as after a revoke.
    """
    if not kiosk.exit_pin_hash or hash_token(request.pin.strip()) != kiosk.exit_pin_hash:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="That passcode is not right.")

    kiosk.status = KioskStatus.revoked
    kiosk.revoked_at = datetime.utcnow()
    kiosk.device_token_hash = None
    kiosk.exit_pin_hash = None
    session.add(kiosk)
    await session.commit()
    return KioskExitResponse(ok=True)


async def _remaining_spend_response_fields(student_id: UUID, wallet: Wallet, session: AsyncSession) -> tuple[Optional[int], Optional[int]]:
    from app.api.orders import _sum_purchase_spend  # local import: internal helper, not part of orders.py's public surface

    today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    week_start = today - timedelta(days=today.weekday())

    daily_remaining = None
    if wallet.daily_limit_minor is not None:
        spent_today = await _sum_purchase_spend(student_id, today, session)
        daily_remaining = max(0, wallet.daily_limit_minor - spent_today)

    weekly_remaining = None
    if wallet.weekly_limit_minor is not None:
        spent_week = await _sum_purchase_spend(student_id, week_start, session)
        weekly_remaining = max(0, wallet.weekly_limit_minor - spent_week)

    return daily_remaining, weekly_remaining


class KioskVerifyRequest(BaseModel):
    student_code: str


class KioskVerifyResponse(BaseModel):
    verification_token: str
    student_id: UUID
    first_name: str
    last_name: str
    class_name: str
    student_code: str
    allergies: List[str]
    dietary_notes: Optional[str]
    balance_minor: int
    daily_remaining_minor: Optional[int]
    weekly_remaining_minor: Optional[int]
    blocked_categories: List[FoodCategory]
    wallet_frozen: bool


async def _build_verify_response(student: Student, kiosk: Kiosk, session: AsyncSession) -> KioskVerifyResponse:
    """The common tail of both verify paths — code and face — once a
    specific student has been identified: mint the one-shot token, spend
    nothing yet, and return exactly what completing this one purchase
    requires. Nothing here should let the token double as a general pupil
    record.
    """
    wallet_stmt = select(Wallet).where(Wallet.student_id == student.id)
    wallet_result = await session.execute(wallet_stmt)
    wallet = wallet_result.scalar_one_or_none()
    if not wallet:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No match found")

    daily_remaining, weekly_remaining = await _remaining_spend_response_fields(student.id, wallet, session)

    token = _generate_token()
    session.add(
        KioskVerification(
            id=uuid4(),
            token_hash=hash_token(token),
            kiosk_id=kiosk.id,
            student_id=student.id,
            expires_at=datetime.utcnow() + timedelta(seconds=VERIFICATION_TOKEN_TTL_SECONDS),
        )
    )
    await session.commit()

    return KioskVerifyResponse(
        verification_token=token,
        student_id=student.id,
        first_name=student.first_name,
        last_name=student.last_name,
        class_name=student.class_name,
        student_code=student.student_code,
        allergies=student.allergies,
        dietary_notes=student.dietary_notes,
        balance_minor=wallet.balance_minor,
        daily_remaining_minor=daily_remaining,
        weekly_remaining_minor=weekly_remaining,
        blocked_categories=wallet.blocked_categories,
        wallet_frozen=wallet.frozen,
    )


@router.post("/kiosks/verify", response_model=KioskVerifyResponse)
async def verify_student(
    request: KioskVerifyRequest,
    kiosk: Kiosk = Depends(get_current_kiosk),
    session: AsyncSession = Depends(get_session),
    _rl=Depends(VERIFY_RATE_LIMIT),
):
    """Turns an anonymous kiosk cart into a named payer (SPEC_KIOSK_AND_VERIFICATION.md §3).

    An unknown code and a code belonging to another school must be
    indistinguishable, or whoever is standing at the kiosk can use it to probe
    which codes exist — both fall through to the same 404 below.
    """
    statement = select(Student).where(Student.student_code == request.student_code.strip())
    result = await session.execute(statement)
    student = result.scalar_one_or_none()
    if not student or student.school_id != kiosk.school_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No match found")

    return await _build_verify_response(student, kiosk, session)


# Cosine similarity a live capture must clear against an enrolled template to
# count as a match. Deliberately conservative pending real calibration data —
# there is no live face traffic yet to tune this against, and the cost of
# being too strict (a pupil falls back to their card, which is always a
# first-class path — see verification.ts on the client) is far lower than
# the cost of being too loose (someone else's face authorises a purchase).
FACE_MATCH_THRESHOLD = 0.55


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class KioskVerifyFaceRequest(BaseModel):
    # A live capture's embedding, computed on the kiosk device — never a
    # photo, and this endpoint has nowhere to store one if it were sent.
    embedding: List[float]
    model_version: str


@router.post("/kiosks/verify-face", response_model=KioskVerifyResponse)
async def verify_student_by_face(
    request: KioskVerifyFaceRequest,
    kiosk: Kiosk = Depends(get_current_kiosk),
    session: AsyncSession = Depends(get_session),
    _rl=Depends(VERIFY_RATE_LIMIT),
):
    """Face is the primary, code-free path (SPEC_KIOSK_AND_VERIFICATION.md §4):
    a pupil looks at the camera and is identified, rather than typing
    anything first — so this searches every pupil enrolled at this kiosk's
    school, not one already-named candidate.

    The match itself happens here, never on the device. A kiosk is an
    unattended device in a corridor — the same assumption the `Kiosk` model
    itself is built on — and a kiosk trusted to decide "yes, this matches"
    on its own can be modified to say yes to everyone. This is the one place
    that decision is made, the same way a wallet balance or a spend limit is
    never something a kiosk is trusted to report about itself.

    No match reaches the same generic refusal an unknown student code does
    (SPEC §3) — a device in a corridor should not be able to tell "no pupil
    looks like this" apart from "that code doesn't exist."
    """
    templates_stmt = (
        select(FaceTemplate, Student)
        .join(Student, FaceTemplate.student_id == Student.id)
        .where(Student.school_id == kiosk.school_id, FaceTemplate.model_version == request.model_version)
    )
    templates_result = await session.execute(templates_stmt)
    candidates = templates_result.all()

    best_student: Optional[Student] = None
    best_score = FACE_MATCH_THRESHOLD
    for template, student in candidates:
        score = _cosine_similarity(request.embedding, template.embedding)
        if score >= best_score:
            best_score = score
            best_student = student

    if not best_student:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No match found")

    return await _build_verify_response(best_student, kiosk, session)


@router.post("/kiosks/orders", response_model=OrderCreateResponse)
async def place_kiosk_order(
    request: OrderPreviewRequest,
    verification_token: str = Header(..., alias="X-Verification-Token"),
    idempotency_key: Optional[str] = Header(None, alias="idempotency-key"),
    kiosk: Kiosk = Depends(get_current_kiosk),
    session: AsyncSession = Depends(get_session),
):
    """The kiosk side of order placement (SPEC §9.4). The verify response
    tells the kiosk what to *display* — it is not permission to spend. Every
    balance, limit, category and stock check in `execute_order` still runs
    server-side against the actual student the token names.
    """
    if not idempotency_key:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Idempotency-Key header required")

    token_stmt = select(KioskVerification).where(KioskVerification.token_hash == hash_token(verification_token))
    token_result = await session.execute(token_stmt)
    verification = token_result.scalar_one_or_none()
    if (
        not verification
        or verification.kiosk_id != kiosk.id
        or verification.used_at is not None
        or verification.expires_at < datetime.utcnow()
    ):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid, expired, or already-used verification token")

    student_stmt = select(Student).where(Student.id == verification.student_id)
    student_result = await session.execute(student_stmt)
    student = student_result.scalar_one_or_none()
    if not student:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Student not found")

    # Spent immediately, before the order logic runs, so a failed order
    # (insufficient balance, vendor closed) cannot be retried against the
    # same token — the payer has to be re-verified, matching a token's
    # single-transaction scope.
    verification.used_at = datetime.utcnow()
    session.add(verification)
    await session.commit()

    return await execute_order(student, request, idempotency_key, None, session)
